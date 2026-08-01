"""pgvector benchmark: ingest + HNSW build, recall-latency sweep, concurrency, filters.

Usage:
  python bench_pgvector.py <scale> ingest
  python bench_pgvector.py <scale> query
  python bench_pgvector.py <scale> concurrent
  python bench_pgvector.py <scale> filtered
"""
import argparse
import concurrent.futures as cf
import subprocess
import time

import numpy as np
import psycopg
import pyarrow.parquet as pq
from tqdm import tqdm

from common import (DIM, FILTERS, K, PG_DSN, Timer, load_base, load_gt,
                    load_queries, paths, percentiles, recall_at_k, save_result)

HNSW = {"m": 16, "ef_construction": 64}
EF_SEARCH = [10, 20, 40, 80, 160, 320]
CONCURRENCY = 8
CONC_EF = 80  # ~0.95 recall 帯 (query 軸の結果を見て要調整)


def vec_literal(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def server_version(cur):
    cur.execute("SELECT version()")
    pg = cur.fetchone()[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
    return {"postgres": pg, "pgvector": cur.fetchone()[0]}


def do_ingest(scale):
    base = np.asarray(load_base(scale))
    meta = pq.read_table(paths(scale)["meta"])
    cat1, cat10 = meta["cat1pct"].to_numpy(), meta["cat10pct"].to_numpy()

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        cur = conn.cursor()
        ver = server_version(cur)
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(f"""CREATE TABLE items (
            id bigint PRIMARY KEY, cat1pct boolean, cat10pct boolean,
            embedding vector({DIM}))""")

        with Timer() as t_copy:
            with cur.copy("COPY items (id, cat1pct, cat10pct, embedding) FROM STDIN") as copy:
                for i in tqdm(range(scale)):
                    copy.write_row((i, bool(cat1[i]), bool(cat10[i]), vec_literal(base[i])))

        with Timer() as t_index:
            cur.execute(
                "CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW['m']}, ef_construction = {HNSW['ef_construction']})")

        # 統計を確定させる。これを怠るとプランナが HNSW+後フィルタを選び、
        # 低選択率フィルタで recall が 1 桁 % に崩壊する (実測で確認済み)
        cur.execute("ANALYZE items")

        cur.execute("SELECT pg_total_relation_size('items'), pg_relation_size('items_hnsw')")
        table_bytes, index_bytes = cur.fetchone()

    save_result(f"pgvector-ingest-{scale}", {
        "versions": ver, "hnsw": HNSW,
        "copy_s": t_copy.elapsed, "copy_vec_per_s": scale / t_copy.elapsed,
        "index_build_s": t_index.elapsed,
        "table_bytes": table_bytes, "index_bytes": index_bytes,
    })


SEARCH_SQL = "SELECT id FROM items ORDER BY embedding <=> %s::vector LIMIT %s"


def run_queries(conn, queries, ef, sql=SEARCH_SQL, args_extra=(), iterative=None):
    cur = conn.cursor()
    cur.execute(f"SET hnsw.ef_search = {ef}")
    if iterative:
        cur.execute(f"SET hnsw.iterative_scan = {iterative}")
    else:
        cur.execute("SET hnsw.iterative_scan = off")
    ids = np.empty((len(queries), K), dtype=np.int64)
    lat = []
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        cur.execute(sql, (*args_extra, vec_literal(q), K))
        rows = cur.fetchall()
        lat.append(time.perf_counter() - t0)
        got = [r[0] for r in rows]
        ids[qi] = got + [-1] * (K - len(got))
    return ids, lat


def do_query(scale):
    queries, gt = load_queries(scale), load_gt(scale)
    series = []
    with psycopg.connect(PG_DSN) as conn:
        run_queries(conn, queries[:50], EF_SEARCH[0])  # warmup
        for ef in EF_SEARCH:
            ids, lat = run_queries(conn, queries, ef)
            series.append({"ef_search": ef,
                           "recall": recall_at_k(ids, gt["unfiltered"]),
                           **percentiles(lat),
                           "qps_1thread": len(queries) / sum(lat)})
            print(series[-1])
    save_result(f"pgvector-query-{scale}", {"hnsw": HNSW, "series": series})


def do_concurrent(scale):
    queries, gt = load_queries(scale), load_gt(scale)
    chunks = np.array_split(np.arange(len(queries)), CONCURRENCY)

    def worker(idx):
        with psycopg.connect(PG_DSN) as conn:
            return run_queries(conn, queries[idx], CONC_EF)

    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:  # warmup pools
        list(ex.map(worker, [c[:10] for c in chunks]))

    with Timer() as t, cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        results = list(ex.map(worker, chunks))

    lat = [x for _, ls in results for x in ls]
    ids = np.concatenate([ids for ids, _ in results])
    order = np.concatenate(chunks)
    ids_sorted = np.empty_like(ids); ids_sorted[order] = ids
    save_result(f"pgvector-concurrent-{scale}", {
        "concurrency": CONCURRENCY, "ef_search": CONC_EF,
        "recall": recall_at_k(ids_sorted, gt["unfiltered"]),
        "qps": len(queries) / t.elapsed, **percentiles(lat)})


def do_filtered(scale):
    """iterative_scan off (素の HNSW + post-filter) と relaxed_order の両系列を取る。

    off は候補 ef 件を取ってから WHERE で絞るため、低選択率では recall が崩壊する
    (pgvector 0.8 で iterative_scan が入った理由そのもの)。プランナが seq scan に
    切り替えて exact になるケースもあるため、実行プランも 1 クエリ分記録する。
    """
    queries, gt = load_queries(scale), load_gt(scale)
    out, plans = {}, {}
    with psycopg.connect(PG_DSN) as conn:
        for name in FILTERS:
            sql = f"SELECT id FROM items WHERE {name} ORDER BY embedding <=> %s::vector LIMIT %s"
            for mode in (None, "relaxed_order"):
                key = f"{name}/{mode or 'off'}"
                run_queries(conn, queries[:50], CONC_EF, sql=sql, iterative=mode)
                series = []
                for ef in EF_SEARCH:
                    ids, lat = run_queries(conn, queries, ef, sql=sql, iterative=mode)
                    series.append({"ef_search": ef, "iterative_scan": mode or "off",
                                   "recall": recall_at_k(ids, gt[name]),
                                   **percentiles(lat)})
                    print(key, series[-1])
                    cur = conn.cursor()
                    cur.execute(f"SET hnsw.ef_search = {ef}")
                    cur.execute("EXPLAIN (COSTS OFF) " + sql,
                                ("[" + ",".join(["0"] * DIM) + "]", K))
                    plans[f"{key}/ef{ef}"] = [r[0] for r in cur.fetchall()][:3]
                out[key] = series
    save_result(f"pgvector-filtered-{scale}", {"hnsw": HNSW, "filters": out, "plans": plans})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=int)
    ap.add_argument("phase", choices=["ingest", "query", "concurrent", "filtered"])
    args = ap.parse_args()
    {"ingest": do_ingest, "query": do_query,
     "concurrent": do_concurrent, "filtered": do_filtered}[args.phase](args.scale)


if __name__ == "__main__":
    main()
