"""LanceDB benchmark: ingest + index build, recall-latency sweep, concurrency, filters.

Usage:
  python bench_lancedb.py <scale> ingest
  python bench_lancedb.py <scale> query
  python bench_lancedb.py <scale> concurrent
  python bench_lancedb.py <scale> filtered
"""
import argparse
import concurrent.futures as cf
import time

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lancedb.index import IvfHnswSq

from common import (DATA_DIR, DIM, FILTERS, K, Timer, load_base, load_gt,
                    load_queries, paths, percentiles, recall_at_k, save_result)

INDEX = {"index_type": "IVF_HNSW_SQ", "m": 16, "ef_construction": 64}
# (nprobes, ef, refine_factor)。SQ 量子化の近似誤差で recall が頭打ちになるため、
# 高 recall 帯は refine_factor (全精度ベクトルでの再ランク) で到達させる。
SWEEP = [
    {"nprobes": 1, "ef": 10, "refine_factor": None},
    {"nprobes": 2, "ef": 20, "refine_factor": None},
    {"nprobes": 4, "ef": 40, "refine_factor": None},
    {"nprobes": 8, "ef": 80, "refine_factor": None},
    {"nprobes": 8, "ef": 80, "refine_factor": 2},
    {"nprobes": 16, "ef": 160, "refine_factor": 4},
    {"nprobes": 32, "ef": 320, "refine_factor": 8},
]
CONCURRENCY = 8
CONC_PARAM = {"nprobes": 8, "ef": 80, "refine_factor": 2}


def db_uri(scale):
    return str(DATA_DIR / f"{scale}" / "lancedb")


def do_ingest(scale):
    base = np.asarray(load_base(scale))
    meta = pq.read_table(paths(scale)["meta"])

    db = lancedb.connect(db_uri(scale))
    try:
        db.drop_table("items")
    except Exception:
        pass

    tbl_arrow = pa.table({
        "id": meta["id"],
        "cat1pct": meta["cat1pct"],
        "cat10pct": meta["cat10pct"],
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(base.ravel(), type=pa.float32()), DIM),
    })

    with Timer() as t_ingest:
        tbl = db.create_table("items", tbl_arrow)

    with Timer() as t_index:
        tbl.create_index("vector", config=IvfHnswSq(
            distance_type="cosine", m=INDEX["m"],
            ef_construction=INDEX["ef_construction"]))
        tbl.wait_for_index(["vector_idx"])

    size_bytes = sum(f.stat().st_size for f in
                     (DATA_DIR / f"{scale}" / "lancedb").rglob("*") if f.is_file())
    save_result(f"lancedb-ingest-{scale}", {
        "versions": {"lancedb": lancedb.__version__},
        "index": INDEX,
        "ingest_s": t_ingest.elapsed, "ingest_vec_per_s": scale / t_ingest.elapsed,
        "index_build_s": t_index.elapsed, "dir_bytes": size_bytes,
    })


def open_table(scale):
    return lancedb.connect(db_uri(scale)).open_table("items")


def run_queries(tbl, queries, param, where=None):
    ids = np.empty((len(queries), K), dtype=np.int64)
    lat = []
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        s = (tbl.search(q).metric("cosine")
             .nprobes(param["nprobes"]).ef(param["ef"]).limit(K)
             .select(["id"]))
        if param.get("refine_factor"):
            s = s.refine_factor(param["refine_factor"])
        if where:
            s = s.where(where, prefilter=True)
        rows = s.to_list()
        lat.append(time.perf_counter() - t0)
        got = [r["id"] for r in rows]
        ids[qi] = got + [-1] * (K - len(got))
    return ids, lat


def do_query(scale):
    queries, gt = load_queries(scale), load_gt(scale)
    tbl = open_table(scale)
    run_queries(tbl, queries[:50], SWEEP[0])  # warmup
    series = []
    for param in SWEEP:
        ids, lat = run_queries(tbl, queries, param)
        series.append({**param,
                       "recall": recall_at_k(ids, gt["unfiltered"]),
                       **percentiles(lat),
                       "qps_1thread": len(queries) / sum(lat)})
        print(series[-1])
    save_result(f"lancedb-query-{scale}", {"index": INDEX, "series": series})


def do_concurrent(scale):
    queries, gt = load_queries(scale), load_gt(scale)
    tbl = open_table(scale)
    chunks = np.array_split(np.arange(len(queries)), CONCURRENCY)

    def worker(idx):
        return run_queries(tbl, queries[idx], CONC_PARAM)

    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        list(ex.map(worker, [c[:10] for c in chunks]))  # warmup

    with Timer() as t, cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        results = list(ex.map(worker, chunks))

    lat = [x for _, ls in results for x in ls]
    ids = np.concatenate([ids for ids, _ in results])
    order = np.concatenate(chunks)
    ids_sorted = np.empty_like(ids); ids_sorted[order] = ids
    save_result(f"lancedb-concurrent-{scale}", {
        "concurrency": CONCURRENCY, **CONC_PARAM,
        "recall": recall_at_k(ids_sorted, gt["unfiltered"]),
        "qps": len(queries) / t.elapsed, **percentiles(lat)})


def do_filtered(scale):
    queries, gt = load_queries(scale), load_gt(scale)
    tbl = open_table(scale)
    out = {}
    for name in FILTERS:
        where = f"{name} = true"
        run_queries(tbl, queries[:50], CONC_PARAM, where)
        series = []
        for param in SWEEP:
            ids, lat = run_queries(tbl, queries, param, where)
            series.append({**param,
                           "recall": recall_at_k(ids, gt[name]), **percentiles(lat)})
            print(name, series[-1])
        out[name] = series
    save_result(f"lancedb-filtered-{scale}", {"index": INDEX, "filters": out})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=int)
    ap.add_argument("phase", choices=["ingest", "query", "concurrent", "filtered"])
    args = ap.parse_args()
    {"ingest": do_ingest, "query": do_query,
     "concurrent": do_concurrent, "filtered": do_filtered}[args.phase](args.scale)


if __name__ == "__main__":
    main()
