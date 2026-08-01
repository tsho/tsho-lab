"""Download DBpedia OpenAI embeddings and materialize base/query/metadata sets.

Usage: python prepare_data.py <scale>

Downloads only the parquet shards needed (each shard ~38.5k rows, 26 total).
Takes the first `scale` rows as the base set and the NEXT N_QUERIES rows as
held-out queries (never inserted). Vectors are L2-normalized so cosine top-k
== inner-product top-k. Category flags derive deterministically from row index.
"""
import argparse
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from common import DIM, FILTERS, N_QUERIES, paths

REPO = "KShivendu/dbpedia-entities-openai-1M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=int)
    args = ap.parse_args()

    p = paths(args.scale)
    p["base"].parent.mkdir(parents=True, exist_ok=True)
    need = args.scale + N_QUERIES

    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, "data", repo_type="dataset")
                   if f.path.endswith(".parquet"))

    vecs = np.empty((need, DIM), dtype=np.float32)
    got = 0
    for path in tqdm(files, desc="shards"):
        local = hf_hub_download(REPO, path, repo_type="dataset")
        t = pq.read_table(local, columns=["openai"])
        arr = np.asarray(t["openai"].combine_chunks().flatten(), dtype=np.float32)
        arr = arr.reshape(-1, DIM)
        take = min(len(arr), need - got)
        vecs[got : got + take] = arr[:take]
        got += take
        if got >= need:
            break
    assert got == need, f"only {got}/{need} rows available"

    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    base, queries = vecs[: args.scale], vecs[args.scale :]
    np.save(p["base"], base)
    np.save(p["queries"], queries)

    ids = np.arange(args.scale, dtype=np.int64)
    cols = {"id": ids}
    for name, mod in FILTERS.items():
        cols[name] = (ids % mod) == 0
    pq.write_table(pa.table(cols), p["meta"])

    print(f"base {base.shape}, queries {queries.shape} -> {p['base'].parent}")


if __name__ == "__main__":
    main()
