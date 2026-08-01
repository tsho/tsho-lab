"""Exact cosine top-K ground truth (unfiltered + per filter), batched NumPy.

Usage: python ground_truth.py <scale>
"""
import argparse

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from common import FILTERS, K, load_base, load_queries, paths


def exact_topk(base, queries, k, mask=None, batch=200):
    ids = np.arange(base.shape[0])
    if mask is not None:
        base = base[mask]
        ids = ids[mask]
    base = np.ascontiguousarray(base)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in tqdm(range(0, len(queries), batch)):
        q = queries[s : s + batch]
        sims = q @ base.T                      # (b, n) — normalized => cosine
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        row = np.arange(len(q))[:, None]
        order = np.argsort(-sims[row, part], axis=1)
        out[s : s + batch] = ids[part[row, order]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=int)
    args = ap.parse_args()

    base = np.asarray(load_base(args.scale))
    queries = load_queries(args.scale)
    meta = pq.read_table(paths(args.scale)["meta"])

    gt = {"unfiltered": exact_topk(base, queries, K)}
    for name in FILTERS:
        mask = meta[name].to_numpy()
        gt[name] = exact_topk(base, queries, K, mask=mask)
        print(f"{name}: candidate pool {mask.sum()}")

    np.savez(paths(args.scale)["gt"], **gt)
    print(f"saved {paths(args.scale)['gt']}")


if __name__ == "__main__":
    main()
