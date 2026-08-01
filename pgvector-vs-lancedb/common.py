"""Shared config and helpers for the pgvector vs LanceDB benchmark."""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", BENCH_DIR / "data"))
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", BENCH_DIR / "results"))

DIM = 1536
N_QUERIES = 1000
K = 10
PG_DSN = "host=127.0.0.1 port=15432 user=bench password=bench dbname=bench"

# 選択率つきカテゴリ (決定的に付与)
FILTERS = {"cat1pct": 100, "cat10pct": 10}  # name -> modulo (i % mod == 0)


def paths(scale: int):
    d = DATA_DIR / f"{scale}"
    return {
        "base": d / "base.npy",          # (scale, DIM) float32, L2-normalized
        "meta": d / "meta.parquet",      # id, cat1pct(bool), cat10pct(bool)
        "queries": d / "queries.npy",    # (N_QUERIES, DIM) float32
        "gt": d / "gt.npz",              # unfiltered/cat1pct/cat10pct -> (N_QUERIES, K) int64
    }


def load_base(scale):
    return np.load(paths(scale)["base"], mmap_mode="r")


def load_queries(scale):
    return np.load(paths(scale)["queries"])


def load_gt(scale):
    return dict(np.load(paths(scale)["gt"]))


def recall_at_k(result_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    """result_ids, gt_ids: (n_queries, K) arrays of int ids."""
    hits = 0
    for res, gt in zip(result_ids, gt_ids):
        hits += len(set(res.tolist()) & set(gt.tolist()))
    return hits / (len(gt_ids) * gt_ids.shape[1])


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def percentiles(latencies_s):
    a = np.array(latencies_s) * 1000  # ms
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def save_result(name: str, payload: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "bench": name,
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
        **payload,
    }
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved {out}")
