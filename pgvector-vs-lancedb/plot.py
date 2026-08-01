#!/usr/bin/env python3
"""Generate the benchmark charts into ./charts from ./results.

Standalone version of the chart generator used for the article at
https://implicit-none.com/en/pgvector-vs-lancedb-benchmark/
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "charts"
SCALE = 100_000

C_PG, C_LDB, C_PG2 = "#336791", "#e8582b", "#7fa8c9"


def load(name):
    return json.loads((BASE / f"{name}-{SCALE}.json").read_text())


def save(fig, fname):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=130)
    print(f"wrote {OUT / fname}")
    plt.close(fig)


plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# recall vs latency
pg, ldb = load("pgvector-query")["series"], load("lancedb-query")["series"]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot([s["recall"] for s in pg], [s["p50_ms"] for s in pg], "o-",
        color=C_PG, linewidth=2, label="pgvector 0.8.6 (HNSW)")
ax.plot([s["recall"] for s in ldb], [s["p50_ms"] for s in ldb], "s--",
        color=C_LDB, linewidth=2, label="LanceDB 0.36 (IVF_HNSW_SQ)")
ax.set_xlabel("recall@10"); ax.set_ylabel("p50 latency (ms, single thread)")
ax.set_title(f"Recall vs latency — {SCALE:,} × 1536-dim, k=10")
ax.legend(loc="upper left")
save(fig, "recall-latency.png")

# filtered (scatter — pgvector plan flips make recall non-monotonic in ef)
pgf, lf = load("pgvector-filtered")["filters"], load("lancedb-filtered")["filters"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, sel in zip(axes, ["cat1pct", "cat10pct"]):
    for data, label, color, m in [
        (pgf[f"{sel}/off"], "pgvector (iterative off)", C_PG2, "o"),
        (pgf[f"{sel}/relaxed_order"], "pgvector (relaxed_order)", C_PG, "o"),
        (lf[sel], "LanceDB (prefilter)", C_LDB, "s"),
    ]:
        ax.scatter([s["recall"] for s in data], [s["p50_ms"] for s in data],
                   color=color, marker=m, s=45, label=label, zorder=3)
    ax.set_xlabel("recall@10")
    ax.set_title(f"selectivity {'1%' if sel == 'cat1pct' else '10%'}")
axes[0].set_ylabel("p50 latency (ms)")
axes[0].legend(loc="upper left")
save(fig, "filtered-latency.png")
