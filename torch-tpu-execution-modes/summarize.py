#!/usr/bin/env python3
"""Summarize TorchTPU 0.2.0 execution-mode JSON results into Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = [
    ("A", "results_eager.json", "Strict Eager (`DEFER_NEVER`)"),
    ("B (env)", "results_defer_fuse.json", "`TORCH_TPU_DEFER_AND_FUSE=1`"),
    ("B (API)", "results_defer_fuse_ctx.json", "`set_eager_mode(DEFER_AND_FUSE)`"),
    ("C", "results_compile.json", "`torch.compile(model)` + eager SGD"),
    ("C (step)", "results_compile_step.json", "`@torch.compile` full step (in-place SGD)"),
    ("D (public)", "results_single_trace.json", "`SingleTraceTrainer` + `Sgd(fused=False)` + `torch.compile`"),
    ("D (static)", "results_single_trace_static.json", "`SingleTraceTrainer` + `Sgd(fused=True)` + `StaticCompiler`"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results"
        if (Path(__file__).parent / "results").is_dir()
        else Path(__file__).parent,
        help="Directory containing results_*.json files",
    )
    args = parser.parse_args()
    rdir: Path = args.results_dir

    eager_data = json.loads((rdir / "results_eager.json").read_text())
    base_tok_s = eager_data["tokens_per_sec"]

    print("| Group | Mode | Configuration | Setup + 1st Step (s) | Steady Median (ms) | p10 – p90 (ms) | tok/s | vs Strict Eager |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for group, fname, label in MODES:
        d = json.loads((rdir / fname).read_text())
        ratio = d["tokens_per_sec"] / base_tok_s
        print(
            f"| **{group}** | `{d['mode']}` | {label} | "
            f"{d['first_step_plus_setup_sec']:.2f} s | "
            f"**{d['step_ms_median']:.2f} ms** | "
            f"{d['step_ms_p10']:.2f} – {d['step_ms_p90']:.2f} | "
            f"**{d['tokens_per_sec']:,.1f}** | "
            f"{ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
