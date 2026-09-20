"""Turn results/*.jsonl into the table and figure the blog post needs.

    python3 summarize.py                    # all results
    python3 summarize.py --markdown         # markdown table for the post
    python3 summarize.py --plot             # figure comparing hardware

The figure is the point of the whole exercise: the same ZeRO stages on NVLink
hardware and on PCIe-only hardware, so the communication cost is visible rather
than asserted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# dataviz: categorical slots in fixed order, never cycled.
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"


def load() -> list[dict]:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"no results yet: {RESULTS_DIR}")
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("results files are empty")
    return rows


def label(r: dict) -> str:
    # FSDP records carry `strategy`; ZeRO records carry `zero_stage`.
    if r.get("strategy"):
        return r["strategy"]
    s = f"ZeRO-{r['zero_stage']}" if r.get("zero_stage") else "DDP (stage 0)"
    return s + (" +offload" if r.get("offload") else "")


def _sort_key(r: dict) -> tuple:
    # DeepSpeed rows first (by stage), then FSDP rows (by strategy name), so the
    # comparison reads engine-by-engine.
    if r.get("strategy"):
        return (1, 99, r["strategy"])
    return (0, r.get("zero_stage") or 0, "")


def hw(r: dict) -> str:
    e = r["env"]
    nv = e.get("nvlink_active_links")
    link = "NVLink" if nv else "PCIe"
    return f"{e['gpu_name']} x{e['gpu_count']} ({link})"


def table(rows: list[dict], markdown: bool) -> None:
    by_hw: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hw[hw(r)].append(r)

    for hardware, rs in by_hw.items():
        rs.sort(key=_sort_key)
        print()
        print(f"### {hardware}")
        model = rs[0]["model"]
        print(f"model {model} · seq {rs[0]['seq_len']} · micro-batch {rs[0]['micro_batch']}")
        print()
        hdr = ["config", "tok/s", "step ms", "peak alloc GB", "NCCL share", "status"]
        if markdown:
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join(["---"] * len(hdr)) + "|")
        else:
            print(f"{hdr[0]:<20}{hdr[1]:>10}{hdr[2]:>10}{hdr[3]:>16}{hdr[4]:>12}  {hdr[5]}")
        for r in rs:
            comm = r.get("profile") or {}
            frac = comm.get("comm_fraction")
            cells = [
                label(r),
                f"{r['tokens_per_sec_global']:,}" if r["tokens_per_sec_global"] else "—",
                f"{r['step_ms_median']}" if r["step_ms_median"] else "—",
                f"{r['peak_alloc_gb']}",
                f"{frac:.1%}" if frac is not None else "—",
                "**OOM**" if r["oom"] else "ok",
            ]
            if markdown:
                print("| " + " | ".join(cells) + " |")
            else:
                print(f"{cells[0]:<20}{cells[1]:>10}{cells[2]:>10}{cells[3]:>16}{cells[4]:>12}  {cells[5]}")


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if not r["oom"] and r["tokens_per_sec_global"]]
    if not ok:
        raise SystemExit("nothing to plot (all runs OOMed?)")

    by_hw: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_hw[hw(r)].append(r)

    stages = sorted({label(r) for r in ok},
                    key=lambda s: (("offload" in s), s))

    fig, (ax_t, ax_c) = plt.subplots(
        2, 1, figsize=(8.0, 6.6), sharex=True, gridspec_kw={"hspace": 0.16}
    )
    fig.patch.set_facecolor(SURFACE)

    width = 0.8 / max(len(by_hw), 1)
    for i, (hardware, rs) in enumerate(sorted(by_hw.items())):
        color = SERIES[i % len(SERIES)]
        lookup = {label(r): r for r in rs}
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(stages))]
        tput = [lookup[s]["tokens_per_sec_global"] if s in lookup else 0 for s in stages]
        share = [
            (lookup[s].get("profile") or {}).get("comm_fraction", 0) * 100
            if s in lookup else 0
            for s in stages
        ]
        # 4px-equivalent rounded ends are an HTML affordance; in matplotlib we
        # keep bars thin and separated by a surface-coloured gap instead.
        ax_t.bar(xs, tput, width=width * 0.88, color=color, label=hardware, zorder=3)
        ax_c.bar(xs, share, width=width * 0.88, color=color, label=hardware, zorder=3)

    for ax, ylabel in ((ax_t, "throughput (tokens/sec)"), (ax_c, "NCCL share of CUDA time (%)")):
        ax.set_facecolor(SURFACE)
        ax.set_ylabel(ylabel, color=INK2, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d5d4cd")
        ax.tick_params(colors=MUTED, labelsize=10)

    ax_c.set_xticks(range(len(stages)))
    ax_c.set_xticklabels(stages, fontsize=10, color=INK2)
    ax_t.set_title(
        "Sharding buys memory. The interconnect decides what it costs.",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    leg = ax_t.legend(frameon=False, fontsize=10, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout()
    out = RESULTS_DIR / "zero_stages_comparison.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    rows = load()
    table(rows, args.markdown)
    if args.plot:
        plot(rows)


if __name__ == "__main__":
    main()
