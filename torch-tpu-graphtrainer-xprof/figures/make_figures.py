#!/usr/bin/env python3
"""Render the figures in this directory from the aggregated JSON in ../results.

Requires matplotlib. Reads only ../results/*.json; writes PNGs next to this
script.

    python figures/make_figures.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
DPI = 170

# One restrained palette: categorical slots in fixed order, plus neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
IDLE = "#b9b8b2"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 12,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 12,
    "axes.labelcolor": INK2,
    "axes.edgecolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 11,
})


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def save(fig, name: str) -> None:
    fig.savefig(HERE / name, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- figure 1
def fig_throughput():
    s = load("summary.json")
    rows = [  # id, key, label, group
        ("A", "A_eager", "eager", 0),
        ("B", "B_perblock", "per-block compile", 1),
        ("F", "F_perblock_separate", "per-block + separate_graph", 1),
        ("C", "C_gt_eager", "graph trainer, eager optim", 2),
        ("D", "D_gt_separate", "graph trainer + separate_graph", 2),
        ("E", "E_gt_ingraph", "graph trainer + in_graph", 2),
        ("C2", "C2_gt_eager_cgn", "C + compile_grad_norm", 2),
        ("D2", "D2_gt_separate_cgn", "D + compile_grad_norm", 2),
    ]
    colors = [IDLE, S[0], S[1]]
    base = s["A_eager"]["tps_per_chip"]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for i, (rid, key, label, g) in enumerate(rows):
        v = s[key]["tps_per_chip"]
        ax.barh(i, v, height=0.66, color=colors[g], edgecolor=SURFACE, linewidth=2)
        ax.text(v + 120, i, f"{v:,}  ({v / base:.2f}x)", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(rows)), [f"{r[0]}  {r[2]}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 14500)
    ax.set_xlabel("tokens / s / chip (median of steps 11-30)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    xgrid(ax)
    ax.set_title("Throughput by trainer and optimizer_compile mode\n"
                 "Llama 3 1B, v6e-4, 2048 tokens/device; (x) = relative to A", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, ["TPUTrainer eager", "TPUTrainer per-block compile", "TpuGraphTrainer"],
              loc="lower center", bbox_to_anchor=(0.4, -0.3), ncol=3, fontsize=10.5,
              handlelength=1.0, columnspacing=1.2)
    save(fig, "throughput_by_optimizer_mode.png")


# --------------------------------------------------------------- figure 2
def fig_programs_idle():
    small = load("profile_summary.json")["configs"]
    big = load("profile_summary_big.json")["configs"]
    short = {"B_perblock": "B", "C_gt_eager": "C", "D_gt_separate": "D", "E_gt_ingraph": "E"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    offsets = {  # label placement (points) to avoid collisions
        ("2048", "C"): (8, -14), ("2048", "D"): (8, 4), ("2048", "E"): (8, 4), ("2048", "B"): (-14, 6),
        ("8192", "B"): (-14, 6), ("8192", "C"): (8, -14), ("8192", "E"): (8, 4),
    }
    for (label, cfgs, color, mk) in (("2048", small, S[0], "o"), ("8192", big, S[1], "s")):
        xs, ys, names = [], [], []
        for k, n in short.items():
            if k not in cfgs:
                continue
            m = cfgs[k]["metrics"]
            xs.append(m["programs"]["median"])
            ys.append(m["device_idle_between_programs_ms"]["median"])
            names.append(n)
        ax.plot(xs, ys, color=color, linewidth=2, marker=mk, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{label} tokens/device")
        for x, y, n in zip(xs, ys, names):
            dx, dy = offsets.get((label, n), (8, 4))
            ax.annotate(f"{n}  {y:.1f} ms", (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=10.5, color=INK, ha="right" if dx < 0 else "left")
    ax.set_xlim(-5, 125)
    ax.set_ylim(0, 20)
    ax.set_xlabel("device programs per step")
    ax.set_ylabel("device idle between programs (ms/step)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_title("Device programs per step vs device idle time\n"
                 "Llama 3 1B, v6e-4; B per-block, C/D/E graph trainer; median over ranks",
                 loc="left")
    save(fig, "programs_vs_device_idle.png")


# --------------------------------------------------------------- figure 3
KIND_ORDER = [  # (kind(s) in profile_summary.json, legend label, color)
    (("per_block_compiled",), "per-block compiled programs", S[0]),
    (("fwd_bwd_graph",), "forward/backward graph", S[6]),
    (("step_graph_with_optimizer",), "step graph incl. optimizer", S[2]),
    (("collective_program",), "collective programs", S[1]),
    (("optimizer",), "optimizer program", S[3]),
    (("grad_norm_clip",), "grad-norm / clip programs", S[4]),
    (("other_eager_op", "fill"), "other eager ops (incl. fill)", S[7]),
]


def fig_breakdown():
    cfgs = load("profile_summary.json")["configs"]
    order = [("B_perblock", "B per-block"), ("C_gt_eager", "C graph"),
             ("D_gt_separate", "D graph + separate"), ("E_gt_ingraph", "E graph + in_graph")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    used = set()
    for i, (key, label) in enumerate(order):
        c = cfgs[key]
        pk = c["program_kinds_rank0"]
        left = 0.0
        for kinds, lname, color in KIND_ORDER:
            ms = sum(pk[k]["ms_per_step"] for k in kinds if k in pk)
            if ms <= 0:
                continue
            ax.barh(i, ms, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5)
            if ms >= 20:
                ax.text(left + ms / 2, i, f"{ms:.0f}", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
            used.add(lname)
            left += ms
        idle = statistics.median(s["device_idle_between_programs_ms"] for s in c["per_step_rank0"])
        ax.barh(i, idle, left=left, height=0.62, color=IDLE, edgecolor=SURFACE, linewidth=1.5)
        ax.text(left + idle + 2, i, f"{left + idle:.0f} ms", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 235)
    ax.set_xlabel("device time per step, rank 0 (ms)")
    xgrid(ax)
    handles, labels = [], []
    for _, lname, color in KIND_ORDER:
        if lname in used:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(lname)
    handles.append(plt.Rectangle((0, 0), 1, 1, color=IDLE))
    labels.append("device idle between programs")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=2,
              fontsize=10, handlelength=1.0)
    ax.set_title("Per-step device time by program kind\n"
                 "Llama 3 1B, v6e-4, 2048 tokens/device; rank 0, median of profiled steps",
                 loc="left")
    save(fig, "step_time_breakdown.png")


# --------------------------------------------------------------- figure 4
def fig_memory():
    d = load("day2_summary.json")
    pts = [("mp_default", "default"), ("mp_full", "full"),
           ("mp_tpu_selective", "tpu_selective (w2)"),
           ("mp_full_and_offload", "full_and_offload")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # the first three share an x value; their labels are stacked to the left
    place = {"default": (-12, 16), "tpu_selective (w2)": (-12, 0), "full": (-12, -16),
             "full_and_offload": (12, 0)}
    for i, (k, lab) in enumerate(pts):
        r = d[k]
        x, y = r["device_hbm_peak_gib_any_chip"], r["tps_per_chip"]
        ax.plot(x, y, marker="o", markersize=9, color=S[0] if i < 3 else S[1],
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
        dx, dy = place[lab]
        ax.annotate(f"{lab}: {y:,} tok/s, {x:.2f} GiB", (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.5, color=INK, va="center",
                    ha="left" if dx > 0 else "right")
    ax.set_xlim(6, 33)
    ax.set_ylim(10800, 12100)
    ax.axvline(31.24, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(31.0, 10850, "HBM capacity\n31.24 GiB", ha="right", va="bottom", fontsize=10, color=INK2)
    ax.set_xlabel("device HBM peak, tpu-info, any chip (GiB)")
    ax.set_ylabel("tokens / s / chip")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Throughput vs device HBM peak by memory_policy\n"
                 "graph trainer + in_graph, Llama 3 1B, v6e-4, 2048 tokens/device "
                 "(y axis truncated)\nw2 = save_regions=feed_forward.w2", loc="left", fontsize=12.5)
    save(fig, "memory_policy_tradeoff.png")


# --------------------------------------------------------------- figure 5
def fig_batch():
    s = load("summary.json")
    d = load("day2_summary.json")
    p_small = load("profile_summary.json")["configs"]
    p_big = load("profile_summary_big.json")["configs"]
    ids = [("A", "A_eager", "big_A_eager"), ("B", "B_perblock", "big_B_perblock"),
           ("C", "C_gt_eager", "big_C_gt_eager"), ("E", "E_gt_ingraph", "big_E_ingraph")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    w = 0.38
    b_small = s["A_eager"]["tps_per_chip"]
    b_big = d["big_A_eager"]["tps_per_chip"]
    for i, (n, ks, kb) in enumerate(ids):
        vs = s[ks]["tps_per_chip"] / b_small
        vb = d[kb]["tps_per_chip"] / b_big
        ax1.bar(i - w / 2, vs, w, color=S[0], edgecolor=SURFACE, linewidth=1.5,
                label="2048" if i == 0 else None)
        ax1.bar(i + w / 2, vb, w, color=S[1], edgecolor=SURFACE, linewidth=1.5,
                label="8192" if i == 0 else None)
        if n == "A":  # both are 1.00 by definition; one label
            ax1.text(i, 1.03, "1.00", ha="center", fontsize=9.5, color=INK)
            continue
        ax1.text(i - w / 2, vs + 0.03, f"{vs:.2f}", ha="center", fontsize=9.5, color=INK)
        ax1.text(i + w / 2, vb + 0.03, f"{vb:.2f}", ha="center", fontsize=9.5, color=INK)
    ax1.set_xticks(range(len(ids)), [i[0] for i in ids])
    ax1.set_ylim(0, 2.45)
    ax1.set_ylabel("throughput relative to A (same batch)")
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(title="tokens/device", loc="upper left", fontsize=10, title_fontsize=10, ncol=2,
               handlelength=1.0)
    ax1.set_title("Throughput vs eager", fontsize=12)

    # share of the profiled step: sync collectives and device idle
    cfgs = [("B", "B_perblock"), ("C", "C_gt_eager"), ("E", "E_gt_ingraph")]
    y = 0
    yt, yl = [], []
    for n, k in cfgs:
        for tok, p, color in (("2048", p_small, S[0]), ("8192", p_big, S[1])):
            m = p[k]["metrics"]
            wall = m["wall_ms"]["median"]
            coll = 100 * m["collective_sync_ms"]["median"] / wall
            idle = m["device_idle_pct"]["median"]
            ax2.barh(y, coll, height=0.7, color=color, edgecolor=SURFACE, linewidth=1.5)
            ax2.barh(y, idle, left=coll, height=0.7, color=IDLE, edgecolor=SURFACE, linewidth=1.5)
            ax2.text(coll + idle + 1, y, f"{coll:.0f}% + {idle:.1f}%", va="center", fontsize=9.5, color=INK)
            yt.append(y)
            yl.append(f"{n} {tok}")
            y += 1
        y += 0.4
    ax2.set_yticks(yt, yl, fontsize=10.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 62)
    ax2.set_xlabel("% of profiled step")
    xgrid(ax2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S[0]), plt.Rectangle((0, 0), 1, 1, color=S[1]),
               plt.Rectangle((0, 0), 1, 1, color=IDLE)]
    ax2.legend(handles, ["sync collectives, 2048", "sync collectives, 8192", "device idle"],
               loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=9.5,
               handlelength=1.0)
    ax2.set_title("Step share: collectives + idle", fontsize=12)
    fig.suptitle("2048 vs 8192 tokens per device, Llama 3 1B, v6e-4", x=0.02, ha="left",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "batch_size_comparison.png")


if __name__ == "__main__":
    fig_throughput()
    fig_programs_idle()
    fig_breakdown()
    fig_memory()
    fig_batch()
