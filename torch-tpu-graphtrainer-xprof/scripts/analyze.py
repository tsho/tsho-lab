#!/usr/bin/env python3
"""Parse TorchTitan metrics lines into a per-configuration summary.

Takes the median over steps 11..N, discarding the first 10 as warmup, which is
what the previous experiment did. Each step is logged once per rank, so the
duplicates are collapsed by step number before taking the median.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

# Strip ANSI colour, then pull the fields.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[\d.]+)\s+"
    r"grad_norm:\s*(?P<gn>[\d.naN]+)\s+"
    r"memory:\s*(?P<mem>[\d.]+)GiB\((?P<mempct>[\d.]+)%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+"
    r"tflops:\s*(?P<tflops>[\d.]+)\s+"
    r"mfu:\s*(?P<mfu>[\d.]+)%\s+"
    r"time_end_to_end:\s*(?P<t>[\d.]+)s"
)

WARMUP = 10


def parse(path: Path) -> dict | None:
    by_step: dict[int, dict] = {}
    for raw in path.read_text(errors="replace").splitlines():
        m = LINE.search(ANSI.sub("", raw))
        if not m:
            continue
        s = int(m.group("step"))
        by_step[s] = {
            "loss": float(m.group("loss")),
            "grad_norm": float(m.group("gn")) if m.group("gn").replace(".", "").isdigit() else None,
            "mem_gib": float(m.group("mem")),
            "mem_pct": float(m.group("mempct")),
            "tps": int(m.group("tps").replace(",", "")),
            "tflops": float(m.group("tflops")),
            "mfu": float(m.group("mfu")),
            "step_s": float(m.group("t")),
        }
    if not by_step:
        return None
    steady = [v for k, v in sorted(by_step.items()) if k > WARMUP]
    if not steady:
        return None
    med = lambda key: statistics.median(v[key] for v in steady)  # noqa: E731
    last = by_step[max(by_step)]
    gns = [v["grad_norm"] for v in steady if v["grad_norm"] is not None]
    return {
        "steps_logged": len(by_step),
        "steady_steps": len(steady),
        "step_ms": round(med("step_s") * 1000, 1),
        "tps_per_chip": int(med("tps")),
        "mfu_pct": round(med("mfu"), 2),
        "tflops": round(med("tflops"), 2),
        "alloc_peak_gib": round(med("mem_gib"), 2),
        "alloc_peak_pct": round(med("mem_pct"), 2),
        "final_loss": last["loss"],
        "grad_norm_median": round(statistics.median(gns), 4) if gns else None,
        "grad_norm_all_zero": bool(gns) and all(g == 0.0 for g in gns),
    }


ORDER = ["A_eager", "B_perblock", "C_gt_eager", "D_gt_separate", "E_gt_ingraph",
         "F_perblock_separate", "C2_gt_eager_cgn", "D2_gt_separate_cgn"]

out = {}
d = Path(sys.argv[1] if len(sys.argv) > 1 else "bench")
for name in ORDER:
    p = d / f"{name}.log"
    out[name] = parse(p) if p.exists() else None

base = out.get("A_eager", {}) or {}
basetps = base.get("tps_per_chip")

hdr = f"{'config':24s} {'step ms':>8s} {'tok/s/chip':>11s} {'MFU%':>6s} {'alloc GiB':>10s} {'vs eager':>9s} {'loss':>9s} {'grad_norm':>10s}"
print(hdr)
print("-" * len(hdr))
for name in ORDER:
    r = out[name]
    if not r:
        print(f"{name:24s}  (no data)")
        continue
    sp = f"{r['tps_per_chip']/basetps:.2f}x" if basetps else "-"
    gn = "ALL ZERO" if r["grad_norm_all_zero"] else (f"{r['grad_norm_median']}" if r["grad_norm_median"] is not None else "-")
    print(f"{name:24s} {r['step_ms']:8.1f} {r['tps_per_chip']:11,d} {r['mfu_pct']:6.2f} "
          f"{r['alloc_peak_gib']:10.2f} {sp:>9s} {r['final_loss']:9.5f} {gn:>10s}")

Path("summary.json").write_text(json.dumps(out, indent=2))
print("\nwrote summary.json")
