#!/usr/bin/env python3
"""Window-level numbers from the XProf .xplane.pb files (needs `pip install xprof`).

xprof writes cache files (*.hlo_proto.pb, op_stats) next to the xplane it
reads, so each file is first copied to a temporary directory: results/raw/ is
never written to. Adds an "xplane" section to results/profile_summary.json
(run analyze_profiles.py first).

Extracted per config/rank:
  * overview_page: device duty cycle / idle % over the whole profiled window
    (not per step: the tool reports "No step time measured" for this trace)
  * memory_profile: HBM capacity, peak bytes in use, heap vs stack-reserved
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

from xprof.convert import raw_to_tool_data as r2t

HERE = Path(__file__).resolve().parent.parent  # experiment root (scripts/ lives below it)
RAW = HERE / "results" / "raw" / "prof"
OUT = HERE / "results" / "profile_summary.json"
CONFIGS = ["B_perblock", "C_gt_eager", "D_gt_separate", "E_gt_ingraph"]
GIB = 1024 ** 3


def pct(s: str) -> float | None:
    try:
        return float(s.rstrip("%"))
    except (TypeError, ValueError):
        return None


def one(xplane: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.xplane.pb"
        shutil.copyfile(xplane, p)
        ov, _ = r2t.xspace_to_tool_data([str(p)], "overview_page", {"use_saved_result": False})
        mp, _ = r2t.xspace_to_tool_data([str(p)], "memory_profile", {"use_saved_result": False})
    ov = json.loads(ov)
    props = ov[0]["p"]
    mp = json.loads(mp)
    alloc = mp["memoryProfilePerAllocator"]["0"]
    summ = alloc["profileSummary"]
    peak = summ["peakStats"]
    return {
        "device_duty_cycle_pct": pct(props.get("device_duty_cycle_percent")),
        "device_idle_time_pct": pct(props.get("device_idle_time_percent")),
        "mxu_utilization_pct_reported": pct(props.get("mxu_utilization_percent")),
        "hbm_capacity_gib": int(summ["memoryCapacity"]) / GIB,
        "hbm_peak_in_use_gib": int(peak["peakBytesInUse"]) / GIB,
        "hbm_peak_heap_allocated_gib": int(peak["heapAllocatedBytes"]) / GIB,
        "hbm_stack_reserved_gib": int(peak["stackReservedBytes"]) / GIB,
        "hbm_memory_snapshots": len(alloc.get("memoryProfileSnapshots", [])),
    }


def main() -> int:
    global RAW, OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW, help="dir holding <CONFIG>/profiling/traces/ (default results/raw/prof)")
    ap.add_argument("--out", type=Path, default=OUT, help="summary JSON to extend (default results/profile_summary.json)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--ranks", type=int, default=4, help="number of ranks (chips) to read (default 4)")
    a = ap.parse_args()
    RAW, OUT = a.raw.resolve(), a.out.resolve()
    configs = [c for c in a.configs if (RAW / c).is_dir()]
    summary = json.loads(OUT.read_text())
    xp = {"notes": {
        "scope": "whole profiled window (all ~5 steps plus window edges), not per step",
        "mxu_utilization_pct_reported": "reported as 0.0 for every file; hardware perf counters were evidently not captured, so treat as not obtainable",
        "hbm_peak_in_use": "heap allocated + stack reserved (program scratch) at the peak, per XProf memory_profile",
    }, "configs": {}}
    for cfg in configs:
        per = []
        for r in range(a.ranks):
            files = glob.glob(str(RAW / cfg / "profiling" / "traces" / f"rank_{r}" / "plugins" / "profile" / "*" / "*.xplane.pb"))
            if files:
                per.append(one(Path(files[0])))
        agg = {}
        for k in per[0]:
            vals = [p[k] for p in per if p[k] is not None]
            agg[k] = {"median": round(statistics.median(vals), 3), "min": round(min(vals), 3),
                      "max": round(max(vals), 3)} if vals else None
        xp["configs"][cfg] = {"ranks": len(per), **agg}
        print(cfg, {k: (v["median"] if v else None) for k, v in agg.items()})
    summary["xplane"] = xp
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
