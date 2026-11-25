#!/usr/bin/env python3
"""Median of steps 11..N for every *.log in a directory, plus device HBM peaks
from the tpu-info samples (*.hbm) taken while each run was going."""
import json, re, statistics, sys
from pathlib import Path
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"step:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.]+)\s+memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+tps:\s*([\d,]+)\s+tflops:\s*([\d.]+)\s+mfu:\s*([\d.]+)%")
HBM = re.compile(r"([\d.]+)\s*GiB\s*/\s*([\d.]+)\s*GiB")
d = Path(sys.argv[1]); out = {}
for p in sorted(d.glob("*.log")):
    by = {}
    for raw in p.read_text(errors="replace").splitlines():
        m = LINE.search(ANSI.sub("", raw))
        if m:
            by[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(6).replace(",", "")), float(m.group(8)))
    st = [v for k, v in sorted(by.items()) if k > 10]
    if not st:
        out[p.stem] = None; continue
    r = {"tps_per_chip": int(statistics.median(v[3] for v in st)),
         "mfu_pct": round(statistics.median(v[4] for v in st), 2),
         "alloc_peak_gib": round(statistics.median(v[2] for v in st), 2),
         "final_loss": by[max(by)][0],
         "grad_norm_all_zero": all(v[1] == 0.0 for v in st)}
    h = d / (p.stem + ".hbm")
    if h.exists():
        vals = [float(m.group(1)) for m in HBM.finditer(h.read_text())]
        tot = [float(m.group(2)) for m in HBM.finditer(h.read_text())]
        if vals:
            r["device_hbm_peak_gib_any_chip"] = max(vals)
            r["device_hbm_total_gib"] = max(tot)
            r["hbm_samples"] = len(vals)
    out[p.stem] = r
for k, r in out.items():
    if not r: print(f"{k:22s} (no steps)"); continue
    print(f"{k:22s} tps={r['tps_per_chip']:6,d} mfu={r['mfu_pct']:5.2f} alloc={r['alloc_peak_gib']:5.2f} "
          f"hbm_peak={r.get('device_hbm_peak_gib_any_chip','-')} gn0={r['grad_norm_all_zero']} loss={r['final_loss']}")
Path(sys.argv[2] if len(sys.argv) > 2 else "day2_summary.json").write_text(json.dumps(out, indent=2))
