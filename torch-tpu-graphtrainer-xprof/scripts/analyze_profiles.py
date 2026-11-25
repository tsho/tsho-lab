#!/usr/bin/env python3
"""Per-step breakdown of the torch.profiler traces captured by run_profile.sh.

Stdlib only. Reads results/raw/prof/<CONFIG>/profiling/traces/iteration_20/
rank{N}_trace.json.gz and writes results/profile_summary.json (aggregated
numbers only: no host names, no paths).

Trace layout (as found, torch_tpu + libtpu on v6e):
  * process "/device:TPU:<n>"  -> TensorCore of the local chip. Threads:
      - "XLA Modules"   one event per device program execution (tt_jit_*)
      - "XLA Ops"       synchronous HLO ops inside a program (args.hlo_category)
      - "Async XLA Ops" async start..done spans (copies, collective-permute,
                        SparseCore offload); these overlap "XLA Ops"
      - "XLA TraceMe"   "barrier-cores" = cross-chip barrier wait
  * process "/device:TPU:<n> SparseCore k" -> SparseCore lanes
  * host process (python): "main/<tid>" thread carries cpu_op events and the
    ProfilerStep#N user_annotation that marks host-side step boundaries; a
    separate PJRT thread carries PJRT_LoadedExecutable_Execute (dispatch).
  * no "C" (counter) events -> no HBM counters in the JSON.

All "busy" numbers are interval unions clipped to the host ProfilerStep#N
window, never sums of overlapping intervals.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # experiment root (scripts/ lives below it)
RAW = HERE / "results" / "raw" / "prof"
OUT = HERE / "results" / "profile_summary.json"
CONFIGS = ["B_perblock", "C_gt_eager", "D_gt_separate", "E_gt_ingraph"]
RANKS = range(4)
TOKENS_PER_STEP_PER_CHIP = 2048  # num_tokens_per_microbatch_per_dp_rank, 1 microbatch
TPS_JSON = HERE / "results" / "summary.json"
TPS_KEYS: dict[str, str] = {}  # config -> key in TPS_JSON (default: same name)

COLLECTIVE_CATS = {
    "all-gather": "all_gather",
    "reduce-scatter": "reduce_scatter",
    "all-reduce-scatter fusion": "reduce_scatter_fused",
    "all-reduce": "all_reduce",
}
GAP_THRESHOLD_US = 100.0  # gaps between programs counted as "host gaps" if >= this


# ---------------------------------------------------------------- intervals
def union(iv):
    iv = sorted((a, b) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def total(u):
    return sum(b - a for a, b in u)


def clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi]


def intersect_len(u1, u2):
    i = j = 0
    s = 0.0
    while i < len(u1) and j < len(u2):
        a = max(u1[i][0], u2[j][0])
        b = min(u1[i][1], u2[j][1])
        if b > a:
            s += b - a
        if u1[i][1] < u2[j][1]:
            i += 1
        else:
            j += 1
    return s


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "median": round(statistics.median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


# ---------------------------------------------------------------- programs
def program_kind(name: str) -> str:
    n = re.sub(r"\(\d+\)$", "", name)
    if n.startswith("tt_jit_distributed."):
        return "collective_program"
    if "fused_adamw" in n or "foreach_addcdiv" in n:
        return "optimizer"
    if n.startswith("tt_jit_export_L204C0_dispatch_fx_node_addcdiv"):
        return "step_graph_with_optimizer"
    if n.startswith("tt_jit_autograd_"):
        return "fwd_bwd_graph"
    if n.startswith("tt_jit_export_L589C0_fx_to_mlir"):
        return "per_block_compiled"
    if any(k in n for k in ("foreach_norm", "foreach_mul_", "linalg_vector_norm",
                            "clamp", "reciprocal", "stack")):
        return "grad_norm_clip"
    if n in ("tt_jit_fill_",):
        return "fill"
    return "other_eager_op"


def short(name: str) -> str:
    n = re.sub(r"\(\d+\)$", "", name)
    return n[:90]


# ---------------------------------------------------------------- per rank
def analyze_rank(path: Path) -> dict:
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    pname = {}
    tname = {}
    for e in ev:
        if e.get("ph") != "M":
            continue
        if e["name"] == "process_name":
            pname[e["pid"]] = e["args"]["name"]
        elif e["name"] == "thread_name":
            tname[(e["pid"], e["tid"])] = e["args"]["name"]
    tc_pids = {p for p, n in pname.items() if re.fullmatch(r"/device:TPU:\d+", n)}
    sc_pids = {p for p, n in pname.items() if re.fullmatch(r"/device:TPU:\d+ SparseCore \d+", n)}
    assert len(tc_pids) == 1, pname
    tc = next(iter(tc_pids))
    host_pids = {p for p, n in pname.items() if isinstance(p, int) and not n.startswith("/device") and n not in ("TPU",)}

    lanes = defaultdict(list)
    n_counter = 0
    steps = []
    for e in ev:
        ph = e.get("ph")
        if ph == "C":
            n_counter += 1
            continue
        if ph != "X":
            continue
        key = (e["pid"], e["tid"])
        t = tname.get(key, "")
        if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep#"):
            steps.append(e)
            continue
        if e["pid"] == tc:
            lanes["tc:" + t].append(e)
        elif e["pid"] in sc_pids:
            lanes["sc:" + t].append(e)
        elif e["pid"] in host_pids:
            if t.startswith("main/"):
                lanes["host_main"].append(e)
            lanes["host_all"].append(e)
    steps.sort(key=lambda e: e["ts"])

    mods = sorted(lanes["tc:XLA Modules"], key=lambda e: e["ts"])
    ops = lanes["tc:XLA Ops"]
    aops = lanes["tc:Async XLA Ops"]
    barrier = [e for e in lanes["tc:XLA TraceMe"] if e["name"] == "barrier-cores"]
    sc_mod = lanes["sc:Sparse Core Modules"]
    host_main = lanes["host_main"]
    dispatch = [e for e in lanes["host_all"] if e["name"] == "PJRT_LoadedExecutable_Execute"]
    sema = [e for e in lanes["host_all"] if e["name"] == "Acquire semaphore"]

    iv = lambda es: [(e["ts"], e["ts"] + e.get("dur", 0.0)) for e in es]  # noqa: E731

    per_step = []
    for s in steps:
        a, b = s["ts"], s["ts"] + s["dur"]
        wall = b - a
        # --- programs
        smods = [m for m in mods if m["ts"] < b and m["ts"] + m["dur"] > a]
        mu = union(clip(iv(smods), a, b))
        busy_mod = total(mu)
        # gaps between consecutive program intervals (and window edges)
        edges = [a] + [x for seg in mu for x in seg] + [b]
        gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges), 2)]
        big_gaps = [g for g in gaps if g >= GAP_THRESHOLD_US]
        # attribute each gap to the program that ends it ("window_end" for the tail)
        gap_iv = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] > edges[i]]
        gap_by_next = defaultdict(float)
        for g0, g1 in gap_iv:
            nxt = next((m for m in smods if m["ts"] >= g1 - 1e-6), None)
            gap_by_next[program_kind(nxt["name"]) if nxt and g1 < b else "window_end"] += g1 - g0
        # --- ops (sync TensorCore lane)
        sops = clip(iv([o for o in ops if o["ts"] < b and o["ts"] + o["dur"] > a]), a, b)
        ou = union(sops)
        busy_ops = total(ou)
        coll = defaultdict(list)
        for o in ops:
            if o["ts"] < b and o["ts"] + o["dur"] > a:
                cat = o.get("args", {}).get("hlo_category")
                if cat in COLLECTIVE_CATS:
                    coll[COLLECTIVE_CATS[cat]].append((o["ts"], o["ts"] + o["dur"]))
        coll_u = {k: total(union(clip(v, a, b))) for k, v in coll.items()}
        coll_all = union(clip([x for v in coll.values() for x in v], a, b))
        # --- async spans (overlap with sync ops is expected)
        async_by = defaultdict(list)
        for o in aops:
            if o["ts"] < b and o["ts"] + o["dur"] > a:
                async_by[o.get("args", {}).get("hlo_category", "?")].append((o["ts"], o["ts"] + o["dur"]))
        cp = union(clip(async_by.get("collective-permute-start", []), a, b))
        # --- barrier waits (inside programs)
        bu = union(clip(iv([x for x in barrier if x["ts"] < b and x["ts"] + x["dur"] > a]), a, b))
        # --- SparseCore
        scu = union(clip(iv([x for x in sc_mod if x["ts"] < b and x["ts"] + x["dur"] > a]), a, b))
        # --- program table
        by_kind = defaultdict(lambda: [0, 0.0])
        by_name = defaultdict(lambda: [0, 0.0])
        for m in smods:
            k = program_kind(m["name"])
            by_kind[k][0] += 1
            by_kind[k][1] += m["dur"]
            by_name[short(m["name"])][0] += 1
            by_name[short(m["name"])][1] += m["dur"]
        # --- host
        hm = [h for h in host_main if a <= h["ts"] < b and h.get("cat") == "cpu_op"]
        host_busy = total(union(clip(iv(hm), a, b)))
        items = union(clip(iv([h for h in hm if h["name"] == "aten::_local_scalar_dense"]), a, b))
        disp = union(clip(iv([h for h in dispatch if a <= h["ts"] < b]), a, b))
        sem = union(clip(iv([h for h in sema if a <= h["ts"] < b]), a, b))
        hm_u = union(clip(iv(hm), a, b))
        gap_u = union(gap_iv)
        gap_host_op = intersect_len(gap_u, hm_u)
        gap_host_item = intersect_len(gap_u, items)
        per_step.append({
            "step": s["name"],
            "wall_ms": wall / 1e3,
            "programs": len(smods),
            "device_busy_programs_ms": busy_mod / 1e3,
            "device_idle_between_programs_ms": (wall - busy_mod) / 1e3,
            "device_idle_pct": 100 * (wall - busy_mod) / wall,
            "gaps_ge_100us": len(big_gaps),
            "gaps_ge_100us_ms": sum(big_gaps) / 1e3,
            "max_gap_ms": max(gaps) / 1e3 if gaps else 0.0,
            "device_busy_ops_ms": busy_ops / 1e3,
            "in_program_no_sync_op_ms": (busy_mod - intersect_len(mu, ou)) / 1e3,
            "barrier_wait_ms": total(bu) / 1e3,
            "collective_sync_ms": total(coll_all) / 1e3,
            "collective_sync_by_type_ms": {k: v / 1e3 for k, v in coll_u.items()},
            "collective_permute_async_span_ms": total(cp) / 1e3,
            "collective_permute_overlap_with_sync_ops_ms": intersect_len(cp, ou) / 1e3,
            "sparsecore_busy_ms": total(scu) / 1e3,
            "program_kinds": {k: {"n": v[0], "ms": v[1] / 1e3} for k, v in by_kind.items()},
            "program_names": {k: {"n": v[0], "ms": v[1] / 1e3} for k, v in by_name.items()},
            "host_main_busy_ms": host_busy / 1e3,
            "host_blocked_in_item_ms": total(items) / 1e3,
            "host_dispatch_execute_ms": total(disp) / 1e3,
            "host_dispatch_semaphore_wait_ms": total(sem) / 1e3,
            "host_dispatch_excl_semaphore_ms": (total(disp) - intersect_len(disp, sem)) / 1e3,
            "idle_with_host_main_in_cpu_op_ms": gap_host_op / 1e3,
            "idle_with_host_main_in_item_ms": gap_host_item / 1e3,
            "idle_with_no_host_main_cpu_op_ms": (total(gap_u) - gap_host_op) / 1e3,
            "idle_by_next_program_kind_ms": {k: v / 1e3 for k, v in gap_by_next.items()},
        })
    return {"steps": per_step, "counter_events": n_counter, "n_profiler_steps": len(steps)}


# ---------------------------------------------------------------- run.log
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGLINE = re.compile(r"root - INFO - .*step:\s*(\d+).*tps:\s*([\d,]+)")


def log_tps(path: Path) -> dict[int, int]:
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        m = LOGLINE.search(ANSI.sub("", raw))
        if m:
            out[int(m.group(1))] = int(m.group(2).replace(",", ""))
    return out


def bench_tps(cfg: str) -> int | None:
    p = TPS_JSON
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(TPS_KEYS.get(cfg, cfg), {}).get("tps_per_chip")


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(HERE))
    except ValueError:
        return p.name


def parse_args() -> None:
    global RAW, OUT, CONFIGS, TOKENS_PER_STEP_PER_CHIP, TPS_JSON, RANKS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW, help="dir holding <CONFIG>/profiling/traces/ (default results/raw/prof)")
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON (default results/profile_summary.json)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS, help="configs to analyse; missing ones are skipped")
    ap.add_argument("--tokens", type=int, default=TOKENS_PER_STEP_PER_CHIP, help="tokens per step per chip (default 2048)")
    ap.add_argument("--tps-json", type=Path, default=TPS_JSON, help="unprofiled timing JSON with tps_per_chip (default results/summary.json)")
    ap.add_argument("--tps-key", action="append", default=[], metavar="CONFIG=KEY", help="map a config to its key in --tps-json")
    ap.add_argument("--ranks", type=int, default=len(RANKS), help="number of ranks (chips) to read (default 4)")
    a = ap.parse_args()
    RANKS = range(a.ranks)
    RAW, OUT, TOKENS_PER_STEP_PER_CHIP, TPS_JSON = a.raw.resolve(), a.out.resolve(), a.tokens, a.tps_json.resolve()
    CONFIGS = [c for c in a.configs if (RAW / c).is_dir()]
    for kv in a.tps_key:
        k, v = kv.split("=", 1)
        TPS_KEYS[k] = v


# ---------------------------------------------------------------- main
SCALARS = [
    "wall_ms", "programs", "device_busy_programs_ms", "device_idle_between_programs_ms",
    "device_idle_pct", "gaps_ge_100us", "gaps_ge_100us_ms", "max_gap_ms",
    "device_busy_ops_ms", "in_program_no_sync_op_ms", "barrier_wait_ms",
    "collective_sync_ms", "collective_permute_async_span_ms",
    "collective_permute_overlap_with_sync_ops_ms", "sparsecore_busy_ms",
    "host_main_busy_ms", "host_blocked_in_item_ms", "host_dispatch_execute_ms",
    "host_dispatch_semaphore_wait_ms", "host_dispatch_excl_semaphore_ms",
    "idle_with_host_main_in_cpu_op_ms", "idle_with_host_main_in_item_ms",
    "idle_with_no_host_main_cpu_op_ms",
]


def main() -> int:
    parse_args()
    result = {
        "notes": {
            "source": f"torch.profiler Chrome-trace JSON, iteration_20, ranks 0-{len(RANKS) - 1}, under {rel(RAW)}",
            "tokens_per_step_per_chip": TOKENS_PER_STEP_PER_CHIP,
            "step_window": "host-side ProfilerStep#N user_annotation; device intervals clipped to it",
            "aggregation": "per-rank median over the profiled steps, then median/min/max across ranks",
            "device_busy_programs": "union of 'XLA Modules' events (one per device program execution)",
            "device_busy_ops": "union of synchronous 'XLA Ops' events (TensorCore)",
            "in_program_no_sync_op": "time inside a program with no synchronous op running (waiting on async copies/collectives/DMA)",
            "barrier_wait": "'barrier-cores' TraceMe on the device; cross-chip barrier, i.e. waiting for other ranks; it lies inside program time",
            "collective_sync": "union of XLA Ops with hlo_category all-gather / reduce-scatter / all-reduce-scatter fusion / all-reduce; these run on the TensorCore lane and do not overlap other sync ops",
            "collective_permute_async": "collective-permute start..done spans on 'Async XLA Ops'; they overlap sync ops (overlap reported separately)",
            "host_blocked_in_item": "aten::_local_scalar_dense on the main thread (.item(), blocking device->host sync)",
            "host_dispatch_semaphore_wait": "'Acquire semaphore' inside PJRT execute: dispatch blocked because the in-flight program queue is full (host ahead of device)",
            "idle_by_next_program_kind": "device idle gap attributed to the kind of program that ends it; 'window_end' = tail of the step window",
            "idle_with_no_host_main_cpu_op": "device idle time during which the host main thread has no recorded cpu_op (Python-level work is not traced: with_stack/python tracing off)",
            "unprofiled_step_ms_from_tps": f"{TOKENS_PER_STEP_PER_CHIP} tokens/step/chip / tps_per_chip from {rel(TPS_JSON)}; results/summary.json step_ms is rounded to 10 ms because time_end_to_end is logged with 0.01 s resolution",
            "gap_threshold_us": GAP_THRESHOLD_US,
        },
        "configs": {},
    }
    for cfg in CONFIGS:
        tdir = RAW / cfg / "profiling" / "traces" / "iteration_20"
        ranks = {}
        for r in RANKS:
            p = tdir / f"rank{r}_trace.json.gz"
            if p.exists():
                ranks[r] = analyze_rank(p)
        per_rank_med = {}
        for r, rr in ranks.items():
            per_rank_med[r] = {k: statistics.median(s[k] for s in rr["steps"]) for k in SCALARS}
        agg = {k: spread([per_rank_med[r][k] for r in per_rank_med]) for k in SCALARS}
        # program tables: rank 0, median over steps
        r0 = ranks[0]["steps"]
        kinds = sorted({k for s in r0 for k in s["program_kinds"]})
        prog_kinds = {k: {
            "n_per_step": statistics.median(s["program_kinds"].get(k, {"n": 0})["n"] for s in r0),
            "ms_per_step": round(statistics.median(s["program_kinds"].get(k, {"ms": 0})["ms"] for s in r0), 3),
        } for k in kinds}
        names = sorted({k for s in r0 for k in s["program_names"]})
        prog_names = {k: {
            "n_per_step": statistics.median(s["program_names"].get(k, {"n": 0})["n"] for s in r0),
            "ms_per_step": round(statistics.median(s["program_names"].get(k, {"ms": 0})["ms"] for s in r0), 3),
        } for k in names}
        prog_names = dict(sorted(prog_names.items(), key=lambda kv: -kv[1]["ms_per_step"]))
        coll_types = sorted({k for s in r0 for k in s["collective_sync_by_type_ms"]})
        coll_by_type = {k: spread([statistics.median(s["collective_sync_by_type_ms"].get(k, 0.0)
                                                     for s in ranks[r]["steps"]) for r in ranks])
                        for k in coll_types}
        idle_kinds = sorted({k for s in r0 for k in s["idle_by_next_program_kind_ms"]})
        idle_by_next = {k: spread([statistics.median(s["idle_by_next_program_kind_ms"].get(k, 0.0)
                                                     for s in ranks[r]["steps"]) for r in ranks])
                        for k in idle_kinds}
        tps = bench_tps(cfg)
        lt = log_tps(RAW / cfg / "run.log")
        result["configs"][cfg] = {
            "ranks_parsed": len(ranks),
            "profiler_steps": [s["step"] for s in r0],
            "counter_events_in_json": sum(rr["counter_events"] for rr in ranks.values()),
            "unprofiled_step_ms_from_tps": round(TOKENS_PER_STEP_PER_CHIP / tps * 1000, 1) if tps else None,
            "profiled_run_log_tps_steps_11_30_median": statistics.median(v for k, v in lt.items() if 11 <= k <= 30) if lt else None,
            "metrics": agg,
            "collective_sync_by_type_ms": coll_by_type,
            "idle_by_next_program_kind_ms": idle_by_next,
            "program_kinds_rank0": prog_kinds,
            "program_names_rank0": prog_names,
            "per_step_rank0": [{k: round(s[k], 3) if isinstance(s[k], float) else s[k] for k in ["step"] + SCALARS} for s in r0],
        }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    hdr = f"{'config':15s} {'wall':>7s} {'unprof':>7s} {'progs':>5s} {'busyP':>7s} {'idle':>6s} {'idle%':>6s} {'opsU':>7s} {'barr':>6s} {'coll':>6s} {'item':>7s}"
    print(hdr)
    for cfg, c in result["configs"].items():
        m = c["metrics"]
        print(f"{cfg:15s} {m['wall_ms']['median']:7.1f} {c['unprofiled_step_ms_from_tps'] or 0:7.1f} "
              f"{m['programs']['median']:5.0f} {m['device_busy_programs_ms']['median']:7.1f} "
              f"{m['device_idle_between_programs_ms']['median']:6.1f} {m['device_idle_pct']['median']:6.1f} "
              f"{m['device_busy_ops_ms']['median']:7.1f} {m['barrier_wait_ms']['median']:6.1f} "
              f"{m['collective_sync_ms']['median']:6.1f} {m['host_blocked_in_item_ms']['median']:7.1f}")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
