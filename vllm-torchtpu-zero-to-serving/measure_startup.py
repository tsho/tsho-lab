#!/usr/bin/env python3
"""Measure vllm-torchtpu server startup under three cache conditions.

The point of the three conditions is to separate effects that are easy to
conflate:

  A  fresh          no model cache, no compilation cache (includes download)
  B  model-cached   model cache kept, compilation cache cleared
  C  compile-cached model cache kept, compilation cache kept -- a NEW PROCESS

C is a process restart, not a second request inside a warm process. A second
request in a warm process tells you nothing about whether compiled artifacts
persist to disk.

The published docs name three compilation-cache variables without saying how
they relate, so this script records what each directory holds before and
after every run.

Usage (on the TPU VM, inside the venv):
  ./measure_startup.py --condition B --repeats 3 --out results/startup_B.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home()
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", HOME / "vllm_torchtpu_cache"))
# Only this model's directory is ever removed, never the whole HF cache.
MODEL_REPO_DIR = "models--Qwen--Qwen3-8B"
HF_HUB = Path(os.environ.get("HF_HOME", HOME / ".cache" / "huggingface")) / "hub"

# Log lines worth pulling out. vLLM's exact wording moves between releases, so
# each milestone has several candidate patterns and the raw log is always kept.
MILESTONES = {
    "weights_load_s": [
        r"Loading weights took ([\d.]+) seconds",
        r"Model loading took .*? and ([\d.]+) seconds",
    ],
    "weights_gib": [
        r"Model loading took ([\d.]+) GiB",
        r"the model weights take ([\d.]+) GiB",
    ],
    "compile_s": [
        r"torch\.compile takes ([\d.]+) s in total",
        r"Compilation finished in ([\d.]+) s",
        r"graph compilation took ([\d.]+)",
    ],
    "kv_cache_tokens": [
        r"KV cache size: ([\d,]+) tokens",
        r"GPU KV cache size: ([\d,]+) tokens",
        r"Maximum concurrency for ([\d,]+) tokens",
    ],
    "kv_cache_gib": [
        r"KV cache.*?([\d.]+) GiB",
    ],
    "init_engine_s": [
        r"init engine .*?took ([\d.]+) seconds",
    ],
}


def dir_report(path: Path) -> dict:
    """File count and total size of a cache directory."""
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    files, total, links = 0, 0, 0
    exts: dict[str, int] = {}
    for p in path.rglob("*"):
        # Do NOT follow symlinks. The HF hub cache fills snapshots/ with
        # symlinks into blobs/, so following them counts every weight file
        # twice and reports double the real disk usage.
        if p.is_symlink():
            links += 1
            continue
        if p.is_file():
            files += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
            exts[p.suffix or "(none)"] = exts.get(p.suffix or "(none)", 0) + 1
    return {"exists": True, "files": files, "bytes": total,
            "symlinks": links, "by_extension": exts}


def cache_snapshot() -> dict:
    return {
        "tier3": dir_report(CACHE_ROOT / "tier3"),
        "vllm": dir_report(CACHE_ROOT / "vllm"),
        "xla": dir_report(CACHE_ROOT / "xla"),
        "hf_model": dir_report(HF_HUB / MODEL_REPO_DIR),
    }


def clear_compile_caches() -> None:
    """Remove only this experiment's compilation cache directories."""
    for sub in ("tier3", "vllm", "xla"):
        target = CACHE_ROOT / sub
        if target.exists():
            shutil.rmtree(target)


def clear_model_cache() -> None:
    """Remove only this model's directory from the HF hub cache."""
    target = HF_HUB / MODEL_REPO_DIR
    if target.exists():
        shutil.rmtree(target)


def wait_for_health(port: int, deadline: float) -> float | None:
    """Poll /health until it answers. Returns seconds waited, or None."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return None


def parse_log(text: str) -> dict:
    out: dict[str, float | int | str] = {}
    for key, patterns in MILESTONES.items():
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                raw = m.group(1).replace(",", "")
                try:
                    out[key] = float(raw) if "." in raw else int(raw)
                except ValueError:
                    out[key] = raw
                out[f"{key}__pattern"] = pat
                break
    # A persistent-cache run should show no recompilation.
    out["recompilation_warning"] = bool(
        re.search(r"recompilat", text, re.I)
    )
    out["cache_hit_mentions"] = len(re.findall(r"cache hit|loaded from cache|reusing", text, re.I))
    return out


def run_once(condition: str, port: int, logdir: Path, idx: int, timeout_s: int) -> dict:
    if condition == "A":
        clear_model_cache()
        clear_compile_caches()
    elif condition == "B":
        clear_compile_caches()
    elif condition == "C":
        pass  # keep everything
    else:
        raise SystemExit(f"unknown condition {condition!r}")

    before = cache_snapshot()
    logpath = logdir / f"serve_{condition}_{idx}.log"
    script = Path(__file__).resolve().parent / "serve.sh"

    t0 = time.monotonic()
    with logpath.open("wb") as logf:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, "PORT": str(port), "CACHE_ROOT": str(CACHE_ROOT)},
            start_new_session=True,
        )
        waited = wait_for_health(port, deadline=time.monotonic() + timeout_s)
        total = time.monotonic() - t0

        # Capture device memory while the server is up and idle.
        tpu_info = ""
        if waited is not None:
            try:
                tpu_info = subprocess.run(
                    ["tpu-info"], capture_output=True, text=True, timeout=60
                ).stdout
            except Exception as e:  # noqa: BLE001
                tpu_info = f"tpu-info failed: {e!r}"

        # SIGTERM the whole group so no worker process is orphaned.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=120)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    text = logpath.read_text(errors="replace")
    return {
        "condition": condition,
        "run": idx,
        "startup_seconds": round(total, 2) if waited is not None else None,
        "timed_out": waited is None,
        "timeout_budget_s": timeout_s,
        "log": str(logpath.name),
        "milestones": parse_log(text),
        "cache_before": before,
        "cache_after": cache_snapshot(),
        "tpu_info": tpu_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-run budget; condition A includes a ~16 GB download")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logdir", default="results/sanitized_logs")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    runs = []
    for i in range(1, args.repeats + 1):
        print(f"=== condition {args.condition} run {i}/{args.repeats} ===", flush=True)
        r = run_once(args.condition, args.port, logdir, i, args.timeout)
        print(json.dumps({k: r[k] for k in ("startup_seconds", "timed_out")}), flush=True)
        print(json.dumps(r["milestones"], indent=2), flush=True)
        runs.append(r)

    ok = sorted(r["startup_seconds"] for r in runs if r["startup_seconds"] is not None)
    summary = {
        "condition": args.condition,
        "repeats": args.repeats,
        "startup_seconds_all": [r["startup_seconds"] for r in runs],
        "startup_seconds_median": ok[len(ok) // 2] if ok else None,
        "runs": runs,
    }
    out = Path(args.out) if args.out else Path(f"results/startup_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    print(f"median startup: {summary['startup_seconds_median']} s")


if __name__ == "__main__":
    main()
