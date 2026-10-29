#!/usr/bin/env python3
"""Single-GPU half of the 2.15.0 release-note checks. Run under both versions.

    pt2141/bin/python gpu_verify.py > results/gpu-torch-2.14.1.json
    pt2150/bin/python gpu_verify.py > results/gpu-torch-2.15.0.json

Covers the CUDA items that one device can show, including the
linspace/logspace precision claim that produced identical output on CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys

import torch

RESULTS: dict[str, object] = {}


def check(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
        except Exception as e:
            RESULTS[name] = {"raised": f"{type(e).__name__}: {str(e)[:200]}"}
        return fn
    return deco


def exc_of(fn) -> str:
    try:
        fn()
        return "no error"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).split(chr(10))[0][:160]}"


@check("env")
def _env():
    return {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "driver_cuda": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[:1],
    }


@check("linspace_logspace_cuda")
def _linspace_cuda():
    """CPU gave identical output in both versions; the note mentions CUDA."""
    out = {}
    for dt in ("float16", "bfloat16", "float32"):
        d = getattr(torch, dt)
        ls = torch.linspace(0, 1, 17, dtype=d, device="cuda")
        lg = torch.logspace(0, 3, 13, dtype=d, device="cuda")
        out[dt] = {
            "linspace": [float(v) for v in ls.float().cpu()],
            "logspace": [float(v) for v in lg.float().cpu()],
        }
        # Distance from the float64 ideal, summed, to make a drift visible.
        ideal_ls = torch.linspace(0, 1, 17, dtype=torch.float64)
        out[dt]["linspace_abs_err_sum"] = float(
            (ls.double().cpu() - ideal_ls).abs().sum())
    return out


@check("histc_out_dtype")
def _histc():
    x = torch.randn(1000, device="cuda")
    out_f64 = torch.empty(10, dtype=torch.float64, device="cuda")
    out_same = torch.empty(10, dtype=torch.float32, device="cuda")
    return {
        "mismatched_out_dtype": exc_of(lambda: torch.histc(x, bins=10, out=out_f64)),
        "matching_out_dtype": exc_of(lambda: torch.histc(x, bins=10, out=out_same)),
    }


@check("cublas_workspace_memory")
def _cublas():
    """Workspaces moved from cached to per-operation allocation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_reserved = torch.cuda.memory_reserved()
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_reserved()
    for _ in range(5):
        (a @ b).sum().item()
    torch.cuda.synchronize()
    after_mm = torch.cuda.memory_reserved()
    return {
        "reserved_base_mib": round(base_reserved / 2**20, 2),
        "reserved_after_tensors_mib": round(after_alloc / 2**20, 2),
        "reserved_after_matmuls_mib": round(after_mm / 2**20, 2),
        "workspace_growth_mib": round((after_mm - after_alloc) / 2**20, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }


@check("lu_factor_backend")
def _lu():
    """MAGMA deprecated for pivoted LU on CUDA; cuSOLVER used instead."""
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", dtype=torch.float64)
    lu, piv = torch.linalg.lu_factor(a)
    # torch.linalg has no lu_reconstruct; unpack and multiply back.
    P, L, U = torch.lu_unpack(lu, piv)
    recon = P @ L @ U
    pref = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    return {
        "reconstruction_max_abs_err": float((recon - a).abs().max()),
        "lu_checksum": round(float(lu.abs().sum()), 10),
        "pivots_checksum": int(piv.sum()),
        "preferred_linalg_library": str(pref()) if callable(pref) else "missing",
    }


@check("graph_annotations_api")
def _graph_ann():
    ga = getattr(torch.cuda, "graph_annotations", None)
    if ga is None:
        return "torch.cuda.graph_annotations missing"
    return {"has_clear_kernel_annotations": hasattr(ga, "clear_kernel_annotations"),
            "members": [m for m in dir(ga) if not m.startswith("_")][:12]}


@check("curand_dependency")
def _curand():
    """Linux CUDA wheels no longer depend on the cuRAND library."""
    import glob
    import os
    so = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"))
    if not so:
        return "libtorch_cuda.so not found"
    out = subprocess.run(["ldd", so[0]], capture_output=True, text=True).stdout
    return {
        "links_curand": "curand" in out,
        "cuda_libs": sorted({l.split("=>")[0].strip() for l in out.splitlines()
                             if "libcu" in l or "libnv" in l})[:12],
    }


if __name__ == "__main__":
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
