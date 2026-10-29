#!/usr/bin/env python3
"""Check PyTorch 2.15.0's release-note claims that are observable on a CPU/MPS host.

Run the same file under 2.14.x and 2.15.0 and diff the JSON:

    pt2141/bin/python verify.py > r_2141.json
    pt2150/bin/python verify.py > r_2150.json

Each check returns a small JSON-safe value rather than a pass/fail, so the
comparison shows what actually changed instead of trusting the notes.
CUDA, ROCm, XPU and distributed-backend claims are not covered here.
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import warnings

import torch

RESULTS: dict[str, object] = {}


def check(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
        except Exception as e:  # a raising check is itself a result
            RESULTS[name] = {"raised": f"{type(e).__name__}: {e}"}
        return fn
    return deco


def exc_of(fn) -> str:
    """Run fn and report the exception type and message, or 'no error'."""
    try:
        fn()
        return "no error"
    except Exception as e:
        msg = str(e).split("\n")[0][:160]
        return f"{type(e).__name__}: {msg}"


# ---------------------------------------------------------------- environment

@check("env")
def _env():
    return {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "mps_available": torch.backends.mps.is_available(),
    }


# ------------------------------------------------- Python frontend / BC break

@check("reload_torch")
def _reload():
    import importlib
    return exc_of(lambda: importlib.reload(torch))


@check("complex_order_ops")
def _complex_ops():
    z = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex64)
    ops = {
        "sort": lambda: torch.sort(z),
        "argsort": lambda: torch.argsort(z),
        "topk": lambda: torch.topk(z, 1),
        "argmax": lambda: torch.argmax(z),
        "argmin": lambda: torch.argmin(z),
    }
    return {name: exc_of(fn) for name, fn in ops.items()}


@check("split_unbind_grad_fn")
def _grad_fn():
    x = torch.randn(6, requires_grad=True)
    out = {}
    out["split"] = type(torch.split(x, 2)[0].grad_fn).__name__
    out["split_with_sizes"] = type(torch.split_with_sizes(x, [2, 4])[0].grad_fn).__name__
    out["unbind"] = type(torch.unbind(x.view(3, 2))[0].grad_fn).__name__

    # The note is about *regenerated* outputs, i.e. under functionalization,
    # not plain eager. Observe that path separately.
    try:
        from torch._subclasses.functional_tensor import FunctionalTensorMode
        with FunctionalTensorMode():
            y = torch.randn(6, requires_grad=True)
            out["split_functionalized"] = type(torch.split(y, 2)[0].grad_fn).__name__
            out["unbind_functionalized"] = type(
                torch.unbind(y.view(3, 2))[0].grad_fn).__name__
    except Exception as e:
        out["functionalized"] = f"{type(e).__name__}: {str(e)[:100]}"
    return out


@check("optimizer_bare_set")
def _opt_set():
    p = torch.nn.Parameter(torch.zeros(2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            torch.optim.SGD({p}, lr=0.1)
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:120]}"
        return [f"{x.category.__name__}: {str(x.message)[:120]}" for x in w] or "no warning"


@check("weights_only_nested_tensor")
def _weights_only():
    nt = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
    buf = io.BytesIO()
    torch.save({"nt": nt}, buf)
    buf.seek(0)
    return exc_of(lambda: torch.load(buf, weights_only=True))


# ------------------------------------------------------------- new public API

@check("new_apis")
def _new_apis():
    def probe(path):
        obj = torch
        for part in path.split(".")[1:]:
            if not hasattr(obj, part):
                return "missing"
            obj = getattr(obj, part)
        return type(obj).__name__
    return {p: probe(p) for p in (
        "torch.foreach",
        "torch.optim.functional",
        "torch.mps.Stream",
        "torch.mps.current_stream",
        "torch.mps.default_stream",
        "torch.mps.set_stream",
        "torch.compiler.precompile",
    )}


@check("precompile_members")
def _precompile():
    pc = getattr(torch.compiler, "precompile", None)
    if pc is None:
        return "missing"
    return {"type": type(pc).__name__,
            "has_capture": hasattr(pc, "capture"),
            "has_load": hasattr(pc, "load")}


@check("pipelining_rename")
def _pipelining():
    import importlib, pkgutil
    import torch.distributed.pipelining as pp
    found = {}
    for mod in [pp] + [importlib.import_module(f"{pp.__name__}.{m.name}")
                       for m in pkgutil.iter_modules(pp.__path__)]:
        for name in dir(mod):
            if "per_direction_p2p" in name or "per_edge_p2p" in name:
                found.setdefault(name, mod.__name__.split(".")[-1])
    return found or "neither name found in torch.distributed.pipelining"


@check("profiler_metadata_json")
def _profiler():
    from torch.autograd.profiler_util import FunctionEvent
    cls = {k: hasattr(FunctionEvent, k)
           for k in ("metadata_json", "metadata", "event_metadata")}
    # Instance attributes do not show up on the class, so profile for real.
    with torch.profiler.profile() as prof:
        torch.randn(64, 64) @ torch.randn(64, 64)
    ev = prof.events()[0]
    inst = {k: hasattr(ev, k) for k in ("metadata_json", "metadata", "event_metadata")}
    return {"on_class": cls, "on_instance": inst}


# ------------------------------------------------- silent numerical behaviour

def ulp_error(got: torch.Tensor, ref_f64: torch.Tensor, floor: float = 0.5) -> float:
    """Max error in float32 ULPs, ignoring points where the reference is near
    zero. At a zero of sin the result is tiny while the argument-reduction
    error is not, so the ULP ratio there says nothing about kernel quality."""
    import numpy as np
    keep = ref_f64.abs() >= floor
    g = got.double()[keep].numpy()
    r = ref_f64[keep].numpy()
    # np.spacing on float32 gives the float32 ULP. math.ulp would give the
    # float64 ULP and inflate every ratio by about 2**29.
    ulp = np.spacing(r.astype(np.float32)).astype(np.float64)
    return float((np.abs(g - r) / ulp).max())


@check("sin_cos_f32_ulp")
def _sincos():
    torch.manual_seed(0)
    x = torch.linspace(-20, 20, 20001, dtype=torch.float64)
    x32 = x.float()
    out = {}
    for op in ("sin", "cos"):
        ref = getattr(torch, op)(x32.double())  # f64 math on the f32 inputs
        got = getattr(torch, op)(x32)
        out[op] = {
            "max_ulp": round(ulp_error(got, ref), 4),
            # a stable fingerprint of the exact bits produced
            "bits_sha": __import__("hashlib").sha256(
                got.numpy().tobytes()).hexdigest()[:16],
        }
    return out


@check("sum_norm_f32_bits")
def _sum_norm():
    torch.manual_seed(0)
    x = torch.randn(1 << 16, dtype=torch.float32)
    def bits(t):
        return struct.pack("<f", float(t)).hex()
    return {
        "sum": bits(x.sum()),
        "norm": bits(x.norm()),
        "sum_dim0_4096x16": bits(x.view(4096, 16).sum(0).sum()),
    }


@check("linspace_logspace_low_precision")
def _linspace():
    out = {}
    for dt in ("float16", "bfloat16"):
        d = getattr(torch, dt)
        ls = torch.linspace(0, 1, 17, dtype=d)
        lg = torch.logspace(0, 3, 13, dtype=d)
        out[dt] = {
            "linspace": [round(float(v), 6) for v in ls],
            "logspace": [round(float(v), 4) for v in lg],
        }
    return out


if __name__ == "__main__":
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
