#!/usr/bin/env python3
"""Two-GPU half of the 2.15.0 release-note checks.

    torchrun --nproc_per_node=2 dist_verify.py --out results/dist-torch-2.15.0.json

Checks the distributed BC-breaks: the NCCL backend default, what
``all_gather``'s async future resolves to, the per-group device binding, and
FSDP2's gradient dtype when ``reduce_dtype`` is None. Rank 0 writes the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

RESULTS: dict[str, object] = {}


SKIP: set[str] = set()
ONLY: set[str] = set()


def check(name):
    def deco(fn):
        if name in SKIP or (ONLY and name not in ONLY and name != "env"):
            RESULTS[name] = "skipped"
            return fn
        try:
            RESULTS[name] = fn()
        except Exception as e:
            RESULTS[name] = {"raised": f"{type(e).__name__}: {str(e)[:220]}"}
        return fn
    return deco


def exc_of(fn) -> str:
    try:
        fn()
        return "no error"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).split(chr(10))[0][:200]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="",
                    help="comma-separated check names to skip. Before 2.15 the "
                         "cross-device check hangs instead of raising, which "
                         "would stop every later check from running.")
    ap.add_argument("--only", default="", help="comma-separated check names to run")
    args = ap.parse_args()
    skip = {x for x in args.skip.split(",") if x}
    only = {x for x in args.only.split(",") if x}

    SKIP.update(skip)
    ONLY.update(only)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    @check("env")
    def _env():
        return {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "cuda_runtime": torch.version.cuda,
            "world_size": world,
            "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
            "backend": dist.get_backend(),
        }

    @check("backend_registry")
    def _backends():
        """New NCCL backend is the default; the old one stays as nccl-legacy."""
        names = {}
        for name in ("nccl", "nccl-legacy", "gloo"):
            try:
                names[name] = dist.Backend(name)
            except Exception as e:
                names[name] = f"{type(e).__name__}: {str(e)[:80]}"
        return {"resolved": {k: str(v) for k, v in names.items()},
                "default_for_cuda": str(dist.get_backend())}

    @check("all_gather_future_shape")
    def _all_gather_future():
        """2.15 resolves the future to a per-rank list, not a stacked tensor."""
        t = torch.full((4,), float(rank), device="cuda")
        outs = [torch.empty(4, device="cuda") for _ in range(world)]
        work = dist.all_gather(outs, t, async_op=True)
        res = work.get_future().wait()
        # Unwrap the single-element wrapper the future API uses.
        inner = res[0] if isinstance(res, (list, tuple)) and len(res) == 1 else res
        def describe(v):
            if torch.is_tensor(v):
                return {"kind": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
            if isinstance(v, (list, tuple)):
                return {"kind": type(v).__name__, "len": len(v),
                        "first": describe(v[0]) if v else None}
            return {"kind": type(v).__name__}
        return {"future_result": describe(res), "unwrapped": describe(inner)}

    @check("cross_device_tensor")
    def _cross_device():
        """Each group is bound to one device; the other device must fail."""
        if torch.cuda.device_count() < 2:
            return "needs 2 visible devices"
        other = 1 - local_rank
        t = torch.ones(4, device=f"cuda:{other}")
        return exc_of(lambda: dist.all_reduce(t))

    @check("fsdp2_reduce_dtype_none")
    def _fsdp2():
        """With reduce_dtype=None the gradient now lands in grad_dtype."""
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)
        ).cuda()
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=None)
        fully_shard(model, mp_policy=policy)
        x = torch.randn(8, 64, device="cuda")
        model(x).sum().backward()
        grads = {n: str(p.grad.dtype) for n, p in model.named_parameters()
                 if p.grad is not None}
        params = {n: str(p.dtype) for n, p in model.named_parameters()}
        return {"param_dtypes": params, "grad_dtypes": grads,
                "policy": {"param_dtype": "bfloat16", "reduce_dtype": None}}

    dist.barrier()
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=2, ensure_ascii=False)
        print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
