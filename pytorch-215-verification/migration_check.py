#!/usr/bin/env python3
"""Check that the recommended all_gather migration works on both versions.

    torchrun --nproc_per_node=2 migration_check.py --out results/migration-<ver>.json

Runs on **gloo over CPU**, so it is portable. It validates the migration
pattern — read the output list after ``work.wait()`` rather than the future —
not the NCCL future's shape, which is measured separately in dist_verify.py on
real GPUs.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def describe(v):
    if torch.is_tensor(v):
        return {"kind": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return {"kind": type(v).__name__, "len": len(v),
                "first": describe(v[0]) if v else None}
    return {"kind": type(v).__name__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    local_tensor = torch.full((4,), float(rank))
    outputs = [torch.empty_like(local_tensor) for _ in range(world)]

    work = dist.all_gather(outputs, local_tensor, async_op=True)
    work.wait()
    gathered = torch.stack(outputs, dim=0)

    # What the future resolves to on this backend, for contrast only.
    local2 = torch.full((4,), float(rank))
    outs2 = [torch.empty_like(local2) for _ in range(world)]
    w2 = dist.all_gather(outs2, local2, async_op=True)
    future_result = w2.get_future().wait()

    result = {
        "env": {"torch": torch.__version__, "backend": dist.get_backend(),
                "world_size": world},
        "migration_pattern": {
            "gathered_shape": list(gathered.shape),
            "gathered_rows": [[float(x) for x in row] for row in gathered],
            "correct": gathered.shape == torch.Size([world, 4])
            and all(float(gathered[r][0]) == r for r in range(world)),
        },
        "future_result_on_this_backend": describe(future_result),
    }

    dist.barrier()
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
