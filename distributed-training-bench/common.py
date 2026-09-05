"""Shared helpers for the ZeRO and FSDP benchmarks.

Both bench_zero_stages.py (DeepSpeed) and bench_fsdp.py (PyTorch-native) import
from here so their measurements are directly comparable — same model builder,
same synthetic batch, same environment description, same NCCL-share profiler,
same results schema. The only difference between the two is the sharding engine.
"""

from __future__ import annotations

import os
import platform
import socket

import torch
from transformers import AutoConfig, AutoModelForCausalLM


# --------------------------------------------------------------------------- #
# rank helpers (torchrun / deepspeed both set these env vars)
# --------------------------------------------------------------------------- #
def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[bench] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# environment description -- goes into every result so runs are comparable
# --------------------------------------------------------------------------- #
def describe_env() -> dict:
    props = torch.cuda.get_device_properties(local_rank())
    # p2p_matrix: can the GPUs talk directly (NVLink / P2P over PCIe) or must
    # they round-trip through host memory? The single most important hardware
    # fact for interpreting the NCCL share.
    n = torch.cuda.device_count()
    p2p = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(bool(torch.cuda.can_device_access_peer(i, j)) if i != j else True)
        p2p.append(row)

    nvlink = None
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(local_rank())
        active = 0
        for link in range(18):  # NVML caps at 18 links
            try:
                if pynvml.nvmlDeviceGetNvLinkState(h, link) == 1:
                    active += 1
            except pynvml.NVMLError:
                break
        nvlink = active
        pynvml.nvmlShutdown()
    except Exception:
        pass

    ds_version = None
    try:
        import deepspeed

        ds_version = deepspeed.__version__
    except Exception:
        pass

    return {
        "host": socket.gethostname(),
        "gpu_name": props.name,
        "gpu_count": n,
        "gpu_mem_gb": round(props.total_memory / 1024**3, 1),
        "nvlink_active_links": nvlink,
        "p2p_matrix": p2p,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "deepspeed": ds_version,
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------------- #
# model / data
# --------------------------------------------------------------------------- #
def build_model(model_name: str, dtype: torch.dtype, random_init: bool):
    """Load the model. random_init skips downloading weights -- the systems
    behaviour is identical and it makes a cold run much faster."""
    if random_init:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        cfg.torch_dtype = dtype
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        model = model.to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True
        )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model


def synthetic_batch(bs: int, seq_len: int, vocab: int, device) -> dict:
    ids = torch.randint(0, vocab, (bs, seq_len), device=device)
    return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}


# --------------------------------------------------------------------------- #
# profiling -- how much GPU time is collective communication
# --------------------------------------------------------------------------- #
NCCL_HINTS = ("nccl", "AllGather", "ReduceScatter", "AllReduce", "Broadcast")


def nccl_share(prof) -> dict:
    """Fraction of total CUDA kernel time spent in NCCL collectives."""
    total = 0.0
    comm = 0.0
    per_op: dict[str, float] = {}
    for ev in prof.key_averages():
        cuda_us = getattr(ev, "self_device_time_total", 0) or 0
        if cuda_us <= 0:
            continue
        total += cuda_us
        if any(h.lower() in ev.key.lower() for h in NCCL_HINTS):
            comm += cuda_us
            per_op[ev.key] = per_op.get(ev.key, 0.0) + cuda_us
    top = sorted(per_op.items(), key=lambda kv: -kv[1])[:8]
    return {
        "cuda_total_ms": round(total / 1000, 2),
        "cuda_comm_ms": round(comm / 1000, 2),
        "comm_fraction": round(comm / total, 4) if total else None,
        "top_comm_ops": [{"op": k, "ms": round(v / 1000, 2)} for k, v in top],
    }
