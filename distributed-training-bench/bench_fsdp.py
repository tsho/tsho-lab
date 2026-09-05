"""PyTorch-native FSDP2, measured the same way as DeepSpeed ZeRO.

Why this file exists
--------------------
bench_zero_stages.py measures DeepSpeed's answer to "the model doesn't fit":
ZeRO stage 1/2/3. This file measures PyTorch's *native* answer to the same
problem: FSDP2 (the `torch.distributed.fsdp.fully_shard` API). They solve the
same thing — shard parameters, gradients, and optimizer state across ranks so a
model that doesn't fit on one GPU fits across N — but they are different
implementations with different performance, memory, and ergonomics.

Running both through the identical harness (same model, batch, profiler, results
schema in common.py) makes the comparison honest: the only variable is the
sharding engine.

FSDP2 mapping to ZeRO, roughly:
    reshard_after_forward=True   ~ ZeRO-3  (params re-gathered every forward, then freed)
    reshard_after_forward=False  ~ ZeRO-2  (params kept resident after gather)
    CPU offload                  ~ ZeRO-3 + offload
There is no exact FSDP equivalent of ZeRO-1 (optimizer-state-only sharding);
FSDP always shards params + grads. So the honest comparison points are
ZeRO-3 vs FSDP2(reshard=True) and ZeRO-2 vs FSDP2(reshard=False).

Launch (needs torchrun, NOT the deepspeed launcher):
    torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward
    torchrun --nproc_per_node=4 bench_fsdp.py --no-reshard-after-forward
    torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward --offload

Requires torch >= 2.4 for the fully_shard (FSDP2) API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer

from common import (
    build_model,
    describe_env,
    local_rank,
    log,
    nccl_share,
    rank,
    synthetic_batch,
    world_size,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def find_transformer_layers(model) -> list:
    """Locate the repeated decoder blocks to shard individually.

    FSDP2 works best when each transformer layer is its own FSDP unit, so
    parameters are gathered per-layer just-in-time rather than all at once. HF
    causal LMs expose these as model.model.layers (Llama/Qwen/Mistral family)."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    return []


def apply_fsdp2(model, dtype: torch.dtype, reshard: bool, offload: bool):
    """Wrap each transformer layer, then the root, with FSDP2 fully_shard."""
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    kwargs = dict(mp_policy=mp, reshard_after_forward=reshard)
    if offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()

    layers = find_transformer_layers(model)
    if not layers:
        log("WARNING: could not find transformer layers; sharding root only")
    for layer in layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)  # root wraps embeddings / lm_head / the rest
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--reshard-after-forward", dest="reshard", action="store_true", default=True,
                   help="ZeRO-3-like: re-gather params each forward, then free (default)")
    p.add_argument("--no-reshard-after-forward", dest="reshard", action="store_false",
                   help="ZeRO-2-like: keep params resident after gather")
    p.add_argument("--offload", action="store_true", help="CPU offload params+grads+optim")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--real-weights", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())
    device = torch.device("cuda", local_rank())
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    strategy = f"fsdp2-{'reshard' if args.reshard else 'noreshard'}" + ("-offload" if args.offload else "")
    log(f"model={args.model} strategy={strategy} world={world_size()} "
        f"micro_bs={args.micro_batch} seq={args.seq_len} dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = len(tok)

    t_build = time.time()
    model = build_model(args.model, dtype, random_init=not args.real_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built in {time.time() - t_build:.0f}s, {n_params / 1e9:.2f}B params")

    apply_fsdp2(model, dtype, reshard=args.reshard, offload=args.offload)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)

    torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    oom = False
    step = 0
    try:
        for step in range(args.steps):
            batch = synthetic_batch(args.micro_batch, args.seq_len, vocab, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= args.warmup:
                step_times.append(dt)
            if rank() == 0 and step % 10 == 0:
                log(f"  step {step:>3} {dt * 1000:8.1f} ms")
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        log(f"!! OOM at step {step}: {str(e)[:200]}")

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    prof_result = None
    if args.profile and not oom:
        log("profiling window (5 steps) ...")
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                batch = synthetic_batch(args.micro_batch, args.seq_len, vocab, device)
                optim.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
        prof_result = nccl_share(prof)
        log(f"  NCCL share: {prof_result['comm_fraction']:.1%} of CUDA time")

    tokens_per_step_per_rank = args.micro_batch * args.seq_len
    if step_times:
        import statistics
        median_s = statistics.median(step_times)
        tps_global = tokens_per_step_per_rank * world_size() / median_s
    else:
        median_s = None
        tps_global = None

    # Same schema as bench_zero_stages.py, with `strategy` in place of
    # `zero_stage` so both land in the same results file and summarize together.
    record = {
        "tag": args.tag,
        "model": args.model,
        "params_b": round(n_params / 1e9, 3),
        "strategy": strategy,
        "zero_stage": None,
        "offload": args.offload,
        "world_size": world_size(),
        "micro_batch": args.micro_batch,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "oom": oom,
        "steps_measured": len(step_times),
        "step_ms_median": round(median_s * 1000, 1) if median_s else None,
        "step_ms_p10": round(min(step_times) * 1000, 1) if step_times else None,
        "step_ms_p90": round(max(step_times) * 1000, 1) if step_times else None,
        "tokens_per_sec_global": round(tps_global) if tps_global else None,
        "peak_alloc_gb": round(peak_alloc, 2),
        "peak_reserved_gb": round(peak_reserved, 2),
        "profile": prof_result,
        "env": describe_env(),
    }

    if rank() == 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        env = record["env"]
        gpu_slug = env["gpu_name"].replace(" ", "-").replace("/", "-")
        out = RESULTS_DIR / f"{env['host']}-{gpu_slug}-{env['gpu_count']}gpu.jsonl"
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")

        log("=" * 62)
        log(f"{strategy}{'  [OOM]' if oom else ''}")
        if not oom:
            log(f"  throughput      {record['tokens_per_sec_global']:,} tok/s (global)")
            log(f"  step time       {record['step_ms_median']} ms median")
        log(f"  peak allocated  {record['peak_alloc_gb']} GB / rank")
        log(f"  peak reserved   {record['peak_reserved_gb']} GB / rank")
        if prof_result:
            log(f"  NCCL share      {prof_result['comm_fraction']:.1%}")
        log(f"  -> {out}")
        log("=" * 62)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
