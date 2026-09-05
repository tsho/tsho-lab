"""Measure what each DeepSpeed ZeRO stage actually costs.

What this answers
-----------------
ZeRO stage 1/2/3 (and plain DDP) trade GPU memory for communication. The
documentation says so; this script measures how much, on your hardware, for your
model. Three numbers per configuration:

    throughput     tokens/sec across all ranks
    peak memory    max allocated + max reserved per rank
    NCCL share     fraction of GPU time spent in collective communication

The third one is the interesting one. ZeRO-3 shards parameters, so every forward
pass must all-gather them back. On NVLink that is cheap. On PCIe-only hardware it
can dominate. Running this script on a2-ultragpu-4g (A100, NVLink) and on
g2-standard-48 (L4, PCIe only) with everything else identical isolates exactly
that difference.

Why synthetic data
------------------
Batches are random token ids. This is a systems benchmark, not a training run --
we care about step time and memory, and real data would add dataloader noise and
a HF datasets dependency for no benefit. Loss values here are meaningless by
design; do not report them. Pass --real-data to use a small real corpus instead
if you want a sanity check that the loss moves.

Run
---
    # single config
    deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --model Qwen/Qwen2.5-3B

    # with the profiler pass (slower; adds the NCCL share number)
    deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --profile

    # baseline without ZeRO (plain data parallel)
    deepspeed --num_gpus=4 bench_zero_stages.py --stage 0

Results are appended to results/<host>-<gpu>-<n>gpu.jsonl, one line per run, so a
sweep can be assembled afterwards without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DS_CONFIG_DIR = HERE / "ds_configs"



# Shared helpers live in common.py so the ZeRO and FSDP benches measure
# identically. Only the sharding engine differs between the two files.
from common import (  # noqa: E402
    build_model,
    describe_env,
    local_rank,
    log,
    nccl_share,
    rank,
    synthetic_batch,
    world_size,
)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    # deepspeed launcher passes --local_rank=N; we read LOCAL_RANK from env instead but must accept the flag
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3],
                   help="ZeRO stage. 0 = plain data parallel baseline.")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--offload", action="store_true",
                   help="ZeRO-3 optimizer+param offload to CPU")
    p.add_argument("--profile", action="store_true",
                   help="run a short profiled window to measure NCCL share")
    p.add_argument("--real-weights", action="store_true",
                   help="download real weights instead of random init")
    p.add_argument("--tag", default="", help="free-form label recorded in results")
    args = p.parse_args()

    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank())
    device = torch.device("cuda", local_rank())

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg_path = DS_CONFIG_DIR / f"zero{args.stage}.json"
    ds_config = json.loads(cfg_path.read_text())
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch
    ds_config["gradient_accumulation_steps"] = 1
    if args.offload:
        if args.stage != 3:
            raise SystemExit("--offload only applies to --stage 3")
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

    log(f"model={args.model} stage={args.stage} world={world_size()} "
        f"micro_bs={args.micro_batch} seq={args.seq_len} dtype={args.dtype} "
        f"offload={args.offload}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = len(tok)

    t_build = time.time()
    model = build_model(args.model, dtype, random_init=not args.real_weights)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built in {time.time() - t_build:.0f}s, {n_params / 1e9:.2f}B params")

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    oom = False
    try:
        for step in range(args.steps):
            batch = synthetic_batch(args.micro_batch, args.seq_len, vocab, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = engine(**batch).loss
            engine.backward(loss)
            engine.step()
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
                loss = engine(**batch).loss
                engine.backward(loss)
                engine.step()
            torch.cuda.synchronize()
        prof_result = nccl_share(prof)
        log(f"  NCCL share: {prof_result['comm_fraction']:.1%} of CUDA time")

    # Aggregate throughput across ranks.
    tokens_per_step_per_rank = args.micro_batch * args.seq_len
    if step_times:
        median_s = statistics.median(step_times)
        tps_global = tokens_per_step_per_rank * world_size() / median_s
    else:
        median_s = None
        tps_global = None

    record = {
        "tag": args.tag,
        "model": args.model,
        "params_b": round(n_params / 1e9, 3),
        "zero_stage": args.stage,
        "strategy": f"zero-{args.stage}" + ("-offload" if args.offload else ""),
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
        log(f"stage {args.stage}{' +offload' if args.offload else ''}"
            f"{'  [OOM]' if oom else ''}")
        if not oom:
            log(f"  throughput      {record['tokens_per_sec_global']:,} tok/s (global)")
            log(f"  step time       {record['step_ms_median']} ms median")
        log(f"  peak allocated  {record['peak_alloc_gb']} GB / rank")
        log(f"  peak reserved   {record['peak_reserved_gb']} GB / rank")
        if prof_result:
            log(f"  NCCL share      {prof_result['comm_fraction']:.1%}")
        log(f"  -> {out}")
        log("=" * 62)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
