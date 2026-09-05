# distributed-training-bench

What sharded training actually costs — **DeepSpeed ZeRO vs PyTorch-native FSDP2**,
measured on GCP under one harness.

The companion to [`pytorch-quality-bench`](https://github.com/tsho/pytorch-quality-bench)
(inference side) — this one measures the training side.

## The two questions

**1. ZeRO vs FSDP2 — same problem, two engines.** DeepSpeed ZeRO and PyTorch's
native FSDP2 both shard parameters, gradients, and optimizer state so a model too
big for one GPU fits across N. They are different implementations. Which is
faster / lighter / less communication-bound, for the same model on the same
hardware? Run both through the identical harness and find out.

| DeepSpeed | ~ | PyTorch FSDP2 |
|---|---|---|
| ZeRO-3 | ≈ | `fully_shard(reshard_after_forward=True)` |
| ZeRO-2 | ≈ | `fully_shard(reshard_after_forward=False)` |
| ZeRO-1 | (no exact FSDP equivalent) | — |
| ZeRO-3 + offload | ≈ | `+ CPUOffloadPolicy` |

**2. The interconnect decides the price.** Sharding trades GPU memory for
communication — stage 3 / reshard re-gathers params every forward. Whether that
is free or ruinous depends on how the GPUs are wired.

So: run the identical workload on hardware that differs *only* in interconnect.

| | GPUs | Interconnect |
|---|---|---|
| `a2-ultragpu-4g` | A100 80GB × 4 | **NVLink** (~600 GB/s) |
| `g2-standard-48` | L4 24GB × 4 | **PCIe Gen4 only** (~64 GB/s) |

Three numbers per configuration: throughput, peak memory per rank, and the
fraction of GPU time spent inside NCCL collectives.

## Quick start

```bash
# 1. check quota (reuse the script from vllm-benchmark-scripts)
PROJECT=my-proj ./check-quota.sh gpu

# 2. bring up a box (SPOT=1 is ~1/3 the price and fine for this)
SPOT=1 PROJECT=my-proj ZONE=us-central1-a TYPE=a100x4 ./launch-gpu.sh

# 3. copy and set up
gcloud compute scp --recurse setup.sh common.py bench_zero_stages.py bench_fsdp.py \
    ds_configs run_all.sh summarize.py \
    dtb-a100x4:~/ --zone=us-central1-a --project=my-proj
gcloud compute ssh dtb-a100x4 --zone=us-central1-a --project=my-proj

# on the VM
HF_TOKEN=hf_xxx bash setup.sh

# DeepSpeed ZeRO 0/1/2/3 (+offload)
bash run_all.sh

# PyTorch-native FSDP2 — launched with torchrun, not the deepspeed launcher
torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward --profile     # ~ ZeRO-3
torchrun --nproc_per_node=4 bench_fsdp.py --no-reshard-after-forward --profile  # ~ ZeRO-2
torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward --offload --profile

# 4. same thing on the PCIe box, then compare ZeRO and FSDP2 side by side
python3 summarize.py --markdown --plot
```

Both engines write to the same `results/*.jsonl` and `summarize.py` prints them in
one table, so ZeRO-3 vs FSDP2(reshard) is a direct read.

**Delete the instance when done.** `gcloud compute instances delete dtb-a100x4 --zone=... --quiet`

## What gets measured

| Metric | Why it's here |
|---|---|
| `tokens_per_sec_global` | the headline. all ranks combined |
| `step_ms_median` / p10 / p90 | median, because Spot preemption and thermal noise make means lie |
| `peak_alloc_gb` | what ZeRO actually bought you |
| `peak_reserved_gb` | allocator reservation — the number that decides whether the *next* config OOMs |
| `profile.comm_fraction` | NCCL kernels as a share of total CUDA time. **the interesting one** |
| `profile.top_comm_ops` | which collective dominates. ZeRO-3 should show all-gather |
| `oom` | a config that OOMs is a result, not a failure. it gets recorded and the sweep continues |
| `env.nvlink_active_links` | read from NVML — proves which side of the comparison you're on |
| `env.p2p_matrix` | whether GPUs can reach each other directly at all |

Results append to `results/<host>-<gpu>-<n>gpu.jsonl`, one line per run.
Nothing is overwritten, so a sweep can be resumed or extended freely.

## Configurations

`run_all.sh` covers:

| Config | What it shards |
|---|---|
| stage 0 | nothing — plain data parallel. the baseline |
| stage 1 | optimizer state |
| stage 2 | optimizer state + gradients |
| stage 3 | optimizer state + gradients + **parameters** |
| stage 3 + offload | the above, plus optimizer and params pushed to CPU |

The offload run is there to show the other end of the trade: memory stops being
the constraint and PCIe becomes it.

## Synthetic data, on purpose

Batches are random token ids. This is a systems benchmark — step time and memory
are the signal, and a real dataloader would add noise and a dependency for no
benefit.

**Loss values here are meaningless. Do not report them.** Pass `--real-weights`
if you want real weights loaded (systems behaviour is the same; it just takes
longer to start).

## Notes

- Gradient checkpointing is on for every config. Without it, stage 0 OOMs early
  and the comparison loses its baseline.
- `use_cache=False` — incompatible with gradient checkpointing.
- The DLVM image ships a CUDA-optimized torch. `setup.sh` deliberately does not
  reinstall torch; it installs DeepSpeed against whatever is already there.
- L4 is 24GB vs A100's 80GB, so the two boxes will not OOM at the same point.
  That asymmetry is itself worth reporting — pick a model size that fits both if
  you want a like-for-like throughput comparison (Qwen2.5-3B at seq 2048 does).
