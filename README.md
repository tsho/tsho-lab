# tsho-lab

Personal lab: benchmarks and experiments. Each directory is self-contained with its own README.

| Project | What it measures / does |
|---|---|
| [`pgvector-vs-lancedb`](pgvector-vs-lancedb/) | Vector store benchmark at matched recall — ingest, recall-latency curves, concurrency crossover, filtered-search planner behavior |
| [`distributed-training-bench`](distributed-training-bench/) | DeepSpeed ZeRO 0-3 vs PyTorch-native FSDP2 under one harness, NVLink vs PCIe (GCP) |
| [`jax-sharding-bench`](jax-sharding-bench/) | The ICI chapter: JAX/GSPMD replicated (DP) vs sharded (FSDP) on TPU — how much of the sharding tax an interconnect built for collectives removes |
| [`torch-tpu-execution-modes`](torch-tpu-execution-modes/) | `torch_tpu==0.2.0` on TPU v6e-1: Strict Eager vs `DEFER_AND_FUSE` vs `torch.compile` vs `SingleTraceTrainer` across 7 configurations on a 0.87B Transformer |
| [`torch-tpu-graphtrainer-ladder`](torch-tpu-graphtrainer-ladder/) | Four compilation configurations for distributed training on TPU v6e-4 via TorchTitan — eager, per-block `torch.compile`, `TpuGraphTrainer`, and the optimizer in the step graph ([write-up](https://implicit-none.com/en/torch-tpu-distributed-graphtrainer/)) |
| [`torch-tpu-graphtrainer-xprof`](torch-tpu-graphtrainer-xprof/) | Where the `TpuGraphTrainer` speedup comes from on TPU v6e-4: eight `optimizer_compile` configurations timed and profiled (programs per step, device idle, collectives), logged memory vs device HBM, memory policies, batch size, the earlier commit re-run, and Llama 3 8B on v6e-8 |
| [`vllm-torchtpu-zero-to-serving`](vllm-torchtpu-zero-to-serving/) | `vllm-torchtpu==0.33.0` serving `Qwen/Qwen3-8B` on a single TPU v6e chip — the published v6e-1 recipe verified end to end, startup under three cache conditions, and which of the three documented compilation-cache variables actually works |
| [`pytorch-215-verification`](pytorch-215-verification/) | PyTorch 2.15.0 release-note claims checked against 2.14.1 on arm64 and 2x L4 — the NCCL `all_gather` future, complex ops moving to `TypeError`, and the float32 `sin`/`cos` and NEON reduction bit changes ([write-up](https://implicit-none.com/en/pytorch-215-release-notes/)) |

License: Apache-2.0
