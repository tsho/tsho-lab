# tsho-lab

Personal lab: benchmarks and experiments. Each directory is self-contained with its own README.

| Project | What it measures / does |
|---|---|
| [`pgvector-vs-lancedb`](pgvector-vs-lancedb/) | Vector store benchmark at matched recall — ingest, recall-latency curves, concurrency crossover, filtered-search planner behavior |
| [`distributed-training-bench`](distributed-training-bench/) | DeepSpeed ZeRO 0-3 vs PyTorch-native FSDP2 under one harness, NVLink vs PCIe (GCP) |
| [`jax-sharding-bench`](jax-sharding-bench/) | The ICI chapter: JAX/GSPMD replicated (DP) vs sharded (FSDP) on TPU — how much of the sharding tax an interconnect built for collectives removes |
| [`torch-tpu-execution-modes`](torch-tpu-execution-modes/) | `torch_tpu==0.2.0` on TPU v6e-1: Strict Eager vs `DEFER_AND_FUSE` vs `torch.compile` vs `SingleTraceTrainer` across 7 configurations on a 0.87B Transformer |
| [`pytorch-215-verification`](pytorch-215-verification/) | PyTorch 2.15.0 release-note claims checked against 2.14.1 and 2.15.0 on arm64 — exception-type changes, dropped APIs, and the float32 `sin`/`cos` and NEON reduction bit changes |

License: Apache-2.0
