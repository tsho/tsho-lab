# tsho-lab

Personal lab: benchmarks and experiments. Each directory is self-contained with its own README.

| Project | What it measures / does |
|---|---|
| [`pgvector-vs-lancedb`](pgvector-vs-lancedb/) | Vector store benchmark at matched recall — ingest, recall-latency curves, concurrency crossover, filtered-search planner behavior |
| [`distributed-training-bench`](distributed-training-bench/) | DeepSpeed ZeRO 0-3 vs PyTorch-native FSDP2 under one harness, NVLink vs PCIe (GCP) |
| [`jax-sharding-bench`](jax-sharding-bench/) | The ICI chapter: JAX/GSPMD replicated (DP) vs sharded (FSDP) on TPU — how much of the sharding tax an interconnect built for collectives removes |

License: Apache-2.0
