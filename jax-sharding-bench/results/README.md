# Results

Raw output of `bench.py` on a single-host **TPU v6e-4** (4 chips, ICI), one JSON object per run.
Common settings: bf16, `seq_len` 2048, 1 sequence per device (global batch 4), random-init transformer LM.

| file | sharding mode |
|---|---|
| `jax-dp-tpu-4dev.jsonl` | `--mode dp` (parameters replicated) |
| `jax-fsdp-tpu-4dev.jsonl` | `--mode fsdp` (parameters sharded on the `data` axis) |

The `tag` field identifies the model size and the software stack of each run:

| tag | model | Python | jax / jaxlib | libtpu |
|---|---|---|---|---|
| `0p8b` | 0.87B (`--dim 2048 --layers 16`, defaults) | 3.10.12 | 0.6.2 | 0.0.17 |
| `3p2b` | 3.27B (`--dim 3072 --layers 28 --heads 24`) | 3.10.12 | 0.6.2 | 0.0.17 |
| `0p8b-jax0111` | 0.87B | 3.12 | 0.11.1 | 0.0.46.1 |
| `3p2b-jax0111` | 3.27B | 3.12 | 0.11.1 | 0.0.46.1 |

Notes:

- All runs were taken on the same VM and the same chips (2026-09-06, `asia-northeast1-b`, Spot, runtime image `v2-alpha-tpuv6e`).
- The `3p2b` configuration was run twice per mode on jax 0.6.2; both rows are kept. Run-to-run difference is about 0.03%.
- Each configuration is otherwise a single run (n=1). Treat differences of a few percent with care.
