# torch-tpu-graphtrainer-ladder

Four compilation configurations for distributed training on a **TPU v6e-4**, measured through
TorchTitan's TorchTPU path on one model and one harness.

Write-up:
- [TorchTPU x TorchTitan distributed training on TPU v6e-4](https://implicit-none.com/en/torch-tpu-distributed-graphtrainer/) (EN)
- [JA version](https://implicit-none.com/ja/torch-tpu-distributed-graphtrainer/)

## The four configurations

They are **not a stack**. Rungs 2 and 3 are mutually exclusive compilation strategies — with
`compile_mode="aot_fx_trace"` GraphTrainer does not wrap blocks in `torch.compile` — and only
rung 4 adds to rung 3.

| # | Configuration | Flags |
|---|---|---|
| 1 | eager | `--tpu_config.eager_mode=DEFER_NEVER` |
| 2 | per-block `torch.compile` | `--compile.enable --compile.backend=tpu` |
| 3 | `TpuGraphTrainer` | `--tpu_config.use_graph_trainer --tpu_config.compile_mode=aot_fx_trace --tpu_config.graph_trainer_memory_policy=default` |
| 4 | 3 + optimizer in graph | rung 3 plus `--tpu_config.optimizer_in_graph` |

> **These config names were renamed upstream after this run.** They are reproduced as invoked
> against the pinned commit. On current main: `compile.mode`, `memory_policy`,
> `optimizer_compile="in_graph"`, and `TpuGraphTrainer.Config.from_tpu_trainer_config(...)`.

## Results

Model: torchtitan llama3 `1B` config (`dim=2048`, `n_layers=16`, `n_heads=32`, `n_kv_heads=8`,
`vocab=128256`), weight tying disabled (see `bench_cfg.py`). `seq_len=2048`, SimpleFSDP,
30 steps with the first 10 discarded, median reported. `tok/s` is TorchTitan's **per-chip**
figure; multiply by four for the slice.

**2,048 tokens per device** (`results/ladder-batch1.json`):

| # | Configuration | Step | tok/s/chip | MFU | `max_reserved` | vs eager |
|---|---|---:|---:|---:|---:|---:|
| 1 | eager | 320.0 ms | 6,387 | 5.72% | 0.98 GiB | 1.00x |
| 2 | per-block compile | 200.0 ms | 10,339 | 9.25% | 0.98 GiB | 1.62x |
| 3 | GraphTrainer | 190.0 ms | 10,584 | 9.48% | 13.40 GiB | 1.66x |
| 4 | + optimizer in graph | 160.0 ms | 13,131 | 11.76% | 13.42 GiB | **2.06x** |

**8,192 tokens per device** (`results/ladder-batch4.json`):

| # | Step | tok/s/chip | MFU | `max_reserved` | vs eager |
|---|---:|---:|---:|---:|---:|
| 1 eager | 740.0 ms | 11,019 | 9.87% | 3.91 GiB | 1.00x |
| 2 per-block compile | 530.0 ms | 15,369 | 13.76% | 3.91 GiB | 1.39x |
| 4 GraphTrainer + optimizer | 420.0 ms | 19,395 | 17.37% | 19.69 GiB | **1.76x** |

Final loss agrees across all four at the first batch size (`3.36297`–`3.36403`), within bf16
rounding.

### What the memory column is, and is not

TorchTitan's `memory:` log field is `max_reserved_gib`, which comes from
`memory_stats()["reserved_bytes.all.peak"]` — the **PyTorch caching allocator's peak reserved
bytes** — and its percentage is taken against `get_device_properties().total_memory`
(32 GiB here). On TPU, buffers owned by the XLA runtime need not pass through that allocator,
so **this figure is not established to be total device HBM**, and 0.98 GiB is too small to hold
this model's parameters and optimizer state. Read the column as the allocator metric it is; how
it maps to real HBM occupancy is unverified here.

## Environment

| Item | Value |
|---|---|
| Hardware | TPU v6e-4 (Spot) |
| Python | 3.14.10 |
| torch | 2.15.0+cpu |
| torch_tpu | 0.2.0 |
| libtpu | 0.0.50 |
| torchtitan | `google-pytorch/torchtitan` @ `89ef29b` |
| MFU basis | `Peak FLOPS used for computing MFU: 9.180e+14` |

TPU VM images ship Python 3.10; the recipes need 3.14, fetched with `uv`.

```bash
uv python install 3.14
"$(uv python find 3.14)" -m venv ~/venv-titan && source ~/venv-titan/bin/activate
pip install torch==2.15.0 torch_tpu==0.2.0 \
  -r torchtitan/experiments/tpu/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/test/cpu
pip install -r requirements.txt
pip install transformers      # absent from the TPU requirements
PYTHONPATH=$HOME/torchtitan ./run_ladder.sh
```

## Why `bench_cfg.py` exists

`llama3_1b` sets `enable_weight_tying=True`, which fails on TPU inside `Decoder.init_states()`:

```
TypeError: cannot assign 'torch.tpu.BFloat16Tensor' as parameter 'weight'
```

Untying alone is not enough — the tied config gives `tok_embeddings` `_EMBEDDING_SKIP_INIT`, so
the model starts uninitialised and the loss is non-finite at step 1. `bench_cfg.py` unties and
restores a real embedding initialiser.

## Files

| File | Purpose |
|---|---|
| `bench_cfg.py` | the config variant actually trained |
| `run_ladder.sh` | runs all four configurations |
| `results/ladder-batch1.json`, `ladder-batch4.json` | aggregated medians |
| `results/*.metrics.txt` | per-step metric lines, ANSI stripped |
| `results/pip_versions.txt`, `python_version.txt`, `torchtitan_commit.txt` | environment |

License: Apache-2.0
