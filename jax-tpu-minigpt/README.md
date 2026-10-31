# jax-tpu-minigpt

Train a miniGPT language model with JAX and Flax NNX on the **single TPU chip** that Colab
provides. Adapted from the official
[JAX miniGPT tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html).

**Open in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsho/tsho-lab/blob/main/jax-tpu-minigpt/minigpt_tpu.ipynb)

Pick a TPU runtime: **Runtime > Change runtime type > TPU**.

## What changed from the original, and why

The original builds its headline feature on a `(4, 2)` device mesh — 4-way data parallel by
2-way tensor parallel — which needs **8 devices** and targets Kaggle TPU v5e-8. Colab offers
**v5e-1 and v6e-1, one chip**, and the original itself notes that free-tier Colab "can no longer
support SPMD" while its prose still tells readers to expect eight devices.

So the sharding is removed rather than left behind a `device_count() == 1` branch that
demonstrates nothing, and the notebook spends that space on what one chip can actually show:
compilation versus steady-state cost, and how batch size moves throughput. Multi-device
parallelism belongs in its own notebook on 8-chip hardware.

## Two things that bite before any of this runs

**Colab's preinstalled pair cannot import `flax.nnx`.** The runtime ships `jax 0.11.2` with
`flax 0.11.2`; that Flax reaches for `jax._src.core.mutable_array`, which this JAX no longer
has, so `from flax import nnx` raises `ImportError` out of the box. Upgrading Flax is required,
not optional.

**`pip install` alone does not take effect.** Python keeps the already-imported modules, so the
new JAX sits on disk unused until the runtime restarts. Skipping the restart produces the same
`mutable_array` error, which makes it look like a version incompatibility rather than a stale
import. The notebook asserts the versions right after the restart so this cannot pass silently.

Run `env_check.py` to verify a runtime independently of the notebook.

## Pinned versions

| Package | Pinned | Colab default |
|---|---|---|
| `jax[tpu]` | 0.12.0.1 | 0.11.2 |
| `flax` | 0.12.12 | 0.11.2 (cannot import `nnx`) |
| `optax` | 0.2.8 | 0.2.8 |
| `grain` | 0.2.18 | 0.2.18 |
| `tiktoken` | 0.14.0 | not installed |

`libtpu` stays at **0.0.50**, which is what both `jax 0.12.0.1` and the Colab image already use,
so the TPU runtime is not replaced by the upgrade.

## Measurements (Colab TPU v5e-1, 2026-11-01)

miniGPT: `maxlen=256`, `embed_dim=256`, `num_heads=8`, `ff_dim=256`, 8 blocks, GPT-2 vocab.

| batch | first step | steady | tokens/s |
|---:|---:|---:|---:|
| 16 | 33.99 s | 23.2 ms | 176,824 |
| 32 | 44.02 s | 34.7 ms | 235,907 |
| 64 | 44.07 s | 65.1 ms | 251,626 |
| **144** | 46.70 s | 128.7 ms | **286,381** |

Nine times the batch buys 1.6x the throughput: at small batch the chip waits between dispatched
programs rather than computing. No out-of-memory at 144, so `batch_size = 144` — the value the
original picked for eight devices — stays a reasonable default on one chip.

The first step costs 34 to 47 seconds regardless of batch, which is tracing plus XLA
compilation, not work. Timing it without `jax.block_until_ready` measures how fast Python queued
the dispatch instead.

## Note on `nnx.List`

Assigning a plain Python list of blocks makes Flax treat it as a *static* attribute:

```
ValueError: Found data on value of type '<class 'list'>' assigned to static attribute 'blocks'
```

Use `nnx.List([...])`.

## Files

| File | Purpose |
|---|---|
| `minigpt_tpu.ipynb` | the notebook |
| `build_notebook.py` | generates the notebook, so diffs stay readable |
| `env_check.py` | verifies the pinned stack on a runtime |

License: Apache-2.0
