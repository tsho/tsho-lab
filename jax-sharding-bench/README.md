# jax-sharding-bench

Write-up:
- [JAX on TPU v6e: DP vs FSDP-like Benchmark at 0.8B and 3.2B](https://implicit-none.com/en/jax-tpu-v6e-dp-fsdp-benchmark/) (EN) 
- [TPU v6e×4でDP vs FSDP-likeを実測：0.8Bと3.2Bで勝敗が逆転](https://implicit-none.com/ja/jax-tpu-v6e-dp-fsdp-benchmark/) (JA)

The **ICI chapter** of the interconnect series. Companion to
[`distributed-training-bench`](../distributed-training-bench/) (GPU: DeepSpeed
ZeRO vs FSDP2 on NVLink/PCIe) — this measures the same tradeoff on TPU with JAX.

## The question

Sharding trades memory for communication. On GPUs we measured what that costs:
FSDP2 with resharding spends ~51% of GPU time inside NCCL on PCIe and ~28% on NVLink.
TPU's **ICI** (inter-chip interconnect) is the third data point: what does sharding
cost, and what does it buy, when the interconnect is built for collectives?

| mode | params | ~GPU equivalent |
|---|---|---|
| `dp` | replicated on every chip | ZeRO-0 / DDP |
| `fsdp` | sharded on axis 0 across the `data` mesh axis (GSPMD all-gathers per use) | ZeRO-3 / FSDP2 (reshard) |

## Results (single-host TPU v6e-4)

bf16 compute, float32 parameters, `seq_len` 2048, 1 sequence per chip (global batch 4),
15 steps after 3 warm-up steps. Raw output is in [`results/`](results/).

| jax | model | mode | step (ms, median) | tokens/s (global) | peak GB per chip |
|---|---|---|---|---|---|
| 0.6.2 | 0.87B | dp | 61.5 | 133,141 | 3.62 |
| 0.6.2 | 0.87B | fsdp | 55.0 | 148,980 | 4.24 |
| 0.6.2 | 3.27B | dp | 198.2 | 41,346 | 13.03 |
| 0.6.2 | 3.27B | fsdp | 266.2 | 30,778 | 18.88 |
| 0.11.1 | 0.87B | dp | 61.7 | 132,841 | 3.69 |
| 0.11.1 | 0.87B | fsdp | 62.8 | 130,484 | 4.24 |
| 0.11.1 | 3.27B | dp | 196.9 | 41,605 | 13.17 |
| 0.11.1 | 3.27B | fsdp | 299.5 | 27,348 | 17.63 |

The 3.27B rows on jax 0.6.2 are the mean of two runs (they differ by about 0.03%).
Every other row is a single run.

What the numbers show:

- **`fsdp` did not save memory here.** Peak memory per chip was higher with `fsdp`
  than with `dp` at both sizes (18.9 GB vs 13.0 GB at 3.27B). We have not profiled why.
  Note the setup: the update is plain SGD, so there is no optimizer state. The memory
  that ZeRO/FSDP saves in the GPU benchmark (Adam moments) does not exist in this one.
- **The throughput cost of sharding depends on model size.** At 3.27B, `fsdp` is 26%
  slower than `dp` on jax 0.6.2 and 34% slower on jax 0.11.1. At 0.87B, `fsdp` is 12%
  *faster* on jax 0.6.2 and 2% slower on jax 0.11.1.
- **The JAX version moved the `fsdp` numbers, not the `dp` ones.** Going from jax 0.6.2
  to 0.11.1, `dp` throughput stayed within 1%, while `fsdp` dropped 11–12% at both sizes.
  The Python version changed as well (3.10 → 3.12), so this is a change of the whole stack.

What they do not show:

- No XLA trace was analyzed, so the share of step time spent in collectives on ICI is
  **not measured yet**. The comparison with the GPU NCCL fractions above is still open.
- One host, four chips, random initialization, synthetic tokens, n=1 for most rows.
  Treat differences of a few percent with care.

## Run

See [`RUNBOOK.md`](RUNBOOK.md) for the full procedure (create the TPU VM, run, collect
results, delete the VM). On a TPU VM:

```bash
pip install -U "jax[tpu]"

# 0.87B (defaults: --dim 2048 --layers 16)
python3 bench.py --mode dp   --tag 0p8b
python3 bench.py --mode fsdp --tag 0p8b
# 3.27B
python3 bench.py --mode dp   --dim 3072 --layers 28 --heads 24 --tag 3p2b
python3 bench.py --mode fsdp --dim 3072 --layers 28 --heads 24 --tag 3p2b
# optional: XLA trace for collective-time analysis
python3 bench.py --mode fsdp --profile-dir /tmp/trace
```

Results append to `results/*.jsonl`: median step time, tokens/sec, and peak memory
per device (`peak_bytes_in_use` of device 0). Use `--tag` to record the model size and
software stack of a run; see [`results/README.md`](results/README.md).

Pure JAX (no flax) · random init · synthetic tokens — systems behavior only;
losses are meaningless by design.
