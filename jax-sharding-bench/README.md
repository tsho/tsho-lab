# jax-sharding-bench

The **ICI chapter** of the interconnect series. Companion to
[`distributed-training-bench`](../distributed-training-bench/) (GPU: DeepSpeed
ZeRO vs FSDP2 on NVLink/PCIe) — this measures the same tradeoff on TPU with JAX.

## The question

Sharding trades memory for communication. We measured what that costs on
PCIe (NCCL ~51% of GPU time) and NVLink (~28%). TPU's **ICI** (inter-chip
interconnect, torus topology) is the third data point: how much of the
sharding tax disappears when the interconnect is designed for collectives?

| mode | params | ~GPU equivalent |
|---|---|---|
| `dp` | replicated | ZeRO-0 / DDP |
| `fsdp` | sharded on axis 0 (GSPMD all-gathers per use) | ZeRO-3 / FSDP2(reshard) |

## Run (TPU VM, e.g. v5e-8 / v6e-8)

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 bench.py --mode dp   --steps 15
python3 bench.py --mode fsdp --steps 15
# bigger model (FSDP should win on memory):
python3 bench.py --mode dp   --dim 3072 --layers 24
python3 bench.py --mode fsdp --dim 3072 --layers 24
# optional: XLA trace for collective-time analysis
python3 bench.py --mode fsdp --profile-dir /tmp/trace
```

Results append to `results/*.jsonl` (same spirit as the GPU bench: median step
time, tokens/sec, peak memory per device).

Pure JAX (no flax) · random init · synthetic tokens — systems behavior only;
losses are meaningless by design.
