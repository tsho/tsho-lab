# torch-tpu-execution-modes

Write-up:
- [Benchmarking TorchTPU 0.2.0: SingleTraceTrainer Is 3.4x Faster than Eager — Why Isn't torch.compile Faster?](https://implicit-none.com/en/torch-tpu-execution-modes/) (EN)
- [JA version](https://implicit-none.com/ja/torch-tpu-execution-modes/)

Single-chip (**TPU v6e-1**) execution-method benchmark for [`google-pytorch/torch_tpu`](https://github.com/google-pytorch/torch_tpu) (`torch-tpu==0.2.0`). Uses the exact same **0.871B Pre-LN Transformer LM** architecture (`vocab=32000`, `dim=2048`, `layers=16`, `heads=16`, `ff_mult=4`, tied embedding head, `float32` master weights + `bfloat16` compute, `seq_len=2048`, `batch_size=1`) as [`jax-sharding-bench`](../jax-sharding-bench/).

## Results (TPU v6e-1, `torch==2.14.1+cpu`, `torch_tpu==0.2.0`, `libtpu==0.0.51`)

Each configuration runs in its own isolated Python process (because TorchTPU locks environment variables at `import torch`), synchronizing every step via `torch.accelerator.synchronize()`. 5 warmup steps + 30 measured steps. Raw outputs and logs are in [`results/`](results/).

| Group | Mode (`--mode`) | Configuration | Setup + 1st Step (s) | Steady Median (ms) | p10 – p90 (ms) | tok/s | vs Strict Eager | XLA Compilations (`cache_stats`) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **A** | `eager` | Default (`EagerMode.DEFER_NEVER`) | 0.78 s | **164.05 ms** | 163.26 – 164.96 | **12,483.7** | 1.00x | 87 (`reqs=56,702`) |
| **B (env)** | `defer_fuse` | `TORCH_TPU_DEFER_AND_FUSE=1` | 17.63 s | **93.70 ms** | 93.49 – 93.99 | **21,857.7** | **1.75x** | 41 (`reqs=15,856`) |
| **B (API)** | `defer_fuse_ctx` | `torch.tpu.set_eager_mode(EagerMode.DEFER_AND_FUSE)` | 17.59 s | **93.70 ms** | 93.35 – 94.43 | **21,858.1** | **1.75x** | 42 (`reqs=15,857`) |
| **C** | `compile` | `torch.compile(model)` (fwd+loss) + eager bwd/SGD | 22.69 s | **94.75 ms** | 94.40 – 95.26 | **21,614.9** | **1.73x** | 6 (`reqs=74`) |
| **C (step)** | `compile_step` | `@torch.compile` wrapping fwd + `autograd.grad` + in-place SGD | 26.54 s | **121.38 ms** | 121.00 – 121.74 | **16,872.8** | 1.35x | 6 (`reqs=40`) |
| **D (public)** | `single_trace` | `SingleTraceTrainer` + `Sgd(fused=False)` + `torch.compile` | 19.57 s | **48.56 ms** | 48.29 – 48.89 | **42,176.0** | **3.38x** | 3 (`reqs=3`) |
| **D (static)** | `single_trace_static` | `SingleTraceTrainer` + `Sgd(fused=True)` + `StaticCompiler` (donation) | 11.62 s | **48.15 ms** | 47.78 – 48.53 | **42,533.2** | **3.41x** | 3 (`reqs=3`) |

## Files

- [`bench_modes.py`](bench_modes.py) — Self-contained benchmark script supporting all 7 execution configurations, Tier-2 cache tests, and dynamic shape sweeps.
- [`run_all.sh`](run_all.sh) — End-to-end runner (`bash run_all.sh --on-vm` on a TPU VM, or `bash run_all.sh --provision --project <GCP_PROJECT> --zone europe-west4-a`).
- [`summarize.py`](summarize.py) — Regenerates the Markdown comparison table from `results/results_*.json`.
- [`results/`](results/) — Raw JSON outputs (`versions.json`, `results_*.json`) and pip dependency resolution logs (`pip_no_extra_index_dryrun.log`, `pip_install_torchtpu.log`).

## Run

On a TPU `v6e-1` VM (`v2-alpha-tpuv6e`, Python 3.11+):

```bash
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Run individual modes (1 process per mode):
python3 bench_modes.py --mode eager               --output results/results_eager.json --dump-versions results/versions.json
python3 bench_modes.py --mode defer_fuse          --output results/results_defer_fuse.json
python3 bench_modes.py --mode defer_fuse_ctx      --output results/results_defer_fuse_ctx.json
python3 bench_modes.py --mode compile             --output results/results_compile.json
python3 bench_modes.py --mode compile_step        --output results/results_compile_step.json
python3 bench_modes.py --mode single_trace        --output results/results_single_trace.json
python3 bench_modes.py --mode single_trace_static --output results/results_single_trace_static.json

# Print summary table from JSON outputs:
python3 summarize.py
```
