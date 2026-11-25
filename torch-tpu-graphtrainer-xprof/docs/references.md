# References

| Source | Used for |
|---|---|
| [google-pytorch/torchtitan](https://github.com/google-pytorch/torchtitan), `torchtitan/experiments/tpu` | The TPU trainer, `TpuGraphTrainer`, the `llama3_1b` / `llama3_8b` / `qwen3_8b` recipes, the `optimizer_compile` and `memory_policy` options, and the setup instructions `setup.sh` follows |
| [TorchTPU: performance profiling with XProf](https://torchtpu.readthedocs.io/en/latest/performance_profiling_with_xprof/) | The `torch.tpu.profiler.TpuProfilerConfig` capture path with `torch.profiler` |
| [Cloud TPU: profile your model on TPU VMs](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) | Profile output layout (`plugins/profile/<session>/`) and capture options |
| [openxla/xprof](https://github.com/openxla/xprof) | Reading `.xplane.pb` files; `analyze_xplane.py` uses its `raw_to_tool_data` converter for the overview page and memory profile |
| [TorchTPU x TorchTitan distributed training on TPU v6e-4](https://implicit-none.com/en/torch-tpu-distributed-graphtrainer/) | The earlier article whose 2.06x result this experiment examines |
| [`../torch-tpu-graphtrainer-ladder/`](../../torch-tpu-graphtrainer-ladder/) | The earlier experiment: published numbers for `89ef29b` and its version record (compared in `findings.md` section 2) |
