# Troubleshooting

Notes from the November 2026 runs (torch 2.15.0, torch_tpu 0.2.0, libtpu
0.0.50, Python 3.14.10, TorchTitan `e78271f7` unless stated). Each Note gives
the symptom, the cause or observation, and what worked. Causes are marked
**confirmed** (seen in this run's logs or code), **from source** (read in the
TorchTitan source, not triggered here) or **unverified**.

---

## Configuration

### Note 1: per-block compile on the plain TPUTrainer is rejected with the default backend

**Symptom.** Configurations B and F exited within about 20 s:

```
RuntimeError: torch.compile(..., backend='inductor') on TPU is experimental and does
not yet have full TPU support; compilation is likely to fail with cryptic errors, ...
Pass backend='tpu' or do not specify `backend` in torch.com[...]
```

(The retained log line is truncated at that point.)

**Cause (confirmed).** Setting `compile.enable = True` on the plain
`TPUTrainer` config leaves `compile.backend` at its default, `"inductor"`,
which torch_tpu refuses. The `TpuGraphTrainer` compile config defaults to
`"tpu"`.

**What worked.** `config.compile.backend = "tpu"` (`cfg_b_perblock` and
`cfg_f_perblock_separate` in `scripts/bench_cfg.py`).

---

### Note 2: `optimizer_compile="in_graph"` constraints

**Observation (from source).** In the graph-trainer config, `in_graph`
requires `compile.mode == "aot_fx_trace"` and cannot be combined with
`compile_grad_norm=True`; it also rejects gradient accumulation above 1.
These were read in the TorchTitan source before the runs and respected in
`bench_cfg.py` (every graph-trainer config sets `mode="aot_fx_trace"`); the
corresponding errors were not triggered here.

**What worked.** To measure the effect of compiling the gradient norm, use
`compile_grad_norm=True` on the `eager` and `separate_graph` configurations
instead (C2 and D2).

---

### Note 3: `in_graph` logs `grad_norm: 0.0000`

**Symptom.** With `optimizer_compile="in_graph"`, the logged `grad_norm` is
0.0000 on every step (configuration E and every `in_graph` run in
`results/*.json`, field `grad_norm_all_zero`). Loss still decreases normally.

**Observation (confirmed).** In the installed build,
`TpuGraphTrainer._train_step_optimizer_in_graph` does not call the
`_clip_grad_norm` helper; the `eager` and `separate_graph` step paths do
(`scripts/discover.py`, section 8, prints this).

**Not verified.** Whether gradients are clipped inside the traced step graph
in `in_graph` mode. A reading of the source before the runs suggested that the
traced function clips internally; this was not checked at runtime. Either way,
E does not take the same clipping path as C and D, and the logged 0.0000 is
not the gradient norm.

**What worked.** Do not use the logged `grad_norm` to judge an `in_graph` run.

---

### Note 4: graph-trainer configs set `xla_enable_hlo_sharding_v3=False`

**What the code does.** For every graph-trainer configuration,
`bench_cfg.py` sets the config's `xla_flags` to
`tpu_flags.merge_flags(xla_enable_hlo_sharding_v3=False)` from
`torchtitan.experiments.tpu`; if that import or call raises, it falls back to
the plain dict `{"xla_enable_hlo_sharding_v3": False}`. Which of the two
branches ran was not logged. The plain-TPUTrainer configurations (A, B, F) do
not set it.

**Why (unverified here).** This mirrors the shipped `qwen3_8b` graph-trainer
recipe, whose comment says Shardy HLO sharding v3 is on by default from libtpu
0.0.48 and segfaults in `mlir::sdy::buildTensorFactorShardings` under the
graph trainer. No graph-trainer run was made without the flag, so the
segfault was not reproduced on libtpu 0.0.50.

---

### Note 5: `memory_policy="tpu_selective"` needs `save_regions`

**Symptom (confirmed).** The run exits at startup:

```
ValueError: memory_policy='tpu_selective' needs a region in save_regions or
save_last_n_layers; known regions: attention.inner_attention, feed_forward.w2,
moe.ep_dispatch, moe.routing.
```

**What worked.** `save_regions=("feed_forward.w2",)` (`cfg_mp_tpu_selective`).
Only this region was measured. `attention.inner_attention` was not measured;
no retained log shows what happens with it on `llama3_1b`.

---

### Note 6: the `llama3_8b` recipe is not an eager baseline

**Symptom.** The 8B "eager" run (`cfg_8b_a_eager`) and the per-block run gave
the same throughput (2,536 and 2,537 tok/s/chip) and the same final loss.

**Cause (confirmed).** The config dump in the 8B eager run's log shows
`compile.enable: true` with `backend: "tpu"`: the `llama3_8b` recipe already
enables per-block compile. The 1B recipe does not.

**What worked.** Nothing was re-run; there is no 8B eager baseline. To get
one, set `config.compile.enable = False` explicitly.

---

## Memory

### Note 7: 8B graph trainer does not fit on v6e-8 at 2048 tokens per device

**Symptoms (confirmed).**

- `in_graph` with `memory_policy="default"` or `"full"`: compile-time failure,
  `Ran out of memory on HBM, the total memory required for HLO temporaries
  (35.09G) exceeds available HBM (31.24G)`.
- Eager optimizer with `default` and with `full_and_offload`: runtime failure,
  `RESOURCE_EXHAUSTED: failed to enqueue execution ... RuntimeProgramAllocationFailure`
  (the `default` run completed eight steps before failing).

**What worked.** Only `in_graph` + `full_and_offload` ran: 2,444 tok/s/chip,
`tpu-info` peak 31.22 GiB of 31.24 GiB. Smaller batches and other policies
were not tried.

---

### Note 8: the memory figure in the TorchTitan log is not HBM in use

**Observation (confirmed).** The `memory:` value in TorchTitan's step log
equals the "stack reserved" (program scratch) figure of the XProf memory
profile in every profiled configuration (0.98 GiB for per-block, 13.40-13.42
GiB for the graph trainer, 2.98 GiB for 8B per-block). It is not the HBM in
use: for per-block compile at 1B, XProf reports a 15.01 GiB peak.

**What worked.** Sample `tpu-info` during the run (`run_memory_batch_oldcommit.sh`
and `run_8b.sh` do this every two seconds), or read the XProf memory profile
(`analyze_xplane.py`).

---

## Profiling

### Note 9: the profiler module is `torch.tpu.profiler`

**Observation (confirmed).** `from torch.tpu.profiler import TpuProfilerConfig`
works; `import torch_tpu.profiler` fails with
`ModuleNotFoundError: No module named 'torch_tpu.profiler'`
(`scripts/discover.py`, section 7). In the end no direct profiler code was
needed: TorchTitan's built-in `--profiler.*` flags produce both outputs.

**Output layout.**

```
<dump_folder>/profiling/traces/iteration_20/rank{N}_trace.json.gz
<dump_folder>/profiling/traces/rank_{N}/plugins/profile/<timestamp>/<hostname>.xplane.pb
```

The VM host name is part of the `.xplane.pb` file name. Remove or rename the
files before sharing them.

---

### Note 10: logged step time is rounded to 10 ms

**Observation (confirmed).** `time_end_to_end` is logged with 0.01 s
resolution, so every step time in the logs is a multiple of 10 ms (for
example 170 ms for E and 180 ms for C, which differ by 1.6% in tokens/s).

**What worked.** Derive step time from throughput:
`tokens per step per chip / (tok/s/chip)`, e.g. 2048 / 11,772 = 174.0 ms.

---

### Note 11: MXU utilisation reads 0.0

**Observation (confirmed).** The XProf overview page reports MXU utilisation
0.0% for every profile file. Hardware performance counters were evidently not
captured with these profiler settings; MXU utilisation was treated as not
obtainable.

---

## Provisioning and setup

### Note 12: capacity, not quota, limits v6e

**Observation (unverified in retained logs).** Create requests were refused
with "There is no more capacity in the zone"; a quota increase does not help.
`scripts/provision.sh` tries a list of zones in turn. The 8B runs used Spot
capacity on v6e-8. No provisioning log was kept for this experiment; the
behaviour is the same as recorded in
[`../vllm-torchtpu-zero-to-serving/docs/troubleshooting.md`](../../vllm-torchtpu-zero-to-serving/docs/troubleshooting.md).

### Note 13: `READY` with the `health` field absent

**Observation (unverified in retained logs).** On a usable node,
`gcloud compute tpus tpu-vm describe --format="value(state,health)"` can
return `READY` with an empty health value. Waiting for the literal string
`HEALTHY` never finishes. `provision.sh` accepts `READY` with health absent or
`HEALTHY`.

### Note 14: setup details

- **No apt lock wait.** `setup.sh` installs nothing through apt (only `uv` and
  pip), so it does not wait for the boot-time dpkg lock. An earlier version
  that did wait gave up after 600 s on one v6e-8 VM whose lock never cleared
  (recorded in the script comments; no log retained).
- **pip reports a dependency conflict; it is not a failure.** In this run
  pip's resolver printed `gcsfs 2026.10.1 requires fsspec>=2026.9.0, but you
  have fsspec 2026.2.0` and setup continued normally. The `tokamax` / `tyro`
  conflict over `typeguard` that `setup.sh` mentions was met in earlier setups
  and does not appear in this run's setup log.
- **`transformers` is installed separately.** The TorchTitan TPU
  requirements did not include it in an earlier setup, so `setup.sh` installs
  it after the requirements files. Not re-checked against the requirements
  file at `e78271f7`.
- **Python.** The TPU VM image ships Python 3.10; the recipe and the
  `torch_tpu` wheels need a newer one. `uv python install 3.14` was used.
- **Index URL.** `setup.sh` installs torch with
  `--extra-index-url https://download.pytorch.org/whl/cpu`. Without an extra
  index, pip resolves a CUDA build of torch (seen in earlier setups, not
  re-tested here).
