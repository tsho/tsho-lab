# vLLM TorchTPU: zero to serving on a single TPU v6e chip

Serving `Qwen/Qwen3-8B` with `vllm-torchtpu==0.33.0` on one Cloud TPU v6e chip
(`v6e-1`), from an empty project to an answering OpenAI-compatible endpoint.

Measured 2026-11-24. Everything here is an observation from that run; where
something is inferred rather than measured, it says so.

## Why this configuration

The published [v6e recipes][v6e] list `Qwen/Qwen3-8B` as a recommended model
for a single v6e chip. But that checkpoint appears in neither the
[Validated Models][models] table — CI covers TPU v7x only — nor the
[v6e nightly support matrix][matrix], whose single row is a different model
marked untested in every column. The recipes page describes its own v6e
commands as "reference configuration examples", unlike the v7x recipes which
mirror CI benchmark scripts.

So this is a recommended configuration with no published verification behind
it. The question here is not whether it is fast. It is whether it works as
written, and where the documentation and the software disagree.

**It works.** The recipe ran unmodified: no flag changes, no Hugging Face
token, no patches.

[v6e]: https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/deployment/v6e_recipes/
[models]: https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/user_guide/supported_models/
[matrix]: https://github.com/vllm-project/vllm-torchtpu/blob/main/support_matrices/nightly/v6e/model_support_matrix.csv

## Results

### Time to a serving endpoint

| Phase | Time |
|---|---:|
| Fresh VM to a verified TorchTPU backend | **187 s** |
| — of which waiting for the boot-time dpkg lock | 60 s |
| — of which `uv pip install vllm-torchtpu` (210 packages) | 46 s |
| First serve: download + compile | **374 s** |
| Serve with the model cached, compile cache cleared | **186 s** |
| Serve on a warm restart | **72 s** |

Weight loading was 5.4–6.2 s in all seven runs, so the cold figure is almost
entirely compilation, not I/O.

Condition B was run three times (186.0, 186.0, 188.0 s). Condition C was run
three times (220.1, 72.0, 72.0 s — see below). Condition A was run once: each
repeat re-downloads 16 GB, and that cost measures network speed rather than
anything about vllm-torchtpu.

### The compilation cache does not behave as documented

The documentation names three environment variables for persisting compiled
artifacts and does not say how they relate. Pointing all three at one root and
measuring each directory before and after every run gives a clear answer:

| Variable | Result |
|---|---|
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | **the one that works** — 151 MB, read not rewritten on warm restarts |
| `VLLM_CACHE_ROOT` | written every run, never reused: 463 MB → 1.2 GB across the series |
| `VLLM_XLA_CACHE_PATH` | **0 files, 0 bytes — never written at all** |

`VLLM_XLA_CACHE_PATH` is the variable the [quickstart][qs] and the
[environment-variable reference][env] tell you to set to persist compiled
artifacts across restarts. Nothing wrote to it, across seven runs on one node
and a full session on a second, independently provisioned node.

**Why the first restart is not fast.** Condition C's first run took 220 s and
only later runs hit 72 s. The server explains it directly:

```
WARNING compile_prefix isolation: Qwen3Model has no self.prefix;
        using non-deterministic fallback _tpu_instance_0
        (its on-disk compile cache will not persist across restarts)
```

For the very model the v6e-1 recipe recommends, the vLLM-layer compile cache
key falls back to a non-deterministic value, and the build states that such a
cache will not survive a restart. `VLLM_CACHE_ROOT` growing on every run and
never being reused matches that exactly. What persists is the native PJRT
tier3 cache, which is keyed differently — hence fast restarts, but only after
one warm run has populated it.

**Inference, not measurement:** that tier3 supplies the speedup while the
vLLM-layer cache does not is consistent with every number above, but it was not
proven by disabling tier3 alone and re-measuring. A clean single-variable test
is the obvious follow-up.

[qs]: https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/getting_started/quickstart/
[env]: https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/reference/env_vars/

### Serving facts

| Item | Value |
|---|---|
| `--kv-cache-dtype fp8` resolves to | `fp8_e5m2`, selected automatically for v6e |
| KV cache size | 185,088 tokens |
| Compiled shape buckets | `[16, 32, 64, 128, 256, 512, 1024, 2048]` |
| Weight dtype | `torch.bfloat16` |
| Model on disk | 16 GB, five safetensors shards |
| Device HBM with the server idle | **28.30 GiB / 31.24 GiB** (`tpu-info`) |
| Compile backend | `vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor` |

Note that `--gpu-memory-utilization 0.90` applies to **usable** HBM, not the
nominal 32 GB: 0.90 × 31.24 GiB = 28.1 GiB, matching the 28.30 GiB observed.
Sizing against "32 GB" overstates the budget by roughly 0.75 GiB. The HBM
figure above is a device-level reading from `tpu-info`, not a framework
allocator statistic.

### API

All four endpoints exist and answer; none had to be recorded as unsupported.

| Endpoint | Status |
|---|---|
| `GET /health` | 200 |
| `GET /v1/models` | 200 |
| `POST /v1/completions` | 200 |
| `POST /v1/chat/completions` | 200 |

Through the OpenAI Python SDK: non-streaming prose, structured-list and code
prompts all succeeded, and streaming returned **TTFT 0.057 s** over 256 chunks.

**Note on `finish_reason`.** Every response stopped on `length`, never `stop`.
That is the model, not the server: Qwen3-8B is a reasoning model and opens with
a `<think>` block, so at `max_tokens: 256` the whole budget goes to the
thinking preamble and no final answer is returned. Budget far more output
tokens, or disable thinking, before concluding something is broken.

### Single-request smoke benchmark

A functional check, not a performance study. One request per shape.

| Shape | In / out | TTFT | TPOT | Output tok/s |
|---|---|---:|---:|---:|
| short | 136 / 128 | *(invalid, see below)* | 14.13 ms | 24.8 |
| prefill-heavy | 4104 / 128 | 206 ms | 14.36 ms | 63.0 |
| decode-heavy | 136 / 1024 | 23 ms | 14.18 ms | 70.5 |

**The `short` TTFT of 3362 ms in the raw data must not be quoted.**
`vllm bench serve` runs no warmup, and `short` executed first, so that request
paid first-touch compilation for its shape bucket. The other two shapes show
what TTFT actually looks like. `benchmark_smoke.py` here does run a warmup and
reports it separately; it was unused because the official CLI was available.

What is trustworthy is TPOT: 14.13, 14.36 and 14.18 ms, flat to within 1.6%
while input length varies 30× and output length 8×.

No conclusion is drawn about throughput limits, saturation or bottlenecks. One
request per shape cannot support one.

## Versions

The installation docs say `pyproject.toml` pins compatible versions
"automatically" without giving numbers. Resolved from PyPI on 2026-11-24:

| Package | Version |
|---|---|
| Python | 3.12.16 |
| `vllm-torchtpu`, `vllm-torchtpu-core` | 0.33.0 |
| `torch` | 2.13.0+cpu |
| `torch-tpu` | 0.2.0 |
| `libtpu` | 0.0.50 |
| `jax`, `jaxlib` | 0.11.2 |
| `transformers` | 5.20.2 |

Two things worth noting. `torch` is the **CPU** build, which is correct: the
accelerator is reached through `torch_tpu` and `libtpu`, and this is why
`UV_TORCH_BACKEND=cpu` matters. And `jax` is installed even though this is the
PyTorch backend, consistent with the TPU kernels being JAX/Pallas custom ops
called from inside the compiled graph — its presence does not mean the JAX
serving backend is in use.

There is also **no separate `vllm` distribution**. `pip freeze` contains only
`vllm-torchtpu` and `vllm-torchtpu-core`; `import vllm` still reports 0.33.0
and the CLI works normally. Pinning `vllm==` alongside this will surprise you.

## Reproducing

Requires a GCP project with TPU v6e quota, `gcloud` authenticated, and roughly
40 minutes.

```bash
# 1. Capture a chip. Tries several zones; v6e capacity, not quota, is
#    usually what blocks you.
PROJECT=your-project ./provision.sh          # add SPOT=0 for on-demand

# 2. Copy the scripts up (zone from provision.sh output)
gcloud compute tpus tpu-vm scp *.sh *.py requirements.txt \
  vllm-torchtpu-serve:~/ --project=your-project --zone=ZONE

# 3. Install, serve, verify the API and run the smoke benchmark
gcloud compute tpus tpu-vm ssh vllm-torchtpu-serve \
  --project=your-project --zone=ZONE \
  --command='cd ~ && nohup setsid ./runall.sh </dev/null >/dev/null 2>&1 &'

# 4. Startup conditions, separately
#    A fresh  B model cached  C warm restart
./measure_startup.py --condition B --repeats 3 --out results/startup_B.json

# 5. Delete it. On-demand v6e-1 bills by the hour.
gcloud compute tpus tpu-vm delete vllm-torchtpu-serve \
  --project=your-project --zone=ZONE --quiet
```

`serve.sh` binds to `127.0.0.1`. To reach it from a workstation, forward the
port rather than exposing the endpoint:

```bash
gcloud compute tpus tpu-vm ssh vllm-torchtpu-serve \
  --project=your-project --zone=ZONE -- -N -L 8000:localhost:8000
```

## Notes for anyone attempting this

Four things cost time here and none are in the documentation. Details and
exact error text in [`docs/troubleshooting.md`](docs/troubleshooting.md).

1. **Capacity, not quota, is the binding constraint.** Four of five zones
   refused with "no more capacity". A quota increase would not have helped.
   Refusals took anywhere from 1.5 to 14 minutes each.
2. **`state: READY` does not mean a node is usable**, and the `health` field is
   **absent** on a healthy node — it appears only when something is wrong. So
   waiting for `health == "HEALTHY"` loops forever, and checking `state` alone
   accepts a node that refuses SSH.
3. **A fresh TPU VM holds the dpkg lock** for its first minutes, so installing
   `libibverbs1` immediately fails. `setup.sh` waits for the lock.
4. **The VM image ships Python 3.10**, but `vllm-torchtpu` requires
   `>=3.12,<3.13`. The official install page begins with `uv venv --python
   3.12`, which presupposes an interpreter that is not there.

On Spot: two of three nodes were preempted, one after about 15 minutes, which
voided a measurement series each time. The final run used on-demand. Spot is
fine for a single serve, but a multi-run measurement wants a stable node.

## Scope

Covered: install, serving, API verification, startup under three cache
conditions, device memory, and a single-request smoke benchmark.

Not covered: concurrency sweeps, serving-parameter tuning, any comparison
against the JAX backend or against GPUs, tensor-parallel scaling, and XProf
profiling.

## Layout

| Path | Contents |
|---|---|
| `provision.sh` | capture a v6e-1 and wait until it is genuinely usable |
| `setup.sh` | install, record versions, verify the backend is TorchTPU |
| `serve.sh` | the published v6e-1 recipe, with all three cache roots set |
| `runall.sh` | install → serve → API checks → benchmark, detached |
| `measure_startup.py` | startup under cache conditions A, B and C |
| `smoke_test.py` | the four endpoints plus OpenAI SDK cases |
| `benchmark_smoke.sh`, `benchmark_smoke.py` | the three-shape smoke benchmark |
| `sanitize_logs.py` | strip host detail from logs before publishing |
| `results/` | raw JSON, versions, timings, sanitized logs |
| `docs/experiment_notes.md` | full observations, including what went wrong |
| `docs/troubleshooting.md` | each problem, its error text and its fix |
| `docs/references.md` | every source URL, and the conflicts between them |
