# Experiment notes

Observed facts only. Where something is inferred rather than measured, it says so.

Date: 2026-11-24. Accelerator: Cloud TPU v6e-1 (single chip), Spot.

## Phase 1 — provisioning

### Capacity, not quota, is the binding constraint

A v6e-1 Spot VM was requested in four zones in sequence. Three refused for
capacity; the fourth succeeded. Total elapsed time from first attempt to a
`READY` node was about 9 minutes.

| # | Zone | Result | Time |
|---|---|---|---|
| 1 | `asia-northeast1-b` | no capacity | 19:14 |
| 2 | `us-east5-b` | no capacity | 19:18 |
| 3 | `us-east5-a` | no capacity | 19:21 |
| 4 | `us-central1-b` | **created** | 19:23 |

The refusal is explicit about which constraint was hit:

> There is no more capacity in the zone "asia-northeast1-b"; you can try in
> another zone where Cloud TPU Nodes are offered

This matters for anyone planning a run: the failure is a stock-out, not a
quota denial, so requesting a quota increase would not have helped. Expect to
iterate over zones. Each refusal took 2–3 minutes to come back.

### The provisioning path that worked

The vllm-torchtpu TPU setup page routes TPU v6e through the Compute Engine API
and lists only Standard and Flex-start capacity models, never Spot. The Cloud
TPU API path used here is documented but carries a notice that the API is "no
longer under active development". It is nonetheless the path that worked, and
it is the one that accepts `--spot`:

```bash
gcloud compute tpus tpu-vm create vllm-torchtpu-serve \
  --zone=us-central1-b \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e \
  --spot
```

Reported back by `describe`: `state=READY`, `acceleratorType=v6e-1`,
`runtimeVersion=v2-alpha-tpuv6e`.

### What the fresh VM actually is

| Property | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1015-gcp |
| Architecture | x86_64 |
| vCPUs | 44 |
| RAM | 172 GiB total, 170 GiB available |
| Root disk | 97 GB, 89 GB free on a fresh image |
| **Default Python** | **3.10.12** |
| TPU device nodes | `/dev/vfio/0`, `/dev/vfio/devices`, `/dev/vfio/vfio` |
| `/dev/accel*` | absent |
| `uv` preinstalled | no |

Two of these decide the install:

- **Python 3.10.12 is the image default, and vllm-torchtpu requires
  `>=3.12,<3.13`.** The interpreter has to be installed; it cannot be assumed.
  Neither the TPU setup page nor the installation page says how to do this on
  the TPU VM image — the installation page's commands begin with
  `uv venv --python 3.12`, which presupposes a 3.12 that is not there.
- **TPU chips appear under `/dev/vfio/`, not `/dev/accel*`.** The setup page's
  `ls /dev/vfio/` check matches this image; a habit of looking for
  `/dev/accel0` (the older PyTorch/XLA-era device node) finds nothing and
  invites a wrong conclusion that the chip is missing.

Disk headroom is not a concern: 89 GB free against roughly 16 GB of weights.
The installation docs give no boot-disk guidance, and on this image none is
needed.

## Phase 2 — installation

### The first install attempt failed on a dpkg lock

`libibverbs1` is a documented prerequisite, so the install script runs
`apt-get install` first. On a freshly created TPU VM that fails:

```
E: Could not get lock /var/lib/dpkg/lock-frontend.
   It is held by process 4806 (apt-get)
E: Unable to acquire the dpkg frontend lock
```

The image runs its own apt work at boot, so dpkg is busy for the first minutes
of a node's life. Nothing is wrong with the VM; the install simply started too
early. Under `set -euo pipefail` the whole script aborts, which makes this look
more serious than it is.

The fix is to wait for the lock rather than to retry blindly:

```bash
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
   || sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
  sleep 10
done
```

This is worth building into any automated TPU provisioning script. It is not
mentioned in the installation documentation, which assumes a VM that has
settled.

### Install timings

Second attempt, with the lock wait in place. All figures from
`setup_timings.txt`, which records a unix timestamp before each step.

| Step | Seconds |
|---|---:|
| apt phase (`libibverbs1`) | 123 |
| — of which waiting for the dpkg lock | 60 |
| install `uv` | 1 |
| `uv python install 3.12` | 1 |
| `uv venv --python 3.12` | <1 |
| **`uv pip install vllm-torchtpu`** | **46** |
| record versions | 9 |
| verify backend | 7 |
| **total, fresh VM to verified backend** | **187** |

Three minutes, start to finish, and 46 seconds of that is the actual install of
210 packages. The single largest cost is waiting for the boot-time apt work to
finish. Step resolution is one second, so the sub-second rows should not be
read as precise.

### The version table the documentation does not publish

The installation page says `pyproject.toml` "pins compatible `torch-tpu`,
`jax`, `libtpu`, and `vllm` versions automatically" without giving the
numbers. Resolved on 2026-11-24 from PyPI, they are:

| Package | Version |
|---|---|
| Python | 3.12.16 |
| `vllm-torchtpu` | 0.33.0 |
| `vllm-torchtpu-core` | 0.33.0 |
| `torch` | **2.13.0+cpu** |
| `torchvision` | 0.28.0+cpu |
| `torch-tpu` | 0.2.0 |
| `libtpu` | 0.0.50 |
| `jax` | 0.11.2 |
| `jaxlib` | 0.11.2 |
| `tpu-sync-torch` | 0.0.1.dev20261029081917 |
| `tpu-info` | 0.7.1 |
| `transformers` | 5.20.2 |

Total packages installed: 210.

Two things here are worth more than the rest:

- **`torch` is 2.13.0, and it is the CPU build.** The accelerator is reached
  through `torch_tpu` and `libtpu`, so a CPU torch wheel is correct and not a
  misconfiguration. This is also why `UV_TORCH_BACKEND=cpu` matters: the
  default index would serve a CUDA build of the same version for no benefit.
- **`jax` and `jaxlib` are installed even though this is the PyTorch backend.**
  That is consistent with the documented architecture, in which the TPU kernels
  are JAX/Pallas kernels registered as custom ops and called from inside the
  compiled graph. JAX being present does not mean the JAX serving backend is
  in use.

### There is no separate `vllm` distribution

The launch blog says `pip install vllm-torchtpu` "installs a matching vLLM".
On this install that is not a separate package. `pip freeze` contains no
`vllm` distribution at all — only `vllm-torchtpu` and `vllm-torchtpu-core`,
both 0.33.0. The module still imports under the ordinary name:

```
>>> import vllm; vllm.__version__
'0.33.0'
>>> os.path.dirname(vllm.__file__)
'/home/user/uv_venv/lib/python3.12/site-packages/vllm'
```

So `import vllm` and the `vllm` CLI work as normal, but anyone checking
installed versions by looking for a `vllm` package, or pinning `vllm==` in a
requirements file alongside this, will be surprised.

### Confirming the backend is TorchTPU and not JAX or CPU

Four checks, all of which passed:

| Check | Result |
|---|---|
| `current_platform.device_name` | `tpu` |
| `tpu-info` chip enumeration | 1 × `TPU v6e chip` at `/dev/vfio/0` |
| libtpu version announced at init | 0.0.50 |
| `tpu-inference` / `vllm-jax` present? | no (expected) |

Startup also logs the PJRT handshake, which is the clearest single line that
the TPU path is live rather than a CPU fallback:

```
GetPjrtApi was found for TPU at .../site-packages/libtpu/libtpu.so
Successfully renamed PrivateUse1 backend to 'tpu'. Device: device(type='tpu')
```

`Pin memory is not supported on TPU.` appears on every invocation, including
`vllm --version`. It is informational.

### `vllm serve --help` does not list the flags

Worth knowing before scripting a flag check: `vllm serve --help` prints only a
list of config groups, not the options. Any grep for a flag name against it
finds nothing and looks like the flag is missing. The full list needs
`vllm serve --help=all`, which is 2,462 lines.

Checked that way, all eight flags in the published v6e-1 recipe exist in this
build, and `fp8` is among the accepted `--kv-cache-dtype` values (the build
accepts 22 of them).

## Interruption — Spot maintenance event during the first serve

About fifteen minutes into the session, partway through the condition A run
(the fresh start, which includes the model download), the node became
unreachable. `gcloud` reported:

```
WARNING: !!! This TPU is going through a maintenance event,
         and might be unavailable !!!
```

followed by `ssh: connect to host ... port 22: Connection refused` on every
retry. `describe` then showed a split status that is easy to misread:

| Field | Value |
|---|---|
| `state` | `READY` |
| `health` | `UNHEALTHY_MAINTENANCE` |

**`state: READY` does not mean the node is usable.** The state field describes
the resource, not the host; health is the field that matters for whether work
can run. A provisioning script that waits on `state=READY` alone — which is
the obvious thing to write, and what the create call itself returns on — will
hand back a node that refuses connections.

This is the cost of Spot, and it is the documented behaviour rather than a
fault: Spot capacity can be reclaimed or taken down for maintenance at any
time. For a measurement run it has a specific consequence worth planning
around: **the condition A timing is void**, because the download was cut
partway through. Condition A clears the model cache before it runs, so
repeating it is clean rather than resumed, but the attempt has to be redone
from the start.

The practical lesson for a reproduction script: poll `health`, not `state`,
and treat a mid-run disappearance as expected rather than exceptional.

### Outcome: the node was reclaimed, not recovered

Health was polled every 30 seconds. It never returned to `HEALTHY`; three
minutes later the resource moved to `DELETING` and then vanished. So this was
a preemption, not a transient maintenance window, and the node and its
downloaded weights were gone with it.

Total useful life of that node: roughly 15 minutes, enough for the install and
the backend verification but not for a first serve.

This is the reason `provision.sh` in this directory polls `health` and gives up
on a node that never becomes healthy, rather than waiting indefinitely on a
resource that is already being torn down.

### The `health` field is absent when the node is healthy

A detail that inverts the obvious check. On a working node, `describe` returns
no health field at all:

```json
{ "state": "READY" }
```

The field appears only when something is wrong — that is how
`UNHEALTHY_MAINTENANCE` surfaced during the preemption. So:

| Observed | Meaning |
|---|---|
| `state: READY`, no `health` field | usable |
| `state: READY`, `health: UNHEALTHY_MAINTENANCE` | not usable, refuses SSH |

Waiting for `health == "HEALTHY"` therefore loops forever on a perfectly good
node, while checking `state` alone accepts a node that refuses connections.
The condition that works is: `state == READY` **and** health is either absent
or `HEALTHY`.

This was a real bug in the first version of `provision.sh` in this directory,
found by checking the raw JSON of a node that had just been captured
successfully rather than trusting the field to be populated.

## Phase 3 and 5 — serving, and what the compilation cache really does

Zero-to-serving succeeded. The published v6e-1 reference recipe ran
`Qwen/Qwen3-8B` unmodified, with no flag changes and no Hugging Face token
(the model is open-weight and downloaded unauthenticated).

### Startup, three conditions

| Condition | Runs (s) | Median |
|---|---|---:|
| **A** fresh — no model cache, no compile cache | 374.1 | **374.1** |
| **B** model cached, compile cache cleared | 186.0 / 186.0 / 188.0 | **186.0** |
| **C** both cached, **process restart** | **220.1** / 72.0 / 72.0 | **72.0** |

Condition B is tight across three runs (186.0, 186.0, 188.0), so the ~186 s
figure is stable. Condition A was measured once rather than three times: each
repeat re-downloads the weights, and that cost is network speed, not a
property of vllm-torchtpu.

**Weight loading is not the bottleneck anywhere.** It took 5.4–6.2 s in all
seven runs. Condition B's 186 s is almost entirely compilation.

### Which cache variable actually works

The documentation names three variables and does not say how they relate.
Pointing all three at one root and measuring each directory per run answers it:

| Variable | Directory after the series | Verdict |
|---|---|---|
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | 151 MB | **this is the one that works** |
| `VLLM_CACHE_ROOT` | 1.2 GB, grew every single run | written, never reused |
| `VLLM_XLA_CACHE_PATH` | **0 files, 0 bytes, in all 7 runs** | never written at all |

`VLLM_XLA_CACHE_PATH` is the variable the quickstart and the environment-variable
reference tell you to set to "persist compiled artifacts across restarts". On
this build, with this model, **nothing ever wrote to it.**

The 186 s → 72 s improvement tracks `tier3` exactly. During the two fast
condition C runs that directory held 10 files and 128.2 MB and was
byte-for-byte unchanged before and after — it was read, not written.

### Why the first restart is not fast

Condition C's first run took 220 s; only the second and third hit 72 s. The
server log gives the reason directly:

```
WARNING compile_prefix isolation: Qwen3Model has no self.prefix;
        using non-deterministic fallback _tpu_instance_0
        (its on-disk compile cache will not persist across restarts)
```

So for the very model the v6e-1 recipe recommends, the vLLM-layer compile
cache key falls back to a **non-deterministic** value, and the build says
plainly that such a cache will not survive a restart. The directory sizes agree:
`VLLM_CACHE_ROOT` grew on every run and was never reused — 463 MB after
condition A, then 896 MB, 1,040 MB, 1,183 MB across the C runs. It accumulates
without bound.

What persists is the native PJRT `tier3` cache, which is keyed differently.
That is why a restart is eventually fast, and why it takes one warm run to get
there.

**Inference, not measurement:** the reading that `tier3` supplies the speedup
while the vLLM-layer cache does not is consistent with every number above, but
it was not proven by disabling `tier3` alone and re-measuring. A clean
single-variable test would set each root independently. That is the obvious
next experiment.

**One observation left unexplained:** condition A wrote 158 files / 282.7 MB
into `tier3`, while condition B runs, which also compiled from scratch, wrote
essentially nothing there (1 file, then 0, then 0). Both reached a serving
state. This asymmetry is recorded as observed and not accounted for.

### Serving facts from the log

| Item | Value |
|---|---|
| `--kv-cache-dtype fp8` resolves to | **`fp8_e5m2`**, chosen automatically for v6e |
| Compiled shape buckets | `[16, 32, 64, 128, 256, 512, 1024, 2048]` (8 buckets) |
| Weight dtype | `torch.bfloat16` |
| KV cache size | **185,088 tokens** (conditions A and B), 185,344 (condition C) |
| Engine | V1 LLM engine, `device_config=tpu`, `enforce_eager=False` |
| Compile backend | `vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor` |
| KV cache layout | LBNHC |

The build applies roughly forty named "TPU patch" callbacks to upstream vLLM at
import time, logged individually. That is worth knowing when reading a stack
trace: much of the serving path is patched rather than native upstream code.

### Against the July JAX-backend run

Same model, same single v6e chip, same `--max-model-len 8192`, TP=1. From
`experiments/tpu-v6e1-vs-a100/results/README.md`:

| | 2026-07-24, JAX backend | 2026-11-24, TorchTPU |
|---|---|---|
| Package / image | `vllm/vllm-tpu:v0.25.0` | `vllm-torchtpu==0.33.0` |
| KV cache | 98,048 tok (`fp8_e5m2` auto) | **185,088 tok** (`fp8_e5m2` auto) |
| Startup | "a few minutes" (XLA) | 186 s cold compile, 72 s warm restart |

Both backends independently chose `fp8_e5m2`. The KV cache figure is roughly
1.9× larger on the TorchTPU path at the same `max-model-len`, though the two
runs used different `--max-num-batched-tokens` (4096 then, 2048 now) and
different `--gpu-memory-utilization`, so this is not a controlled comparison.
**No throughput comparison is drawn here; that is out of scope for this
experiment.**

### A measurement bug of my own, and the correction

The per-run cache snapshots reported the Hugging Face model directory as
32,794.9 MB. `du -sh` on the same directory reported 4.5 MB. Both cannot be
right.

The cause was in this experiment's own tooling: the directory walker used
`Path.is_file()` and `Path.stat()`, both of which **follow symlinks**. The HF
hub cache stores each file once under `blobs/` and links to it from
`snapshots/`, so every weight file was counted twice — hence a figure almost
exactly double the expected bf16 size of an 8B model.

`measure_startup.py` now skips symlinks and counts them separately. The
download-size figure is therefore **not reported** from the interrupted series;
it is being re-measured with `du` on a fresh node rather than inferred. The
lesson is the same one as the memory metric in the distributed-training post:
a number that looks plausible, and happens to be an exact multiple of the
expected value, deserves a second method before it is published.

## Phase 4 — the OpenAI-compatible API

Verified on a fresh on-demand node in `europe-west4-a`. Cold start on that
node, including the download, took **380 s** — independent agreement with
condition A's 374 s measured in `us-east1-d`.

All four endpoints exist and answer. None had to be recorded as unsupported.

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | 200 | |
| `GET /v1/models` | 200 | served id is `Qwen/Qwen3-8B` |
| `POST /v1/completions` | 200 | 4 prompt / 16 completion tokens, 0.45 s |
| `POST /v1/chat/completions` | 200 | 22 prompt / 256 completion tokens, 3.68 s |

Through the OpenAI Python SDK (3.58.0), every case passed:

| Case | Result |
|---|---|
| non-streaming, English | 3.659 s, 22 prompt / 256 completion |
| non-streaming, Japanese | 3.624 s, 27 prompt / 256 completion |

> The Japanese case above was measured as stated. The published
> `smoke_test.py` has since been changed to English-only prompts (prose,
> structured list, code), because this repository is English-only, so a re-run
> will not reproduce that specific row. The measurement is kept rather than
> deleted.
| non-streaming, code generation | 3.624 s, 22 prompt / 256 completion |
| streaming, English | **TTFT 0.057 s**, 3.659 s total, 256 chunks |

### Every response stopped on `length`, and the reason is the model

All four cases returned `finish_reason: length`, never `stop`. That is not a
serving fault: Qwen3-8B is a reasoning model and opens with a `<think>` block.
At `max_tokens: 256` the entire budget is consumed by the thinking preamble, so
what comes back is reasoning that gets cut off mid-sentence and **no final
answer at all**:

```
<think>
Okay, I need to explain the difference between prefill and decode in LLM
inference. Let me start by recalling what I know about large language models
```

Anyone smoke-testing this model over the OpenAI API should budget far more than
256 output tokens, or disable thinking, or they will conclude something is
broken when the serving stack is fine.

A second observation, from the run that still used a Japanese prompt: it
produced an English `<think>` block. Since every response was truncated before
its answer, this experiment cannot say what language the final answer would
have been in. Response quality was out of scope regardless, and the published
script no longer covers non-English input.

## Phase 6 — single-request smoke benchmark

Measured with the official `vllm bench serve` CLI, one request per shape.
**This is a functional check, not a performance study.**

| Shape | In / out | TTFT | TPOT | Output tok/s | Total tok/s |
|---|---|---:|---:|---:|---:|
| short | 136 / 128 | **3362 ms** | 14.13 ms | 24.8 | 51.2 |
| prefill-heavy | 4104 / 128 | 206 ms | 14.36 ms | 63.0 | 2084.5 |
| decode-heavy | 136 / 1024 | 23 ms | 14.18 ms | 70.5 | 79.8 |

**The `short` TTFT of 3362 ms is not a valid number and must not be quoted.**
`vllm bench serve` does not run a warmup, and `short` executed first, so that
request paid first-touch compilation for its shape bucket. The two later shapes
show what TTFT actually looks like: 206 ms for a 4k prefill and 23 ms for a
128-token prefill. This is exactly the first-inference-versus-warm distinction
the experiment set out to keep separate, and the official CLI does not keep it
separate for you. The local fallback harness in `benchmark_smoke.py` does run a
warmup, but it was not used because the official CLI was available.

What *is* trustworthy here is **TPOT: 14.13, 14.36, 14.18 ms across the three
shapes** — flat to within 1.6% while input length varies by 30× and output
length by 8×. For a single-request decode that is the expected shape of the
result, and its consistency is a good sign the server is behaving sanely.

No conclusion is drawn about throughput limits, saturation or bottlenecks. One
request per shape cannot support one.

## Device memory, measured at the device

With the server up and idle, `tpu-info` reported:

```
Chip 0   28.30 GiB / 31.24 GiB   duty cycle 2.34%
```

Two things worth separating carefully, given that an earlier post in this series
conflated an allocator metric with HBM:

- **This is a device-level HBM reading from `tpu-info`, not a framework
  allocator statistic.** It is the figure the runtime reports for the chip.
- **`--gpu-memory-utilization 0.90` applies to usable HBM, not the nominal
  32 GB.** 0.90 × 31.24 GiB = 28.1 GiB, which matches the 28.30 GiB observed.
  Sizing against "32 GB" overstates the budget by about 0.75 GiB.

## Download size, re-measured

On the fresh node, with `du` rather than the directory walker:

| Path | Size |
|---|---|
| `~/.cache/huggingface/hub` | **16 GB** |
| `~/.cache/huggingface/xet` | 588 KB |
| Root disk, before | 8.7 GB used |
| Root disk, after | 30 GB used |

**16 GB** is the model, consistent with 8.2B parameters in bfloat16, in five
safetensors shards. The earlier 32.8 GB reading was the symlink
double-count described above.

One layout detail that caused the confusion and is worth recording: in this
version of `huggingface_hub` the blobs live in a **shared** `hub/blobs/` tree,
not under `hub/models--Qwen--Qwen3-8B/blobs/`. The per-model directory contains
only small files and symlinks pointing up into that shared tree, so `du` on the
model directory alone reports a few megabytes and looks like the weights are
missing.

## Cache directories, second node

The `VLLM_XLA_CACHE_PATH` result reproduced on a second, independently
provisioned node:

| Directory | Size after one cold start plus the benchmark |
|---|---|
| `tier3` | 271 MB |
| `vllm` | 457 MB |
| **`xla`** | **4.0 KB — the empty directory itself** |

Seven runs on the first node and a full session on the second, and nothing has
ever written to the path the quickstart documents for persisting compiled
artifacts.
