# Troubleshooting

Every problem hit during the 2026-11-24 run, with the command, the error text,
and what resolved it. Causes are marked **confirmed** or **suspected**.

---

## 1. `v6e-1` creation refused: no capacity

**Command**

```bash
gcloud compute tpus tpu-vm create vllm-torchtpu-serve \
  --zone=asia-northeast1-b --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e --spot
```

**Error**

```
ERROR: (gcloud.compute.tpus.tpu-vm.create) {
  "message": "There is no more capacity in the zone \"asia-northeast1-b\";
   you can try in another zone where Cloud TPU Nodes are offered"
```

**When** Four of five zones refused on the first attempt round. Over two
rounds, refusals came from `asia-northeast1-b`, `us-east5-a`, `us-east5-b` and
`us-central1-b`.

**Cause (confirmed)** Zone stock-out, stated in the message. This is not a
quota denial, so requesting more quota does not help.

**Fix** Iterate over zones. `provision.sh` does this. Budget time: refusals
took between 1.5 and 14 minutes each, and the slowest was a zone that had
succeeded an hour earlier.

---

## 2. `state: READY` on a node that refuses SSH

**Symptom**

```
WARNING: !!! This TPU is going through a maintenance event,
         and might be unavailable !!!
ssh: connect to host ... port 22: Connection refused
```

while `describe` reported:

```
state   = READY
health  = UNHEALTHY_MAINTENANCE
```

**Cause (confirmed)** `state` describes the resource, not the host. Health is
the field that reflects whether work can run.

**Fix** Poll `health`, not `state`. But see the next entry — the obvious
version of that check is also wrong.

---

## 3. Waiting for `health == "HEALTHY"` hangs forever

**Symptom** A provisioning script that polls until health equals `HEALTHY`
never returns, even on a node that is working perfectly.

**Cause (confirmed)** On a healthy node the field is **absent** from the
response:

```json
{ "state": "READY" }
```

It is populated only when something is wrong.

**Fix** Treat the node as usable when `state == READY` **and** health is either
absent or `HEALTHY`:

```bash
if [ "$state" = "READY" ] && { [ -z "$health" ] || [ "$health" = "HEALTHY" ]; }
```

This was a real bug in the first version of `provision.sh`, found by reading the
raw JSON of a node that had just been captured successfully.

---

## 4. `apt-get install libibverbs1` fails on a fresh VM

**Command**

```bash
sudo apt-get install -y libibverbs1
```

**Error**

```
E: Could not get lock /var/lib/dpkg/lock-frontend.
   It is held by process 4806 (apt-get)
E: Unable to acquire the dpkg frontend lock
```

**When** Within the first minutes of a node's life.

**Cause (confirmed)** The image runs its own apt work at boot. Under
`set -euo pipefail` this aborts the whole install script, which makes it look
worse than it is.

**Fix** Wait for the lock rather than retrying blindly. `setup.sh` waits; it
took 60 s on the measured run.

```bash
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
   || sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
  sleep 10
done
```

---

## 5. `uv venv --python 3.12` with no 3.12 present

**Cause (confirmed)** The TPU VM image ships Python **3.10.12**;
`vllm-torchtpu` requires `>=3.12,<3.13`. The official installation page opens
with `uv venv --python 3.12` and does not cover installing that interpreter on
the TPU VM image.

**Fix**

```bash
uv python install 3.12        # took ~1 s
uv venv --python 3.12 ~/uv_venv
```

---

## 6. `uv: command not found` over non-interactive SSH

**Symptom** A command that works in an interactive session fails when passed
via `gcloud ... ssh --command=`.

**Cause (confirmed)** The uv installer appends to a shell profile that a
non-interactive shell does not read.

**Fix** Source its env explicitly before use:

```bash
source $HOME/.local/bin/env
source ~/uv_venv/bin/activate
```

---

## 7. `vllm serve --help` appears to list none of the flags

**Symptom** Grepping the output of `vllm serve --help` for `--kv-cache-dtype`,
`--max-model-len` and so on finds nothing, suggesting the build lacks them.

**Cause (confirmed)** `vllm serve --help` prints only a list of config groups.

**Fix** Use `vllm serve --help=all` (2,462 lines). All eight flags in the
published v6e-1 recipe are present, and `fp8` is one of 22 accepted
`--kv-cache-dtype` values.

---

## 8. Persistent compile cache appears not to work on the first restart

**Symptom** With the model and compilation caches both retained, the first
process restart still took 220 s; only later restarts reached 72 s.

**Cause (confirmed by the server's own log)**

```
WARNING compile_prefix isolation: Qwen3Model has no self.prefix;
        using non-deterministic fallback _tpu_instance_0
        (its on-disk compile cache will not persist across restarts)
```

The vLLM-layer cache key is non-deterministic for this model, so that cache is
never reused — `VLLM_CACHE_ROOT` grew on every run, 463 MB to 1.2 GB, without
being pruned.

**Also confirmed** `VLLM_XLA_CACHE_PATH` was never written: 0 bytes across
seven runs on one node and a whole session on another. It is nonetheless the
variable the quickstart recommends for this purpose.

**Suspected** The restart speedup comes from
`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`, which held 10 files and 128.2 MB
unchanged across the two fast runs. Not proven: tier3 was not disabled on its
own and re-measured.

**Practical fix** Set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent
path, expect one warm run before restarts get fast, and watch
`VLLM_CACHE_ROOT` for unbounded growth.

---

## 9. Every API response stops on `length`, never `stop`

**Cause (confirmed)** Qwen3-8B is a reasoning model and begins with a
`<think>` block. At `max_tokens: 256` the budget is consumed by the thinking
preamble and no final answer is produced — the serving stack is fine.

**Fix** Allow far more output tokens, or disable thinking.

---

## 10. Measurement errors made here, and their corrections

Recorded because both produced plausible-looking numbers.

**Directory walker double-counted the model.** Per-run snapshots reported the
Hugging Face cache as 32,794.9 MB; `du` reported 16 GB. `Path.is_file()` and
`Path.stat()` follow symlinks, and the hub stores each file once under a
**shared** `hub/blobs/` tree with symlinks from `snapshots/`, so every weight
was counted twice. The figure was almost exactly double the expected bf16 size
of an 8B model, which is what prompted the second check. `measure_startup.py`
now skips symlinks. **The `hf_model` byte counts inside
`results/startup_*.json` are from the uncorrected version** — use 16 GB.

**Log sanitizer redacted loopback.** The first version replaced every
IPv4-looking string, turning 196 occurrences of `127.0.0.1` into `x.x.x.x`.
That address is not host-identifying and is the evidence that the endpoint was
bound to localhost. `sanitize_logs.py` now preserves loopback and wildcard
addresses and redacts only routable ones.
