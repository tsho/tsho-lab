# Runbook — running the benchmark on a TPU v6e-4

From `git pull` to deleting the VM. About 30 minutes once the TPU is up.
This is the procedure that produced the numbers in [`results/`](results/)
(2026-09-06, single-host TPU v6e-4, Spot, `asia-northeast1-b`).

Replace `YOUR_PROJECT` and the zone with your own.

## Phase 0: get the code (Cloud Shell or any machine with `gcloud`)

```bash
git clone https://github.com/tsho/tsho-lab.git   # or: cd tsho-lab && git pull
cd tsho-lab/jax-sharding-bench
```

## Phase 1: create the TPU VM (Cloud TPU API, Spot)

```bash
gcloud compute tpus tpu-vm create tpu-v6e \
    --project=YOUR_PROJECT \
    --zone=asia-northeast1-b \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --spot
```

- `v6e-4` is a single host with 4 chips. `v2-alpha-tpuv6e` is the TPU runtime image;
  it ships with Python 3.10 and `pip`.
- Spot has no queue: the command either succeeds (about a minute in our run) or fails
  right away when the zone has no capacity. If it fails, try another v6e zone.
- A Spot TPU can be preempted at any time. The benchmark appends one JSON line per
  finished run, so a preemption only costs the run in flight.

## Phase 2: copy the script and log in

```bash
gcloud compute tpus tpu-vm scp bench.py tpu-v6e: \
    --zone=asia-northeast1-b --project=YOUR_PROJECT
gcloud compute tpus tpu-vm ssh tpu-v6e \
    --zone=asia-northeast1-b --project=YOUR_PROJECT
```

## Phase 3: install JAX and run (inside the VM)

```bash
pip install -q -U "jax[tpu]"
python3 -c "import jax; print(jax.__version__, jax.default_backend(), len(jax.devices()))"
# expected: <version> tpu 4

# 0.87B (defaults: --dim 2048 --layers 16)
python3 bench.py --mode dp   --tag 0p8b
python3 bench.py --mode fsdp --tag 0p8b
# 3.27B
python3 bench.py --mode dp   --dim 3072 --layers 28 --heads 24 --tag 3p2b
python3 bench.py --mode fsdp --dim 3072 --layers 28 --heads 24 --tag 3p2b

cat results/*.jsonl
```

On the image's Python 3.10, `pip` resolves `jax[tpu]` to jax 0.6.2 / libtpu 0.0.17,
because newer JAX releases require a newer Python. To run a newer JAX, use a separate
interpreter and virtualenv:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv
python3.12 -m venv ~/jax011
~/jax011/bin/pip install -q -U "jax[tpu]==0.11.1"
~/jax011/bin/python bench.py --mode dp --tag 0p8b-jax0111
```

Use a different `--tag` per software stack. The tag is the only place the stack is
recorded (see [`results/README.md`](results/README.md)).

## Phase 4: collect the results and delete the VM

```bash
exit   # leave the VM
gcloud compute tpus tpu-vm scp --recurse tpu-v6e:results ./tpu-results \
    --zone=asia-northeast1-b --project=YOUR_PROJECT
gcloud compute tpus tpu-vm delete tpu-v6e \
    --zone=asia-northeast1-b --project=YOUR_PROJECT --quiet
```

Nothing deletes the VM for you. Delete it as soon as the results are copied.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `create` fails immediately with a capacity error | No Spot capacity in that zone right now. Try another v6e zone, or another time of day. |
| `The TPU is already in use by process with pid N`, and `ps -fp N` shows no such process | A stale lock from an earlier run. `sudo rm -f /tmp/libtpu_lockfile`, then rerun. |
| Backend prints `cpu` instead of `tpu` | `jax[tpu]` is not installed in the interpreter you are running. Check which `python`/`pip` you used. |
| Out of memory | Lower `--dim` or `--layers`. Do not assume `fsdp` fits where `dp` does not: at 3.27B on v6e-4, `fsdp` peaked higher per device than `dp` (18.9 GB vs 13.0 GB). |

## What did not work for us: a GCE instance with `FLEX_START`

We first tried to get the TPU as a Compute Engine instance
(`gcloud compute instances create --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START`,
image family `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`). We record it here so you can
recognize the same failure.

- `--request-valid-for-duration` is capped at 2 hours
  (`must be at least 90 and less or equal to 7200 seconds`). When the request expires,
  the pending instance disappears without an error or a notification.
- Requests in `us-east5-a` and `us-east5-b` expired unfulfilled. `europe-west4-a`
  provisioned a VM twice.
- On both VMs JAX could not open the TPU:
  `TPU initialization failed: GRPC_ERROR error message: START_SESSION failed`.
  This happened with jax 0.6.2 / libtpu 0.0.17 and with jax 0.10.2 / libtpu 0.0.42.1,
  and a reboot did not change it. `/dev/vfio/0-3` were present and world-writable, and
  `tpu-info` listed all four chips.
- The same `bench.py` on the same jax 0.6.2 / libtpu 0.0.17 ran on the first attempt
  on a VM created through the Cloud TPU API (Phase 1 above). We did not find the cause
  of the GCE-path failure.
- That Ubuntu image also has no `pip`; install it with `sudo apt-get install -y python3-pip`.
