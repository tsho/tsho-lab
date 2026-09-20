#!/usr/bin/env bash
#
# Create a GPU VM for the distributed-training benchmark.
# Follows vllm-benchmark-scripts/launch-gpu.sh, with multi-GPU shapes added.
#
# Usage:
#   PROJECT=p ZONE=us-central1-a TYPE=a100x4 ./launch-gpu.sh   # NVLink
#   PROJECT=p ZONE=us-central1-a TYPE=l4x4    ./launch-gpu.sh   # PCIe only (control)
#   PROJECT=p ZONE=us-central1-a TYPE=a100x2  ./launch-gpu.sh   # smallest shape
#   SPOT=1 PROJECT=p ZONE=us-central1-a TYPE=a100x4 ./launch-gpu.sh
#
# Prerequisite: confirm quota and stock first with check-quota.sh
#       (vllm-benchmark-scripts/check-quota.sh works as is)
#
# Why both a100x4 and l4x4:
#   ZeRO-3 issues a large volume of all-gather / reduce-scatter every step,
#   so the GPU-to-GPU interconnect becomes the bottleneck directly.
#     A100 (NVLink, ~600GB/s)  vs  L4 (PCIe Gen4 only, ~64GB/s)
#   Running the same code on both changes the share of time spent in NCCL substantially.
#   That difference is the measured evidence for what it means to saturate a GPU.

set -euo pipefail

PROJECT="${PROJECT:?PROJECT is required}"
ZONE="${ZONE:?ZONE is required (e.g. us-central1-a)}"
TYPE="${TYPE:?TYPE=a100x4 | a100x2 | l4x4 | h100x2}"
DISK_SIZE="${DISK_SIZE:-500GB}"
SPOT="${SPOT:-0}"

case "$TYPE" in
  a100x4)
    MACHINE=a2-ultragpu-4g
    ACCEL="type=nvidia-a100-80gb,count=4"
    ;;
  a100x2)
    MACHINE=a2-ultragpu-2g
    ACCEL="type=nvidia-a100-80gb,count=2"
    ;;
  l4x4)
    # L4 has no NVLink. It serves as the PCIe-only control.
    MACHINE=g2-standard-48
    ACCEL="type=nvidia-l4,count=4"
    ;;
  h100x2)
    MACHINE=a3-highgpu-2g
    ACCEL="type=nvidia-h100-80gb,count=2"
    ;;
  *) echo "TYPE must be one of: a100x4 a100x2 l4x4 h100x2" >&2; exit 2 ;;
esac

NAME="${NAME:-dtb-$TYPE}"

# Deep Learning VM (CUDA 12.9 + NVIDIA driver 580 + Ubuntu 24.04 LTS)。
# Same image as vllm-benchmark-scripts, so the environment does not differ.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

EXTRA=()
if [[ "$SPOT" == "1" ]]; then
  # Spot is enough for experiments, and its preemptions are useful material for the fault-injection stage.
  EXTRA+=(--provisioning-model=SPOT --instance-termination-action=DELETE)
else
  EXTRA+=(--restart-on-failure)
fi

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform \
  "${EXTRA[@]}"

cat <<EOF

Created: $NAME (zone=$ZONE, machine=$MACHINE, spot=$SPOT)

Next steps:
  gcloud compute scp --recurse setup.sh bench_zero_stages.py ds_configs run_all.sh \\
      $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT

  # inside the VM:
  #   HF_TOKEN=hf_xxx bash setup.sh
  #   bash run_all.sh                 # runs stage 1/2/3 + baseline in order
  #   # to run one configuration:
  #   deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --model Qwen/Qwen2.5-3B

Do not forget to delete the VM:
  gcloud compute instances delete $NAME --zone=$ZONE --project=$PROJECT --quiet
EOF
