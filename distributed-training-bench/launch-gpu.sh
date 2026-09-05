#!/usr/bin/env bash
#
# 分散学習ベンチ用の GPU VM を作成する。
# vllm-benchmark-scripts/launch-gpu.sh を踏襲し、multi-GPU 構成を追加したもの。
#
# 使い方:
#   PROJECT=p ZONE=us-central1-a TYPE=a100x4 ./launch-gpu.sh   # NVLink 側
#   PROJECT=p ZONE=us-central1-a TYPE=l4x4    ./launch-gpu.sh   # PCIe のみ (対照)
#   PROJECT=p ZONE=us-central1-a TYPE=a100x2  ./launch-gpu.sh   # 最小構成
#   SPOT=1 PROJECT=p ZONE=us-central1-a TYPE=a100x4 ./launch-gpu.sh
#
# 前提: check-quota.sh で在庫・クォータを確保しておくこと
#       (vllm-benchmark-scripts/check-quota.sh がそのまま使える)
#
# なぜ a100x4 と l4x4 の両方か:
#   ZeRO-3 は毎 step で大量の all-gather / reduce-scatter を出すため、
#   GPU 間の相互接続がそのままボトルネックになる。
#     A100 (NVLink, ~600GB/s)  vs  L4 (PCIe Gen4 のみ, ~64GB/s)
#   同じコードを両方で走らせると NCCL 時間の占める割合が大きく変わる。
#   これが「GPU を saturate させるとは何か」を実測で示す材料になる。

set -euo pipefail

PROJECT="${PROJECT:?PROJECT 必須}"
ZONE="${ZONE:?ZONE 必須 (例: us-central1-a)}"
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
    # L4 は NVLink を持たない。PCIe のみの対照群として使う。
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
# vllm-benchmark-scripts と同じイメージを使い、環境差を作らない。
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

EXTRA=()
if [[ "$SPOT" == "1" ]]; then
  # Spot は実験用途では十分。段階4 (障害注入) の題材としてもむしろ好都合。
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

次のステップ:
  gcloud compute scp --recurse setup.sh bench_zero_stages.py ds_configs run_all.sh \\
      $NAME:~/ --zone=$ZONE --project=$PROJECT
  gcloud compute ssh $NAME --zone=$ZONE --project=$PROJECT

  # VM 内で:
  #   HF_TOKEN=hf_xxx bash setup.sh
  #   bash run_all.sh                 # stage 1/2/3 + baseline を順に実行
  #   # 個別に回すなら:
  #   deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --model Qwen/Qwen2.5-3B

削除を忘れずに:
  gcloud compute instances delete $NAME --zone=$ZONE --project=$PROJECT --quiet
EOF
