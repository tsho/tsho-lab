#!/usr/bin/env bash
# Render job.yaml.tmpl per config and run SEQUENTIALLY (one 4-GPU node).
set -euo pipefail
TAG=${1:?l4|a100}
PROJECT=${PROJECT:?}; REGION=${REGION:-us-central1}
BUCKET=${BUCKET:-${PROJECT}-dtb}; REPO=${REPO:-dtb}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bench:latest"
GKE_ACCEL=${GKE_ACCEL:?}

declare -a NAMES CMDS
for S in 0 1 2 3; do
  NAMES+=("${TAG}-zero${S}")
  CMDS+=("deepspeed --num_gpus=4 bench_zero_stages.py --stage ${S} --steps 15 --profile")
done
NAMES+=("${TAG}-zero3-offload"); CMDS+=("deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --offload --steps 15 --profile")
NAMES+=("${TAG}-fsdp-reshard");  CMDS+=("torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward --steps 15 --profile")
NAMES+=("${TAG}-fsdp-noreshard");CMDS+=("torchrun --nproc_per_node=4 bench_fsdp.py --no-reshard-after-forward --steps 15 --profile")
NAMES+=("${TAG}-fsdp-offload"); CMDS+=("torchrun --nproc_per_node=4 bench_fsdp.py --reshard-after-forward --offload --steps 15 --profile")

MODE=${MODE:-watch}   # MODE=all で一括投入して即終了 (K8s が1本ずつ実行・GCSへ自動アップロード)

cd "$(dirname "$0")"
render() {
  JOB_NAME="$1" BENCH_CMD="$2" IMAGE="$IMAGE" BUCKET="$BUCKET" GKE_ACCEL="$GKE_ACCEL" python3 - <<'PY'
import os, pathlib, re
t = pathlib.Path("job.yaml.tmpl").read_text()
print(re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), t))
PY
}
if [ "$MODE" = "all" ]; then
  for i in "${!NAMES[@]}"; do
    echo "=== submit ${NAMES[$i]} ==="
    render "${NAMES[$i]}" "${CMDS[$i]}" | kubectl apply -f -
  done
  echo ""
  echo "${#NAMES[@]}本を投入しました。ノードは 4 GPU なので K8s が自動で1本ずつ実行します。"
  echo "セッションを閉じてOK。進捗確認 (いつでも・再接続後に):"
  echo "  kubectl get jobs"
  echo "  kubectl get pods"
  echo "  gsutil ls gs://${BUCKET}/"
  exit 0
fi

for i in "${!NAMES[@]}"; do
  N="${NAMES[$i]}"; C="${CMDS[$i]}"
  echo "=== ${N} ==="
  render "$N" "$C" | kubectl apply -f -
  for _ in $(seq 1 240); do
    ST=$(kubectl get job "dtb-${N}" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "")
    [ "$ST" = "Complete" ] && break
    [ "$ST" = "Failed" ] && { echo "!! ${N} FAILED"; kubectl logs "job/dtb-${N}" --tail=40 || true; break; }
    sleep 10
  done
  kubectl logs "job/dtb-${N}" --tail=12 2>/dev/null || true
done
echo "done. results: gs://${BUCKET}/"
