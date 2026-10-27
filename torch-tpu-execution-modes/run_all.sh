#!/usr/bin/env bash
# End-to-end runner for TorchTPU 0.2.0 execution-mode benchmark on TPU v6e-1.
#
# Usage:
#   1) Run directly on an existing TPU v6e-1 VM:
#        bash run_all.sh --on-vm
#   2) Provision a Spot TPU v6e-1 VM from local machine, run all benchmarks,
#      fetch results, and delete the TPU VM automatically:
#        bash run_all.sh --provision [--zone asia-northeast1-b] [--name torchtpu-modes]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-}"
TPU_NAME="torchtpu-modes"
TPU_ZONE="asia-northeast1-b"
TPU_TYPE="v6e-1"
TPU_VERSION="v2-alpha-tpuv6e"
MODE="on-vm"
KEEP_VM="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --on-vm)
      MODE="on-vm"
      shift
      ;;
    --provision)
      MODE="provision"
      shift
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --zone)
      TPU_ZONE="$2"
      shift 2
      ;;
    --name)
      TPU_NAME="$2"
      shift 2
      ;;
    --keep-vm)
      KEEP_VM="1"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

run_on_vm() {
  echo "=== [1/5] Preparing Python 3.12 virtual environment on TPU VM ==="
  cd "${SCRIPT_DIR}"

  if ! command -v python3.12 >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y python3.12 python3.12-venv
  fi

  # Capture verbatim log of pip install WITHOUT --extra-index-url (--dry-run)
  python3.12 -m venv .venv_dryrun_test
  echo "=== Capturing pip install torch_tpu==0.2.0 WITHOUT --extra-index-url ==="
  .venv_dryrun_test/bin/pip install --upgrade pip >/dev/null 2>&1 || true
  .venv_dryrun_test/bin/pip install --dry-run torch_tpu==0.2.0 > pip_no_extra_index_dryrun.log 2>&1 || true
  rm -rf .venv_dryrun_test

  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip

  echo "=== [2/5] Installing torch_tpu==0.2.0 with CPU index URL ==="
  pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu | tee pip_install_torchtpu.log

  echo "=== [3/5] Running 4 core execution modes (1 process per mode) ==="
  # Clear any shared memory Tier2 cache before baseline runs
  rm -rf /dev/shm/torch_tpu_cache || true

  # Mode A: Strict eager (default DEFER_NEVER)
  python bench_modes.py \
    --mode eager \
    --dump-versions versions.json \
    --output results_eager.json \
    --tag "mode_a_strict_eager"

  # Mode B: DEFER_AND_FUSE via environment variable (locked at import)
  python bench_modes.py \
    --mode defer_fuse \
    --output results_defer_fuse.json \
    --tag "mode_b_defer_and_fuse_env"

  # Mode B-ctx: DEFER_AND_FUSE via torch.tpu.set_eager_mode context manager
  python bench_modes.py \
    --mode defer_fuse_ctx \
    --output results_defer_fuse_ctx.json \
    --tag "mode_b_defer_and_fuse_ctx"

  # Mode C: torch.compile (model forward + loss)
  python bench_modes.py \
    --mode compile \
    --output results_compile.json \
    --tag "mode_c_torch_compile"

  # Mode C-full: torch.compile wrapping full step (fwd + bwd + manual SGD)
  python bench_modes.py \
    --mode compile_step \
    --output results_compile_step.json \
    --tag "mode_c_torch_compile_full_step"

  # Mode D: SingleTraceTrainer with torch.compile
  python bench_modes.py \
    --mode single_trace \
    --output results_single_trace.json \
    --tag "mode_d_single_trace_trainer"

  # Mode D-static: SingleTraceTrainer with StaticCompiler + input donation
  python bench_modes.py \
    --mode single_trace_static \
    --output results_single_trace_static.json \
    --tag "mode_d_single_trace_static_donated"

  echo "=== [4/5] Measuring Tier-2 Compilation Cache effect (cold vs warm) ==="
  rm -rf /dev/shm/torch_tpu_cache/bench_tier2 || true
  python bench_modes.py \
    --mode compile \
    --steps 5 \
    --warmup 2 \
    --tier2-cache bench_tier2 \
    --output results_tier2_cold.json \
    --tag "tier2_cache_cold"

  python bench_modes.py \
    --mode compile \
    --steps 5 \
    --warmup 2 \
    --tier2-cache bench_tier2 \
    --output results_tier2_warm.json \
    --tag "tier2_cache_warm"

  echo "=== [5/5] Measuring Dynamic Shapes (seq=512,1024,2048) ==="
  # Note: --compile-dynamic none (Bounded Dynamism) fails on 2D causal attention mask (1,16,T,T)
  # with stablehlo.broadcast_in_dim single bounded dimension limit; --compile-dynamic false recompiles cleanly.
  python bench_modes.py \
    --mode compile \
    --steps 5 \
    --warmup 2 \
    --dynamic-seqs "512,1024,2048" \
    --compile-dynamic false \
    --output results_dynamic_shapes.json \
    --tag "dynamic_shapes_static_recompile"

  echo "=== All benchmarks finished on TPU VM ==="
}

provision_and_run() {
  echo "=== Target GCP Project: ${PROJECT} | Zone: ${TPU_ZONE} | TPU: ${TPU_NAME} (${TPU_TYPE}) ==="
  cleanup_tpu() {
    if [[ "${KEEP_VM}" == "0" ]]; then
      echo "=== Deleting TPU VM ${TPU_NAME} (${TPU_ZONE}, project=${PROJECT}) ==="
      gcloud compute tpus tpu-vm delete "${TPU_NAME}" --project="${PROJECT}" --zone="${TPU_ZONE}" --quiet || true
    else
      echo "=== Keeping TPU VM ${TPU_NAME} (${TPU_ZONE}, project=${PROJECT}) as requested (--keep-vm) ==="
    fi
  }
  trap cleanup_tpu EXIT

  echo "=== Creating Spot TPU VM ${TPU_NAME} (${TPU_TYPE}) in ${TPU_ZONE} (project=${PROJECT}) ==="
  gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${TPU_ZONE}" \
    --accelerator-type="${TPU_TYPE}" \
    --version="${TPU_VERSION}" \
    --spot

  echo "=== Copying benchmark scripts to ${TPU_NAME} ==="
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --project="${PROJECT}" --zone="${TPU_ZONE}" \
    --command="mkdir -p ~/torch-tpu-execution-modes"
  gcloud compute tpus tpu-vm scp \
    "${SCRIPT_DIR}/bench_modes.py" \
    "${SCRIPT_DIR}/run_all.sh" \
    "${TPU_NAME}:~/torch-tpu-execution-modes/" \
    --project="${PROJECT}" \
    --zone="${TPU_ZONE}"

  echo "=== Executing benchmark suite on ${TPU_NAME} ==="
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --project="${PROJECT}" --zone="${TPU_ZONE}" \
    --command="bash ~/torch-tpu-execution-modes/run_all.sh --on-vm"

  echo "=== Copying results back to ${SCRIPT_DIR} ==="
  gcloud compute tpus tpu-vm scp \
    "${TPU_NAME}:~/torch-tpu-execution-modes/*.json" \
    "${TPU_NAME}:~/torch-tpu-execution-modes/*.log" \
    "${SCRIPT_DIR}/" \
    --project="${PROJECT}" \
    --zone="${TPU_ZONE}"
}

if [[ "${MODE}" == "on-vm" ]]; then
  run_on_vm
else
  provision_and_run
fi
