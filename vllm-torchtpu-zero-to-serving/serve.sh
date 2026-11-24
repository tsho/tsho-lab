#!/bin/bash
# Serve Qwen/Qwen3-8B on a single TPU v6e chip with vllm-torchtpu.
#
# The flags are the published v6e-1 reference recipe, unchanged:
#   https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/deployment/v6e_recipes/
# That page notes the v6e recipes are reference configurations sized for v6e
# HBM capacity, and are NOT run in CI (CI covers v7x only).
#
# Usage:
#   ./serve.sh                       # serve with all three caches enabled
#   CACHE_ROOT=/tmp/x ./serve.sh     # point the caches somewhere else
set -euo pipefail

VENV="${VENV:-$HOME/uv_venv}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/vllm_torchtpu_cache}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The docs name three compilation-cache variables and do not say how they
# relate. Point all three inside one root so the experiment can show what
# each one actually writes.
#   TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT -- native PJRT binaries (.bin)
#     https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/user_guide/compilation_cache/
#   VLLM_CACHE_ROOT -- Python-level artifacts (Dynamo bytecode, guards, ...)
#     same page; described there as "completely independent" of the above
#   VLLM_XLA_CACHE_PATH -- "persistent XLA/PJRT compilation cache artifacts"
#     https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/reference/env_vars/
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/tier3"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export VLLM_XLA_CACHE_PATH="$CACHE_ROOT/xla"
mkdir -p "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$VLLM_CACHE_ROOT" "$VLLM_XLA_CACHE_PATH"

# Surfaces any recompilation that happens after warmup, which is what
# distinguishes a working persistent cache from a warm process.
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"

echo "=== cache roots ==="
echo "  tier3: $TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
echo "  vllm:  $VLLM_CACHE_ROOT"
echo "  xla:   $VLLM_XLA_CACHE_PATH"
echo "=== starting at $(date -Is) ==="

# --host 127.0.0.1 keeps the endpoint off the public internet. Reach it from
# a workstation with:
#   gcloud compute tpus tpu-vm ssh NAME --zone=ZONE -- -N -L 8000:localhost:8000
exec vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --host 127.0.0.1 \
  --port "$PORT"
