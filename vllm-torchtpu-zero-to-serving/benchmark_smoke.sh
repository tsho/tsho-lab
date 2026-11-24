#!/bin/bash
# Single-request smoke benchmark across three shapes. This is a functional
# check, not a performance study: one request per shape, no concurrency
# sweep, no tuning. Do not draw conclusions about throughput limits from it.
#
#   short         128 in / 128 out
#   prefill-heavy 4096 in / 128 out
#   decode-heavy  128 in / 1024 out
#
# Warmup is run first and reported separately, so the first-request
# compilation cost never lands in the measured numbers.
set -euo pipefail

VENV="${VENV:-$HOME/uv_venv}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
OUT="${OUT:-results/benchmark_smoke.json}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$(dirname "$OUT")"

if command -v vllm >/dev/null && vllm bench serve --help >/dev/null 2>&1; then
  echo "=== using the official vllm bench serve CLI ==="
  for shape in "128 128 short" "4096 128 prefill_heavy" "128 1024 decode_heavy"; do
    set -- $shape
    echo "--- $3 ($1 in / $2 out) ---"
    vllm bench serve \
      --backend openai-chat \
      --model "$MODEL" \
      --base-url "http://127.0.0.1:$PORT" \
      --endpoint /v1/chat/completions \
      --dataset-name random \
      --random-input-len "$1" \
      --random-output-len "$2" \
      --num-prompts 1 \
      --save-result \
      --result-filename "results/bench_$3.json" || echo "$3 failed"
  done
else
  echo "=== vllm bench serve unavailable; using the local harness ==="
  python benchmark_smoke.py --model "$MODEL" --port "$PORT" --out "$OUT"
fi
