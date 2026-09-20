#!/usr/bin/env bash
#
# Measure ZeRO stage 0/1/2/3 in order under identical conditions.
#
#   bash run_all.sh
#   MODEL=Qwen/Qwen2.5-7B GPUS=4 bash run_all.sh
#   SEQ=4096 bash run_all.sh            # longer sequences, to see where ZeRO matters
#
# The script keeps going when a stage runs out of memory.
# Where it runs out of memory is itself a result, recorded as oom=true in results/*.jsonl.

set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
MICRO_BS="${MICRO_BS:-1}"
SEQ="${SEQ:-2048}"
STEPS="${STEPS:-30}"
TAG="${TAG:-}"

echo "model=$MODEL gpus=$GPUS micro_bs=$MICRO_BS seq=$SEQ steps=$STEPS"
echo

for STAGE in 0 1 2 3; do
  echo "############################################################"
  echo "# ZeRO stage $STAGE"
  echo "############################################################"
  deepspeed --num_gpus="$GPUS" bench_zero_stages.py \
    --model "$MODEL" --stage "$STAGE" \
    --micro-batch "$MICRO_BS" --seq-len "$SEQ" --steps "$STEPS" \
    --profile --tag "$TAG" || echo ">> stage $STAGE failed (recorded)"
  echo
  sleep 5   # wait for GPU memory to be released
done

# ZeRO-3 + CPU offload is an extra point: it fits in memory, but it is slow.
echo "############################################################"
echo "# ZeRO stage 3 + CPU offload"
echo "############################################################"
deepspeed --num_gpus="$GPUS" bench_zero_stages.py \
  --model "$MODEL" --stage 3 --offload \
  --micro-batch "$MICRO_BS" --seq-len "$SEQ" --steps "$STEPS" \
  --profile --tag "$TAG" || echo ">> stage 3+offload failed (recorded)"

echo
echo "Done. Results are in results/*.jsonl"
echo "Make a table:  python3 summarize.py"
