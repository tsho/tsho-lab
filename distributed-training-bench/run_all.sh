#!/usr/bin/env bash
#
# ZeRO stage 0/1/2/3 を同一条件で順に測る。記事の段階1がこれ1本で終わる。
#
#   bash run_all.sh
#   MODEL=Qwen/Qwen2.5-7B GPUS=4 bash run_all.sh
#   SEQ=4096 bash run_all.sh            # 長系列で ZeRO の効きを見る
#
# OOM したステージがあってもスクリプトは止まらない。
# 「どこで OOM するか」自体が測定結果なので、results/*.jsonl に oom=true で残る。

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
  sleep 5   # GPU メモリの解放待ち
done

# ZeRO-3 + CPU offload は「メモリは足りるが遅い」ことを示すための追加点。
echo "############################################################"
echo "# ZeRO stage 3 + CPU offload"
echo "############################################################"
deepspeed --num_gpus="$GPUS" bench_zero_stages.py \
  --model "$MODEL" --stage 3 --offload \
  --micro-batch "$MICRO_BS" --seq-len "$SEQ" --steps "$STEPS" \
  --profile --tag "$TAG" || echo ">> stage 3+offload failed (recorded)"

echo
echo "完了。結果は results/*.jsonl"
echo "表にする:  python3 summarize.py"
