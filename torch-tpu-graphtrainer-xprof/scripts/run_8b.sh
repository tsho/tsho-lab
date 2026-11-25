#!/usr/bin/env bash
# Llama 3 8B on v6e-8 (eight chips, 2048 tokens per device): does the 1B
# ordering hold? Runs per-block compile, the graph trainer with the default
# memory policy, the memory policies that cut HBM on 1B, and profiles.
#
# Note: the llama3_8b recipe already enables per-block compile, so
# 8b_A_eager measures the same configuration as 8b_B_perblock.
# Note: with the default memory policy the graph-trainer runs fail on v6e
# with HBM errors; only in_graph + full_and_offload completed here.
#
# Usage (on the v6e-8 VM, after setup.sh; bench_cfg.py copied into
# ~/torchtitan, analyze_dir.py in $SCRIPTS, default $HOME):
#   ./run_8b.sh
set -uo pipefail
source "$HOME/venv-titan/bin/activate"; export PATH="$HOME/.local/bin:$PATH"
pip install -q tpu-info 2>/dev/null || true
OUT="${OUT:-$HOME/p6}"; mkdir -p "$OUT"
SCRIPTS="${SCRIPTS:-$HOME}"
TOK="${TOK:-2048}"
args() { echo --module=bench_cfg --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset_path=tests/assets/c4_test --training.steps=30 \
  --training.max_context_length=2048 \
  --training.num_tokens_per_microbatch_per_dp_rank="$TOK" --parallelism.use_simple_fsdp; }
run() { name="$1"; cfg="$2"; shift 2; log="$OUT/${name}.log"
  echo "########## ${name} ##########"
  rm -f "$OUT/${name}.hbm"
  ( while true; do tpu-info 2>/dev/null | grep -E "GiB /" >> "$OUT/${name}.hbm"; sleep 2; done ) & sp=$!
  s=$(date +%s)
  ( cd "$HOME/torchtitan" && PYTHONPATH="$HOME/torchtitan" timeout 3000 torchrun --nproc_per_node=8 \
      -m torchtitan.experiments.tpu.train $(args) --config="$cfg" "$@" > "$log" 2>&1 ); rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "${name}: exit=${rc} wall=$(( $(date +%s) - s ))s steps=$(grep -c 'step:' "$log")"
  [ $rc -ne 0 ] && grep -E "RuntimeError|Error:|RESOURCE_EXHAUSTED|ValueError|OOM|out of memory|Ran out of memory" "$log" | sort -u | head -4 | cut -c1-260
  grep -E "step: 30" "$log" | head -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c60-
  echo; }

run 8b_B_perblock  cfg_8b_b_perblock
run 8b_C_gt_eager  cfg_8b_c_gt_eager
run 8b_E_ingraph   cfg_8b_e_gt_ingraph
run 8b_A_eager     cfg_8b_a_eager
# Memory policies that reduced HBM on 1B:
run 8b_E_offload   cfg_8b_e_offload
run 8b_C_offload   cfg_8b_c_offload
run 8b_E_full      cfg_8b_e_full

# Profiles. Only B completed here; the two graph-trainer profile runs failed
# with the same HBM errors as their timed runs.
for pair in "B_perblock cfg_8b_b_perblock" "C_gt_eager cfg_8b_c_gt_eager" "E_gt_ingraph cfg_8b_e_gt_ingraph"; do
  set -- $pair; d="$OUT/prof/$1"; rm -rf "$d"; mkdir -p "$d"
  ( cd "$HOME/torchtitan" && PYTHONPATH="$HOME/torchtitan" timeout 3000 torchrun --nproc_per_node=8 \
      -m torchtitan.experiments.tpu.train $(args) --config="$2" --dump_folder="$d" \
      --profiler.enable_profiling --profiler.profile_freq=20 \
      --profiler.profiler_warmup=3 --profiler.profiler_active=5 > "$d/run.log" 2>&1 )
  echo "profile 8b $1: exit=$? traces=$(find "$d" -name '*.xplane.pb' | wc -l)"
done
python "$SCRIPTS/analyze_dir.py" "$OUT" "$OUT/p6_summary.json"
echo "DONE $(date -Is)"
