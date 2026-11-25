#!/usr/bin/env bash
# 1B on v6e-4: device HBM sampled with tpu-info, the four memory policies,
# the 8192-tokens-per-device batch (timed and profiled), and a re-run of the
# earlier ladder's TorchTitan commit 89ef29b with the flags it used then.
#
# Usage (on the TPU VM, after setup.sh; bench_cfg.py copied into ~/torchtitan,
# bench_cfg_old.py and analyze_dir.py in $SCRIPTS, default $HOME):
#   ./run_memory_batch_oldcommit.sh
set -uo pipefail
source "$HOME/venv-titan/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
pip install -q tpu-info 2>/dev/null || true
OUT="${OUT:-$HOME/day2}"; mkdir -p "$OUT"
SCRIPTS="${SCRIPTS:-$HOME}"

common() {  # $1 = tokens per device
  echo --module=bench_cfg --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test --training.steps=30 \
    --training.max_context_length=2048 \
    --training.num_tokens_per_microbatch_per_dp_rank="$1" \
    --parallelism.use_simple_fsdp
}

# Sample device-level HBM every two seconds while a run is going. This is the
# runtime's per-chip figure, not the memory figure TorchTitan logs.
sample_hbm() {  # $1 = output file
  while true; do
    tpu-info 2>/dev/null | grep -E "GiB /" >> "$1"
    echo "--- $(date +%s)" >> "$1"
    sleep 2
  done
}

run() {  # name cfg tokens [extra flags...]
  name="$1"; cfg="$2"; tok="$3"; shift 3
  log="$OUT/${name}.log"
  rm -f "$OUT/${name}.hbm"
  echo "########## ${name} ##########"
  sample_hbm "$OUT/${name}.hbm" & sp=$!
  s=$(date +%s)
  ( cd "$HOME/torchtitan" && PYTHONPATH="$HOME/torchtitan" timeout 2400 \
      torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
      $(common "$tok") --config="$cfg" "$@" > "$log" 2>&1 )
  rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "${name}: exit=${rc} wall=$(( $(date +%s) - s ))s steps=$(grep -c 'step:' "$log")"
  [ $rc -ne 0 ] && grep -E "RuntimeError|Error:|RESOURCE_EXHAUSTED|ValueError|OOM" "$log" | head -4
  grep -E "step: 30" "$log" | head -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c60-
  echo
}

echo "===== device HBM vs logged memory, 2048 tokens ($(date -Is)) ====="
run hbm_B_perblock  cfg_b_perblock   2048
run hbm_E_ingraph   cfg_e_gt_ingraph 2048

echo "===== memory policies on in_graph ====="
run mp_default          cfg_mp_default          2048
run mp_full             cfg_mp_full             2048
run mp_tpu_selective    cfg_mp_tpu_selective    2048
run mp_full_and_offload cfg_mp_full_and_offload 2048

echo "===== 8192 tokens per device ====="
run big_A_eager     cfg_a_eager      8192
run big_B_perblock  cfg_b_perblock   8192
run big_C_gt_eager  cfg_c_gt_eager   8192
run big_E_ingraph   cfg_e_gt_ingraph 8192

echo "===== profiles at 8192 tokens per device ====="
for pair in "B_perblock cfg_b_perblock" "C_gt_eager cfg_c_gt_eager" "E_gt_ingraph cfg_e_gt_ingraph"; do
  set -- $pair
  d="$OUT/prof_big/$1"; rm -rf "$d"; mkdir -p "$d"
  ( cd "$HOME/torchtitan" && PYTHONPATH="$HOME/torchtitan" timeout 2400 \
      torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
      $(common 8192) --config="$2" --dump_folder="$d" \
      --profiler.enable_profiling --profiler.profile_freq=20 \
      --profiler.profiler_warmup=3 --profiler.profiler_active=5 > "$d/run.log" 2>&1 )
  echo "profile big $1: exit=$? traces=$(find "$d" -name '*.xplane.pb' | wc -l)"
done

echo "===== earlier ladder's commit 89ef29b, flags as used then ====="
OLD="$HOME/torchtitan_old"
if [ ! -d "$OLD/.git" ]; then
  git clone -q https://github.com/google-pytorch/torchtitan "$OLD"
fi
( cd "$OLD" && git checkout -q 89ef29b5a0d03a6883079ab37ff1bdc498fe3485 && git rev-parse HEAD )
cp "$SCRIPTS/bench_cfg_old.py" "$OLD/bench_cfg.py"
oldrun() {
  name="$1"; shift; log="$OUT/${name}.log"
  s=$(date +%s)
  ( cd "$OLD" && PYTHONPATH="$OLD" timeout 2400 torchrun --nproc_per_node=4 \
      -m torchtitan.experiments.tpu.train \
      --module=bench_cfg --config=llama3_1b_untied \
      --hf_assets_path=tests/assets/tokenizer --dataloader.dataset_path=tests/assets/c4_test \
      --training.steps=30 --training.max_context_length=2048 \
      --training.num_tokens_per_microbatch_per_dp_rank=2048 \
      --parallelism.use_simple_fsdp "$@" > "$log" 2>&1 )
  echo "${name}: exit=$? wall=$(( $(date +%s) - s ))s steps=$(grep -c 'step:' "$log")"
  grep -E "step: 30" "$log" | head -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c60-
  echo
}
oldrun old_rung2_compile --compile.enable --compile.backend=tpu
oldrun old_rung3_graph   --tpu_config.use_graph_trainer --tpu_config.compile_mode=aot_fx_trace --tpu_config.graph_trainer_memory_policy=default
oldrun old_rung4_optim   --tpu_config.use_graph_trainer --tpu_config.compile_mode=aot_fx_trace --tpu_config.graph_trainer_memory_policy=default --tpu_config.optimizer_in_graph

python "$SCRIPTS/analyze_dir.py" "$OUT" "$OUT/day2_summary.json"
echo "DONE $(date -Is)"
