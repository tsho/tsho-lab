#!/usr/bin/env bash
# Capture profiles (1B, v6e-4, 2048 tokens per device) for the four
# configurations analysed: B, C, D, E. Separate from the timing runs, which
# are made without the profiler.
#
# The TorchTitan built-in profiler is used: steps 15-19 are recorded
# (profile_freq=20, warmup=3, active=5). Output per configuration:
#   <dir>/profiling/traces/iteration_20/rank{N}_trace.json.gz
#   <dir>/profiling/traces/rank_{N}/plugins/profile/<timestamp>/<hostname>.xplane.pb
# Note: the host name is part of the .xplane.pb file name.
set -uo pipefail
cd "$HOME/torchtitan"; source "$HOME/venv-titan/bin/activate"
export PATH="$HOME/.local/bin:$PATH"; export PYTHONPATH="$HOME/torchtitan:${PYTHONPATH:-}"
ROOT="${ROOT:-$HOME/prof}"; mkdir -p "$ROOT"

cap() {
  name="$1"; cfg="$2"
  out="$ROOT/$name"; rm -rf "$out"; mkdir -p "$out"
  echo "########## profile ${name} ##########"
  s=$(date +%s)
  timeout 2400 torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=bench_cfg --config="$cfg" \
    --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test \
    --training.steps=30 --training.max_context_length=2048 \
    --training.num_tokens_per_microbatch_per_dp_rank=2048 \
    --parallelism.use_simple_fsdp \
    --dump_folder="$out" \
    --profiler.enable_profiling --profiler.profile_freq=20 \
    --profiler.profiler_warmup=3 --profiler.profiler_active=5 \
    > "$out/run.log" 2>&1
  echo "${name}: exit=$? wall=$(( $(date +%s) - s ))s traces=$(find "$out" -name '*.xplane.pb' | wc -l)"
}

cap B_perblock    cfg_b_perblock
cap C_gt_eager    cfg_c_gt_eager
cap D_gt_separate cfg_d_gt_separate
cap E_gt_ingraph  cfg_e_gt_ingraph
echo "PROFILES DONE $(date -Is)"
du -sh "$ROOT"
