#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/torchtitan"
source "$HOME/venv-titan/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/torchtitan:${PYTHONPATH:-}"

COMMON=(
  --module=bench_cfg --config=llama3_1b_untied
  --hf_assets_path=tests/assets/tokenizer
  --dataloader.dataset_path=tests/assets/c4_test
  --training.steps=30
  --training.max_context_length=2048
  --training.num_tokens_per_microbatch_per_dp_rank=2048
  --parallelism.use_simple_fsdp
)

run() {
  name="$1"; shift
  log="$HOME/ladder_${name}.log"
  echo "########## ${name} ##########"
  echo "flags: $*"
  start=$(date +%s)
  timeout 2400 torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    "${COMMON[@]}" "$@" > "$log" 2>&1
  rc=$?
  echo "${name}: exit=${rc} wall=$(( $(date +%s) - start ))s"
  [ $rc -ne 0 ] && { echo "--- error ---"; grep -E "RuntimeError|Error:|Exception|RESOURCE_EXHAUSTED|Segmentation" "$log" | head -4; }
  echo
}

run rung1_eager   --tpu_config.eager_mode=DEFER_NEVER
run rung2_compile --compile.enable --compile.backend=tpu
run rung3_graph   --tpu_config.use_graph_trainer --tpu_config.compile_mode=aot_fx_trace --tpu_config.graph_trainer_memory_policy=default
run rung4_optim   --tpu_config.use_graph_trainer --tpu_config.compile_mode=aot_fx_trace --tpu_config.graph_trainer_memory_policy=default --tpu_config.optimizer_in_graph
echo "ALL DONE"
