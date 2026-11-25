#!/usr/bin/env bash
# Timed runs of the eight optimizer-compilation configurations (1B, v6e-4).
# Profiling is not enabled here: timing and tracing are separate runs.
#
# Usage (on the TPU VM, after setup.sh, with bench_cfg.py copied into
# ~/torchtitan):
#   ./run_benchmarks.sh                              # all eight
#   ./run_benchmarks.sh B_perblock F_perblock_separate   # a subset, by name
set -uo pipefail
cd "$HOME/torchtitan"
source "$HOME/venv-titan/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/torchtitan:${PYTHONPATH:-}"

OUT="${OUT:-$HOME/bench}"
STEPS="${STEPS:-30}"
TOKENS="${TOKENS:-2048}"
mkdir -p "$OUT"

COMMON=(
  --module=bench_cfg
  --hf_assets_path=tests/assets/tokenizer
  --dataloader.dataset_path=tests/assets/c4_test
  --training.steps="${STEPS}"
  --training.max_context_length=2048
  --training.num_tokens_per_microbatch_per_dp_rank="${TOKENS}"
  --parallelism.use_simple_fsdp
)

ONLY=" $* "
run() {
  name="$1"; cfg="$2"; shift 2
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $name "* ]]; then return; fi
  log="$OUT/${name}.log"
  echo "########## ${name} (${cfg}) ##########"
  start=$(date +%s)
  timeout 2400 torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    "${COMMON[@]}" --config="${cfg}" "$@" > "$log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - start ))
  echo "${name}: exit=${rc} wall=${wall}s"
  if [ $rc -ne 0 ]; then
    echo "--- error ---"
    grep -E "RuntimeError|Error:|Exception|RESOURCE_EXHAUSTED|Segmentation|ValueError" "$log" | head -5
  else
    grep -E "step:" "$log" | tail -2
  fi
  echo
}

echo "===== optimizer-compilation configurations: $(date -Is) ====="
run A_eager              cfg_a_eager
run B_perblock           cfg_b_perblock
run C_gt_eager           cfg_c_gt_eager
run D_gt_separate        cfg_d_gt_separate
run E_gt_ingraph         cfg_e_gt_ingraph
run F_perblock_separate  cfg_f_perblock_separate
run C2_gt_eager_cgn      cfg_c2_gt_eager_cgn
run D2_gt_separate_cgn   cfg_d2_gt_separate_cgn
echo "ALL DONE $(date -Is)"
echo "Summarise with: python analyze.py $OUT   (writes summary.json)"
