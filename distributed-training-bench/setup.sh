#!/usr/bin/env bash
#
# Environment setup on a Deep Learning VM (common-cu129-ubuntu-2404-nvidia-580).
#
#   HF_TOKEN=hf_xxx bash setup.sh
#
# The torch that ships with NGC/DLVM is already built for CUDA, so it is left in place.
# Only DeepSpeed and its companions are added, matched to the existing torch.

set -euo pipefail

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo
echo "== GPU topology (this shows whether NVLink is present) =="
nvidia-smi topo -m || true
echo

python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpus :", torch.cuda.device_count())
print("nccl :", ".".join(map(str, torch.cuda.nccl.version())))
PY

echo
echo "== Dependencies =="
# Use the DLVM torch. Do not pass --no-deps; pin torch only.
pip install --upgrade pip
pip install \
  "deepspeed>=0.18.2" \
  "transformers>=4.56,<5.0.0" \
  "accelerate" \
  "nvidia-ml-py" \
  "sentencepiece" \
  "protobuf"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo
  echo "== Hugging Face login =="
  python3 -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

echo
echo "== DeepSpeed environment report =="
ds_report || true

cat <<'EOF'

Setup complete.

Try one run first:
  deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --steps 15

Run everything:
  bash run_all.sh

Notes:
  - The default is random initialization (no --real-weights), so system behavior
    can be measured without downloading weights. Pass --real-weights only when you need real ones.
  - Loss values come from synthetic data and mean nothing. Do not report them.
EOF
