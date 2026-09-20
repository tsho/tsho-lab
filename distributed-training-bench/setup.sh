#!/usr/bin/env bash
#
# Deep Learning VM (common-cu129-ubuntu-2404-nvidia-580) 上での環境構築。
#
#   HF_TOKEN=hf_xxx bash setup.sh
#
# NGC/DLVM の torch は CUDA 最適化済みなので入れ替えない。
# DeepSpeed とその周辺だけを既存の torch に合わせて追加する。

set -euo pipefail

echo "== GPU 確認 =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo
echo "== GPU 間トポロジ (NVLink があるかどうかがここで分かる) =="
nvidia-smi topo -m || true
echo

python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpus :", torch.cuda.device_count())
print("nccl :", ".".join(map(str, torch.cuda.nccl.version())))
PY

echo
echo "== 依存パッケージ =="
# torch は DLVM のものを使う。--no-deps は付けず、torch だけ固定する。
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
  echo "== HuggingFace ログイン =="
  python3 -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

echo
echo "== DeepSpeed 環境レポート =="
ds_report || true

cat <<'EOF'

セットアップ完了。

まず1本試す:
  deepspeed --num_gpus=4 bench_zero_stages.py --stage 3 --steps 15

全部回す:
  bash run_all.sh

注意:
  - 既定は乱数初期化 (--real-weights なし)。重みのダウンロードを待たずに
    システム挙動だけ測れる。実重みで測りたいときだけ --real-weights を付ける。
  - 損失値は合成データなので意味を持たない。記事に載せないこと。
EOF
