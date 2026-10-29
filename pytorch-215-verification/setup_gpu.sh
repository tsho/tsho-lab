#!/usr/bin/env bash
# Build one venv per torch version on a CUDA host, both on the cu132 build.
#
# cu132 is the only CUDA build published for both 2.14.1 and 2.15.0, so it is
# what keeps the comparison honest. torch 2.15.0 dropped the cp310 wheel, and
# GCP's deep-learning images ship Python 3.10, so fetch a newer interpreter.
set -euo pipefail

OLD="${OLD:-2.14.1}"
NEW="${NEW:-2.15.0}"
PYVER="${PYVER:-3.12}"
INDEX="https://download.pytorch.org/whl/cu132"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install "$PYVER"
PYBIN="$(uv python find "$PYVER")"

for v in "$OLD" "$NEW"; do
  d="$HOME/pt${v//./}"
  [ -d "$d" ] || "$PYBIN" -m venv "$d"
  "$d/bin/pip" install -q --upgrade pip
  "$d/bin/pip" install -q "torch==$v" numpy --index-url "$INDEX"
  "$d/bin/python" - <<'PY'
import torch
print(f"{torch.__version__}  cuda_runtime={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
PY
done
