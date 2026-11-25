#!/usr/bin/env bash
# TorchTitan + TorchTPU on a Cloud TPU v6e VM, for the GraphTrainer XProf study.
# Follows torchtitan/experiments/tpu/README.md. The VM image ships Python 3.10;
# the recipes require 3.14, so uv fetches a standalone interpreter.
set -euo pipefail

TITAN_DIR="${TITAN_DIR:-$HOME/torchtitan}"
VENV="${VENV:-$HOME/venv-titan}"
PYVER="${PYVER:-3.14}"
# Empty = track main and record whatever SHA lands. Set to pin.
TITAN_SHA="${TITAN_SHA:-}"

# No dpkg wait here: this setup installs nothing through apt (uv and pip only).
# An earlier version waited for the boot-time apt lock and gave up after 600 s
# on a v6e-8 node whose lock never cleared, for no benefit.
echo "::: 1/6 uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "::: 2/6 Python ${PYVER}"
uv python install "${PYVER}"
PYBIN="$(uv python find "${PYVER}")"
"$PYBIN" --version

echo "::: 3/6 clone torchtitan"
if [ ! -d "$TITAN_DIR/.git" ]; then
  git clone https://github.com/google-pytorch/torchtitan "$TITAN_DIR"
fi
cd "$TITAN_DIR"
if [ -n "$TITAN_SHA" ]; then
  git checkout --quiet "$TITAN_SHA"
fi
# Record the exact tree this run used. The previous experiment used 89ef29b;
# this one tracks main because optimizer_compile="separate_graph" is newer.
git rev-parse HEAD | tee "$HOME/torchtitan_commit.txt"
git log -1 --format='%cI %s' | tee "$HOME/torchtitan_head.txt"

echo "::: 4/6 venv + TPU deps"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip "setuptools<81"   # tensorboard needs <81

# torch 2.15.0 is released, so the stable CPU index is correct. Fall back to the
# test index if the pin is not there yet. Dropping --extra-index-url entirely
# pulls a CUDA build, which is useless on TPU.
pip install -q torch==2.15.0 torch_tpu==0.2.0 \
  -r torchtitan/experiments/tpu/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  || pip install -q torch==2.15.0 torch_tpu==0.2.0 \
       -r torchtitan/experiments/tpu/requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/test/cpu

echo "::: 5/6 torchtitan deps + profiler"
# tokamax and tyro declare conflicting typeguard versions; pip warns but completes.
pip install -q -r requirements.txt || echo "WARN: requirements.txt reported conflicts"
# transformers is missing from the TPU requirements.
pip install -q transformers || true
pip install -q xprof tensorboard tensorboard-plugin-profile || echo "WARN: profiler install issue"

echo "::: 6/6 versions"
pip freeze > "$HOME/pip_versions.txt"
python - <<'PY' | tee "$HOME/versions.json"
import importlib, json, sys
out = {"python": sys.version.split()[0]}
for m in ("torch", "torch_tpu", "torchtitan", "libtpu", "numpy",
          "xprof", "tensorboard", "transformers"):
    try:
        out[m] = getattr(importlib.import_module(m), "__version__", "(no __version__)")
    except Exception as e:
        out[m] = f"ERR: {type(e).__name__}"
print(json.dumps(out, indent=2))
PY
echo "::: setup done"
