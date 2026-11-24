#!/bin/bash
# Install vllm-torchtpu on a fresh Cloud TPU v6e VM.
#
# Usage (on the TPU VM):
#   ./setup.sh 2>&1 | tee ~/setup.log
#
# Records timings to ~/setup_timings.txt so the install cost is measurable.
set -euo pipefail

VENV="${VENV:-$HOME/uv_venv}"
TIMINGS="$HOME/setup_timings.txt"
: > "$TIMINGS"

mark() { echo "$1 $(date +%s)" >> "$TIMINGS"; echo "--- $1 ($(date +%H:%M:%S)) ---"; }

mark begin

# --- Environment facts, before anything is installed ----------------------
mark record_environment
{
  echo "## uname"; uname -a
  echo "## os-release"; cat /etc/os-release
  echo "## cpu"; nproc; grep -m1 "model name" /proc/cpuinfo || true
  echo "## memory"; free -g
  echo "## disk"; df -h /
  echo "## default python"; python3 --version
  echo "## tpu devices"; ls -l /dev/vfio/ || echo "no /dev/vfio"
  echo "## accel devices"; ls -l /dev/accel* 2>/dev/null || echo "no /dev/accel*"
} > "$HOME/environment.txt" 2>&1

# --- Prerequisite: libibverbs.so.1 ----------------------------------------
# Without it, tpu-sync-torch fails to import.
# https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/getting_started/installation/
mark apt_libibverbs
# A freshly created TPU VM runs its own apt work at boot (unattended-upgrades
# and the image's package setup), so dpkg is locked for the first minutes and
# an immediate install dies with:
#   E: Could not get lock /var/lib/dpkg/lock-frontend
# Wait for the lock instead of failing.
wait_for_dpkg() {
  local waited=0
  while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
     || sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
    if [ "$waited" -ge 600 ]; then
      echo "dpkg still locked after ${waited}s; giving up" >&2
      return 1
    fi
    echo "  dpkg locked, waiting... (${waited}s)"
    sleep 10
    waited=$((waited + 10))
  done
  echo "  dpkg free after ${waited}s"
  echo "dpkg_lock_wait_seconds $waited" >> "$TIMINGS"
}
wait_for_dpkg
sudo apt-get update -qq
sudo apt-get install -y -qq libibverbs1

# --- uv -------------------------------------------------------------------
mark install_uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "$HOME/.local/bin/env"

# --- Python 3.12 ----------------------------------------------------------
# vllm-torchtpu requires >=3.12,<3.13. The TPU VM image ships 3.10, so the
# interpreter has to be installed rather than assumed.
mark install_python312
uv python install 3.12

mark create_venv
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- vllm-torchtpu --------------------------------------------------------
# UV_TORCH_BACKEND=cpu routes torch/torchvision to the PyTorch CPU index.
# Without it, pip pulls gigabytes of nvidia-* wheels that are useless on TPU.
mark install_vllm_torchtpu
export UV_TORCH_BACKEND=cpu
uv pip install vllm-torchtpu

mark record_versions
uv pip freeze > "$HOME/pip_versions.txt"
python - <<'PY' > "$HOME/versions.json"
import json, platform, importlib.metadata as md

def ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

info = {
    "python": platform.python_version(),
    "packages": {
        n: ver(n)
        for n in ("vllm", "vllm-torchtpu", "torch", "torch-tpu", "torch_tpu",
                  "libtpu", "libtpu-nightly", "jax", "jaxlib",
                  "tpu-sync-torch", "tpu-info", "transformers")
    },
}
try:
    import torch
    info["torch_version_runtime"] = torch.__version__
except Exception as e:
    info["torch_import_error"] = repr(e)
try:
    import torch_tpu
    info["torch_tpu_version_runtime"] = getattr(torch_tpu, "__version__", "unknown")
except Exception as e:
    info["torch_tpu_import_error"] = repr(e)
print(json.dumps(info, indent=2))
PY

# --- Verify the TorchTPU backend is the one vLLM selected -----------------
mark verify_backend
echo "## current_platform.device_name"
python -c "from vllm.platforms import current_platform; print(current_platform.device_name)"
echo "## tpu-info"
tpu-info || echo "tpu-info failed"
echo "## JAX-backend packages that must NOT be present"
uv pip list 2>/dev/null | grep -iE "tpu-inference|vllm-jax" || echo "none (expected)"

mark end
echo
echo "=== timings (unix seconds) ==="
cat "$TIMINGS"
echo
echo "Wrote: ~/environment.txt ~/pip_versions.txt ~/versions.json ~/setup_timings.txt"
