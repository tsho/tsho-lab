#!/usr/bin/env bash
# Build one venv per torch version, run verify.py in each, then diff.
#
#   ./run.sh                 # 2.14.1 vs 2.15.0
#   OLD=2.13.0 ./run.sh      # pick a different baseline
#
# torch 2.15.0 wheels require Python 3.11+ (3.10 was dropped), so PY must be
# 3.11 or newer.
set -euo pipefail

OLD="${OLD:-2.14.1}"
NEW="${NEW:-2.15.0}"
PY="${PY:-python3.14}"
VENVS="${VENVS:-$(mktemp -d)}"

for v in "$OLD" "$NEW"; do
  d="$VENVS/pt${v//./}"
  [ -d "$d" ] || "$PY" -m venv "$d"
  "$d/bin/pip" install -q --disable-pip-version-check "torch==$v" numpy
  "$d/bin/python" verify.py > "results/torch-$v.json"
  echo "wrote results/torch-$v.json"
done

python3 compare.py "results/torch-$OLD.json" "results/torch-$NEW.json"
