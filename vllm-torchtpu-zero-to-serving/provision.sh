#!/bin/bash
# Capture a single-chip TPU v6e VM and wait until it is actually usable.
#
# Two things this handles that an obvious script does not:
#
#  1. v6e capacity is the binding constraint, not quota. The refusal reads
#     "There is no more capacity in the zone ..." and a quota increase would
#     not help, so the only remedy is to try another zone. Expect several
#     refusals; each takes 2-3 minutes to come back.
#
#  2. `state: READY` does not mean the node is usable. A node under a
#     maintenance event reports state READY with health UNHEALTHY_MAINTENANCE
#     and refuses SSH. Wait on health, not state.
#
# Usage:
#   PROJECT=my-project ./provision.sh            # Spot (default)
#   PROJECT=my-project SPOT=0 ./provision.sh     # on-demand
set -uo pipefail

PROJECT="${PROJECT:-}"
NAME="${NAME:-vllm-torchtpu-serve}"
SPOT="${SPOT:-1}"
ZONES="${ZONES:-us-central1-b us-east5-a us-east5-b asia-northeast1-b us-east1-d europe-west4-a southamerica-west1-a}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

if [ -z "$PROJECT" ]; then
  echo "set PROJECT to your GCP project id" >&2
  exit 2
fi

FLAGS="--accelerator-type=v6e-1 --version=v2-alpha-tpuv6e"
[ "$SPOT" = "1" ] && FLAGS="$FLAGS --spot"

wait_healthy() {
  local zone="$1" waited=0 state health
  while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
    read -r state health < <(gcloud compute tpus tpu-vm describe "$NAME" \
      --project="$PROJECT" --zone="$zone" \
      --format="value(state,health)" 2>/dev/null | tr '\t' ' ')
    if [ -z "${state:-}" ]; then
      echo "  node disappeared (preempted or deleted)" >&2
      return 1
    fi
    # The health field is ABSENT from the response on a healthy node; it
    # appears only when something is wrong (e.g. UNHEALTHY_MAINTENANCE).
    # So an empty health with state READY is the good case, and waiting for
    # the literal string HEALTHY loops forever.
    if [ "$state" = "READY" ] && { [ -z "${health:-}" ] || [ "$health" = "HEALTHY" ]; }; then
      echo "  usable after ${waited}s (state=READY health=${health:-<absent>})"
      return 0
    fi
    echo "  state=$state health=${health:-?} (${waited}s)"
    sleep 15
    waited=$((waited + 15))
  done
  echo "  not healthy within ${HEALTH_TIMEOUT}s" >&2
  return 1
}

for zone in $ZONES; do
  echo "=== requesting $NAME in $zone (spot=$SPOT) ==="
  if gcloud compute tpus tpu-vm create "$NAME" \
       --project="$PROJECT" --zone="$zone" $FLAGS 2>/tmp/provision_err.log; then
    if wait_healthy "$zone"; then
      echo
      echo "CAPTURED_ZONE=$zone"
      echo "Delete it when finished:"
      echo "  gcloud compute tpus tpu-vm delete $NAME --project=$PROJECT --zone=$zone --quiet"
      exit 0
    fi
    echo "  created but never became healthy; deleting and moving on" >&2
    gcloud compute tpus tpu-vm delete "$NAME" --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null
    continue
  fi
  grep -oiE "no more capacity|quota|exhausted|permission denied|already exists" \
    /tmp/provision_err.log | head -1 || echo "  failed (see /tmp/provision_err.log)"
done

echo "no zone had capacity for v6e-1" >&2
exit 1
