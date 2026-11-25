#!/bin/bash
# Create a Cloud TPU v6e VM and wait until it is usable.
#
# Notes from this experiment:
#
#  1. v6e capacity, not quota, was the binding constraint. A refusal reads
#     "There is no more capacity in the zone ..."; a quota increase does not
#     help, so the script moves on to the next zone.
#
#  2. On a healthy node the `health` field can be absent from the describe
#     output, so waiting for the literal string HEALTHY never finishes. The
#     script accepts state READY with health absent or HEALTHY.
#
# Usage:
#   PROJECT=<GCP_PROJECT> ./provision.sh                          # v6e-4, Spot
#   PROJECT=<GCP_PROJECT> ACCEL=v6e-8 ./provision.sh              # v6e-8, Spot
#   PROJECT=<GCP_PROJECT> SPOT=0 ./provision.sh                   # on-demand
#   PROJECT=<GCP_PROJECT> ZONES="us-east5-b" ./provision.sh       # one zone
set -uo pipefail

PROJECT="${PROJECT:-}"
NAME="${NAME:-tpu-bench}"
ACCEL="${ACCEL:-v6e-4}"
SPOT="${SPOT:-1}"
# Zones that offer v6e; edit to match your project's access.
ZONES="${ZONES:-us-central1-b us-east5-a us-east5-b asia-northeast1-b us-east1-d europe-west4-a southamerica-west1-a}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

if [ -z "$PROJECT" ]; then
  echo "set PROJECT to your GCP project id" >&2
  exit 2
fi

FLAGS="--accelerator-type=${ACCEL} --version=v2-alpha-tpuv6e"
[ "$SPOT" = "1" ] && FLAGS="$FLAGS --spot"
ERR="$(mktemp)"

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
  echo "=== requesting $NAME ($ACCEL) in $zone (spot=$SPOT) ==="
  if gcloud compute tpus tpu-vm create "$NAME" \
       --project="$PROJECT" --zone="$zone" $FLAGS 2>"$ERR"; then
    if wait_healthy "$zone"; then
      echo
      echo "CAPTURED_ZONE=$zone"
      echo "Delete it when finished:"
      echo "  gcloud compute tpus tpu-vm delete $NAME --project=$PROJECT --zone=$zone --quiet"
      rm -f "$ERR"
      exit 0
    fi
    echo "  created but never became healthy; deleting and moving on" >&2
    gcloud compute tpus tpu-vm delete "$NAME" --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null
    continue
  fi
  grep -oiE "no more capacity|quota|exhausted|permission denied|already exists" "$ERR" \
    | head -1 || echo "  failed (see $ERR)"
done

echo "no zone had capacity for $ACCEL" >&2
exit 1
