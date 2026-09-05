#!/usr/bin/env bash
# distributed-training-bench on GKE — end-to-end runbook.
#
#   PROJECT=YOUR_PROJECT BUCKET=YOUR_PROJECT-dtb ./run-gke.sh build     # image (Cloud Build)
#   PROJECT=YOUR_PROJECT ./run-gke.sh cluster                        # GKE cluster (no GPU yet)
#   PROJECT=YOUR_PROJECT ./run-gke.sh pool-l4                        # L4x4 Spot node pool
#   PROJECT=YOUR_PROJECT BUCKET=YOUR_PROJECT-dtb ./run-gke.sh jobs-l4    # all bench configs on L4
#   PROJECT=YOUR_PROJECT ./run-gke.sh pool-a100                      # swap to A100-40GBx4 Spot
#   PROJECT=YOUR_PROJECT BUCKET=YOUR_PROJECT-dtb ./run-gke.sh jobs-a100
#   PROJECT=YOUR_PROJECT ./run-gke.sh down                           # DELETE EVERYTHING billable
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-dtb}
BUCKET=${BUCKET:-${PROJECT}-dtb}
REPO=${REPO:-dtb}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bench:latest"

case "${1:?build|cluster|pool-l4|jobs-l4|pool-a100|jobs-a100|down}" in
build)
  gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com container.googleapis.com --project "$PROJECT"
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --project "$PROJECT" 2>/dev/null || true
  gsutil mb -l "$REGION" -p "$PROJECT" "gs://${BUCKET}" 2>/dev/null || true
  (cd "$(dirname "$0")/.." && gcloud builds submit --project "$PROJECT" --config gke/cloudbuild.yaml --substitutions=_IMAGE="$IMAGE" .)
  ;;
cluster)
  gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --num-nodes 1 --machine-type e2-standard-2 \
    --scopes cloud-platform --release-channel regular
  gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT"
  ;;
pool-l4)
  gcloud container node-pools create l4x4 --cluster "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --machine-type g2-standard-48 --accelerator "type=nvidia-l4,count=4,gpu-driver-version=latest" \
    --num-nodes 1 --spot --scopes cloud-platform
  ;;
pool-a100)
  gcloud container node-pools delete l4x4 --cluster "$CLUSTER" --project "$PROJECT" --zone "$ZONE" --quiet 2>/dev/null || true
  gcloud container node-pools create a100x4 --cluster "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --machine-type a2-highgpu-4g --accelerator "type=nvidia-tesla-a100,count=4,gpu-driver-version=latest" \
    --num-nodes 1 --spot --scopes cloud-platform
  ;;
jobs-l4)   GKE_ACCEL=nvidia-l4          "$(dirname "$0")/submit-jobs.sh" l4 ;;
jobs-a100) GKE_ACCEL=nvidia-tesla-a100  "$(dirname "$0")/submit-jobs.sh" a100 ;;
down)
  gcloud container clusters delete "$CLUSTER" --project "$PROJECT" --zone "$ZONE" --quiet
  echo "cluster deleted. (bucket gs://${BUCKET} は残しています — 結果回収後に手動削除可)"
  ;;
esac
