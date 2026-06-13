#!/usr/bin/env bash
# One-command Cloud Run deploy via Cloud Build
# Usage: bash scripts/deploy.sh [--project PROJECT_ID]
set -euo pipefail

PROJECT_ID="${1:-project-d28a1a73-b11b-4cec-b2e}"
REGION="us-central1"
SERVICE="matchmind"
AR_REPO="matchmind-repo"

echo "🚀 Submitting build for project: ${PROJECT_ID}"
echo "   Service : ${SERVICE}"
echo "   Region  : ${REGION}"
echo "   Repo    : ${AR_REPO}"
echo ""

gcloud builds submit \
  --config=cloudbuild.yaml \
  --project="${PROJECT_ID}" \
  --substitutions="_REGION=${REGION},_SERVICE=${SERVICE},_AR_REPO=${AR_REPO}"

echo ""
echo "✅ Build submitted. Fetching Cloud Run URL..."

gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)"
