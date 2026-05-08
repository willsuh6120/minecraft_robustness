#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SYNC_SCRIPT="${SYNC_SCRIPT:-${ROOT_DIR}/scripts/sync_gcp_code_delta.sh}"

FIRST_PROJECT="${FIRST_PROJECT:-envgen-491906}"
FIRST_INSTANCE_NAME="${FIRST_INSTANCE_NAME:-instance-20260417-163859}"
FIRST_ZONE="${FIRST_ZONE:-asia-southeast1-b}"

SECOND_PROJECT="${SECOND_PROJECT:-project-5549d22e-3378-435c-bd8}"
SECOND_INSTANCE_NAME="${SECOND_INSTANCE_NAME:-new-envgen}"
SECOND_ZONE="${SECOND_ZONE:-asia-southeast1-b}"

REMOTE_USER="${REMOTE_USER:-willsuh1114}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/willsuh1114/envgen2/Minestudio}"

if [[ ! -f "${SYNC_SCRIPT}" ]]; then
  echo "Sync script not found: ${SYNC_SCRIPT}" >&2
  exit 1
fi

echo "[sync-both] first=${FIRST_PROJECT}/${FIRST_INSTANCE_NAME}"
CLOUDSDK_CORE_PROJECT="${FIRST_PROJECT}" \
INSTANCE_NAME="${FIRST_INSTANCE_NAME}" \
ZONE="${FIRST_ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT}" \
bash "${SYNC_SCRIPT}"

echo "[sync-both] second=${SECOND_PROJECT}/${SECOND_INSTANCE_NAME}"
CLOUDSDK_CORE_PROJECT="${SECOND_PROJECT}" \
INSTANCE_NAME="${SECOND_INSTANCE_NAME}" \
ZONE="${SECOND_ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT}" \
bash "${SYNC_SCRIPT}"

echo "[sync-both] done"
