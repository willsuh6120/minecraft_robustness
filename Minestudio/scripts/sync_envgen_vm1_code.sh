#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export CLOUDSDK_CORE_PROJECT="${CLOUDSDK_CORE_PROJECT:-envgen-491906}"
export INSTANCE_NAME="${INSTANCE_NAME:-instance-20260417-163859}"
export ZONE="${ZONE:-asia-southeast1-b}"
export REMOTE_USER="${REMOTE_USER:-willsuh1114}"
export REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/${REMOTE_USER}/envgen2/Minestudio}"
export VM_EXTERNAL_IP="${VM_EXTERNAL_IP:-34.143.131.243}"

if [[ "${1:-}" == "--all" ]]; then
  export SYNC_MODE="all_code"
  shift
else
  export SYNC_MODE="${SYNC_MODE:-changed}"
fi

echo "[sync-envgen-vm1] project=${CLOUDSDK_CORE_PROJECT}"
echo "[sync-envgen-vm1] instance=${INSTANCE_NAME}"
echo "[sync-envgen-vm1] zone=${ZONE}"
echo "[sync-envgen-vm1] external_ip=${VM_EXTERNAL_IP}"
echo "[sync-envgen-vm1] remote=${REMOTE_USER}@${REMOTE_REPO_ROOT}"
echo "[sync-envgen-vm1] sync_mode=${SYNC_MODE}"

bash "${ROOT_DIR}/scripts/sync_gcp_code_delta.sh" "$@"
