#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FETCH_SCRIPT="${FETCH_SCRIPT:-${ROOT_DIR}/scripts/fetch_gcp_run_artifacts.sh}"
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT:-${ROOT_DIR}/outputs/imported_vm_runs}"

PROJECT_ID="${PROJECT_ID:-envgen-491906}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260417-163859}"
ZONE="${ZONE:-asia-southeast1-b}"
REMOTE_USER="${REMOTE_USER:-willsuh1114}"
OVERWRITE="${OVERWRITE:-0}"

usage() {
  cat <<'EOF'
usage:
  bash scripts/fetch_vm1_progress_metrics.sh <remote_run_dir> [local_dest_dir]

example:
  bash scripts/fetch_vm1_progress_metrics.sh \
    /home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260427_071429

notes:
  - fetches current metric files even if the run is still in progress
  - also fetches referenced collect/eval worldgen manifests, bank manifests,
    worldgen plans, collect block plans, and generated task yaml files
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REMOTE_RUN_DIR="${1:-}"
LOCAL_DEST_DIR="${2:-}"

if [[ -z "${REMOTE_RUN_DIR}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${FETCH_SCRIPT}" ]]; then
  echo "Fetch script not found: ${FETCH_SCRIPT}" >&2
  exit 1
fi

REMOTE_RUN_DIR="${REMOTE_RUN_DIR%/}"
if [[ -z "${LOCAL_DEST_DIR}" ]]; then
  LOCAL_DEST_DIR="${LOCAL_IMPORT_ROOT}/$(basename "${REMOTE_RUN_DIR}")_progress_vm1"
fi

FETCH_MODE=metrics_with_worldgen_refs \
CLOUDSDK_CORE_PROJECT="${PROJECT_ID}" \
INSTANCE_NAME="${INSTANCE_NAME}" \
ZONE="${ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT}" \
OVERWRITE="${OVERWRITE}" \
bash "${FETCH_SCRIPT}" "${REMOTE_RUN_DIR}" "${LOCAL_DEST_DIR}"
