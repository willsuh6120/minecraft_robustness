#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FETCH_SCRIPT="${FETCH_SCRIPT:-${ROOT_DIR}/scripts/fetch_gcp_run_artifacts.sh}"
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT:-${ROOT_DIR}/outputs/imported_vm_runs}"

FIRST_PROJECT="${FIRST_PROJECT:-envgen-491906}"
FIRST_INSTANCE_NAME="${FIRST_INSTANCE_NAME:-instance-20260417-163859}"
FIRST_ZONE="${FIRST_ZONE:-asia-southeast1-b}"

SECOND_PROJECT="${SECOND_PROJECT:-project-5549d22e-3378-435c-bd8}"
SECOND_INSTANCE_NAME="${SECOND_INSTANCE_NAME:-new-envgen}"
SECOND_ZONE="${SECOND_ZONE:-asia-southeast1-b}"

REMOTE_USER="${REMOTE_USER:-willsuh1114}"
OVERWRITE="${OVERWRITE:-0}"

usage() {
  cat <<'EOF'
usage:
  bash scripts/fetch_both_gcp_run_metrics.sh <vm1_remote_run_dir> <vm2_remote_run_dir> [vm1_local_dest] [vm2_local_dest]

examples:
  bash scripts/fetch_both_gcp_run_metrics.sh \
    /home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260426_165227 \
    /home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_straight_O2_single_env_smoke_stable_vm_20260426_1829033

  bash scripts/fetch_both_gcp_run_metrics.sh \
    <vm1_remote_run_dir> \
    <vm2_remote_run_dir> \
    /home/gyulab/envgen2/Minestudio/outputs/imported_vm_runs/vm1_metrics \
    /home/gyulab/envgen2/Minestudio/outputs/imported_vm_runs/vm2_metrics
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${FETCH_SCRIPT}" ]]; then
  echo "Fetch script not found: ${FETCH_SCRIPT}" >&2
  exit 1
fi

VM1_REMOTE_RUN_DIR="${1:-}"
VM2_REMOTE_RUN_DIR="${2:-}"
VM1_LOCAL_DEST="${3:-}"
VM2_LOCAL_DEST="${4:-}"

if [[ -z "${VM1_REMOTE_RUN_DIR}" || -z "${VM2_REMOTE_RUN_DIR}" ]]; then
  usage >&2
  exit 1
fi

VM1_REMOTE_RUN_DIR="${VM1_REMOTE_RUN_DIR%/}"
VM2_REMOTE_RUN_DIR="${VM2_REMOTE_RUN_DIR%/}"

if [[ -z "${VM1_LOCAL_DEST}" ]]; then
  VM1_LOCAL_DEST="${LOCAL_IMPORT_ROOT}/$(basename "${VM1_REMOTE_RUN_DIR}")_metrics_vm1"
fi
if [[ -z "${VM2_LOCAL_DEST}" ]]; then
  VM2_LOCAL_DEST="${LOCAL_IMPORT_ROOT}/$(basename "${VM2_REMOTE_RUN_DIR}")_metrics_vm2"
fi

echo "[fetch-both-gcp-run-metrics] vm1=${FIRST_PROJECT}/${FIRST_INSTANCE_NAME}"
FETCH_MODE=metrics_only \
CLOUDSDK_CORE_PROJECT="${FIRST_PROJECT}" \
INSTANCE_NAME="${FIRST_INSTANCE_NAME}" \
ZONE="${FIRST_ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT}" \
OVERWRITE="${OVERWRITE}" \
bash "${FETCH_SCRIPT}" "${VM1_REMOTE_RUN_DIR}" "${VM1_LOCAL_DEST}"

echo "[fetch-both-gcp-run-metrics] vm2=${SECOND_PROJECT}/${SECOND_INSTANCE_NAME}"
FETCH_MODE=metrics_only \
CLOUDSDK_CORE_PROJECT="${SECOND_PROJECT}" \
INSTANCE_NAME="${SECOND_INSTANCE_NAME}" \
ZONE="${SECOND_ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT}" \
OVERWRITE="${OVERWRITE}" \
bash "${FETCH_SCRIPT}" "${VM2_REMOTE_RUN_DIR}" "${VM2_LOCAL_DEST}"

echo "[fetch-both-gcp-run-metrics] done"
echo "[fetch-both-gcp-run-metrics] vm1_local_dest=${VM1_LOCAL_DEST}"
echo "[fetch-both-gcp-run-metrics] vm2_local_dest=${VM2_LOCAL_DEST}"
