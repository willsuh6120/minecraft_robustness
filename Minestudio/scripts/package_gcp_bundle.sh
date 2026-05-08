#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(cd "${ROOT}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-minestudio}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${WORK_ROOT}/gcp_transfer}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

GENERATED_TASK_GROUP_DIR="${GENERATED_TASK_GROUP_DIR:-${ROOT}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
FIXED_BANK_ROOT="${FIXED_BANK_ROOT:-${ROOT}/outputs/evaluate_rocket/fixed_eval_bank_mines_v1/20260414_030828}"
LOCK_PATH="${LOCK_PATH:-${WORK_ROOT}/minestudio-linux-64.lock}"

BUNDLE_DIR="${TRANSFER_ROOT}/bundle_${TIMESTAMP}"
PAYLOAD_DIR="${BUNDLE_DIR}/payload"
REPO_PAYLOAD_DIR="${PAYLOAD_DIR}/Minestudio"
ARTIFACTS_DIR="${PAYLOAD_DIR}/artifacts"

mkdir -p "${REPO_PAYLOAD_DIR}" "${ARTIFACTS_DIR}"

echo "[1/5] exporting conda lock: ${LOCK_PATH}"
conda list --explicit -n "${CONDA_ENV_NAME}" > "${LOCK_PATH}"

echo "[2/5] copying repo working tree"
rsync -a \
  --exclude ".git" \
  --exclude "logs" \
  --exclude "outputs" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache" \
  "${ROOT}/" "${REPO_PAYLOAD_DIR}/"

echo "[3/5] copying required experiment artifacts"
mkdir -p "${ARTIFACTS_DIR}/generated_task_group" "${ARTIFACTS_DIR}/fixed_eval_bank_mines_v1"
rsync -a "${GENERATED_TASK_GROUP_DIR}/" "${ARTIFACTS_DIR}/generated_task_group/$(basename "${GENERATED_TASK_GROUP_DIR}")/"
rsync -a "${FIXED_BANK_ROOT}/" "${ARTIFACTS_DIR}/fixed_eval_bank_mines_v1/$(basename "${FIXED_BANK_ROOT}")/"
cp "${LOCK_PATH}" "${PAYLOAD_DIR}/minestudio-linux-64.lock"

cat > "${PAYLOAD_DIR}/README_GCP_BUNDLE.txt" <<EOF
Bundle contents:
- Minestudio repo working tree snapshot
- generated_task_group source at:
  outputs/evaluate_rocket/generated_task_group/$(basename "${GENERATED_TASK_GROUP_DIR}")
- fixed eval bank at:
  outputs/evaluate_rocket/fixed_eval_bank_mines_v1/$(basename "${FIXED_BANK_ROOT}")
- conda explicit lock:
  minestudio-linux-64.lock
EOF

echo "[4/5] creating tarball"
(
  cd "${BUNDLE_DIR}"
  tar -czf "bundle_${TIMESTAMP}.tar.gz" payload
)

echo "[5/5] done"
echo "Bundle dir: ${BUNDLE_DIR}"
echo "Tarball:    ${BUNDLE_DIR}/bundle_${TIMESTAMP}.tar.gz"
echo
echo "Optional next copies:"
echo "  gcloud compute scp ${BUNDLE_DIR}/bundle_${TIMESTAMP}.tar.gz <INSTANCE>:~/ --zone=<ZONE>"
echo "  gcloud compute scp ${ROOT}/scripts/setup_gcp_vm.sh <INSTANCE>:~/ --zone=<ZONE>"

