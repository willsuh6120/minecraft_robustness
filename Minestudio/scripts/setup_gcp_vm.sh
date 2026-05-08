#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle-tar.gz>"
  exit 1
fi

BUNDLE_TARBALL="$1"
TARGET_ROOT="${TARGET_ROOT:-${HOME}/envgen2}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-minestudio}"
REPO_ROOT="${TARGET_ROOT}/Minestudio"
STAGING_ROOT="${TARGET_ROOT}/gcp_bundle_staging"

mkdir -p "${TARGET_ROOT}" "${STAGING_ROOT}"

echo "[1/6] system packages"
sudo apt-get update
sudo apt-get install -y tmux ffmpeg rsync

echo "[2/6] extracting bundle"
tar -xzf "${BUNDLE_TARBALL}" -C "${STAGING_ROOT}"

PAYLOAD_DIR="${STAGING_ROOT}/payload"
if [[ ! -d "${PAYLOAD_DIR}" ]]; then
  echo "payload directory not found under ${STAGING_ROOT}"
  exit 1
fi

echo "[3/6] syncing repo"
mkdir -p "${REPO_ROOT}"
rsync -a --delete "${PAYLOAD_DIR}/Minestudio/" "${REPO_ROOT}/"

echo "[4/6] restoring required artifacts"
mkdir -p "${REPO_ROOT}/outputs/evaluate_rocket/generated_task_group"
mkdir -p "${REPO_ROOT}/outputs/evaluate_rocket/fixed_eval_bank_mines_v1"
rsync -a "${PAYLOAD_DIR}/artifacts/generated_task_group/" "${REPO_ROOT}/outputs/evaluate_rocket/generated_task_group/"
rsync -a "${PAYLOAD_DIR}/artifacts/fixed_eval_bank_mines_v1/" "${REPO_ROOT}/outputs/evaluate_rocket/fixed_eval_bank_mines_v1/"

echo "[5/6] creating conda env"
conda create -n "${CONDA_ENV_NAME}" --file "${PAYLOAD_DIR}/minestudio-linux-64.lock" -y

echo "[6/6] writing shell env"
{
  echo "export MINESTUDIO_DIR=${TARGET_ROOT}/.minestudio"
  echo "export HF_HOME=${HOME}/.cache/huggingface"
  echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "export ROOT=${REPO_ROOT}"
} >> "${HOME}/.bashrc"

echo
echo "Setup complete."
echo "Repo root: ${REPO_ROOT}"
echo "Next:"
echo "  source ~/.bashrc"
echo "  conda activate ${CONDA_ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  nvidia-smi"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
