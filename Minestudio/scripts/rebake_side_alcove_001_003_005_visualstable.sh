#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/gyulab/miniconda3/envs/minestudio/bin/python}"
PROTOCOL="${PROTOCOL:-ours_v1}"
TASK="${TASK:-mine_coal}"
BASE_SEED="${BASE_SEED:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-/home/gyulab/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-/home/gyulab/envgen2/.cache/huggingface}"

SRC_WG_DIR="${SRC_WG_DIR:-/home/gyulab/envgen2/Minestudio/outputs/evaluate_rocket/recheck_side_alcove_left_O2_centerlane_20260420_160209/worldgen/20260420_160209}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/recheck_side_alcove_left_visualstable_$(date +%Y%m%d_%H%M%S)}"
TASK_GROUP="${TASK_GROUP:-recheck_side_alcove_left_visualstable}"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/bake.log}"

mkdir -p "${OUT_DIR}"
cp -a "${SRC_WG_DIR}/." "${OUT_DIR}/"
rm -rf "${OUT_DIR}/_baked_goals"

echo "[run] task_group=${TASK_GROUP}"
echo "[run] src_wg_dir=${SRC_WG_DIR}"
echo "[run] out_dir=${OUT_DIR}"
echo "[run] log_path=${LOG_PATH}"

"${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets \
  --task-group "${TASK_GROUP}" \
  --task-group-path "${OUT_DIR}" \
  --protocol "${PROTOCOL}" \
  --tasks "${TASK}" \
  --base-seed "${BASE_SEED}" \
  --episode-retries "${EPISODE_RETRIES}" \
  --save-debug-assets \
  --overwrite \
  --refresh-task-configs 2>&1 | tee "${LOG_PATH}"

SEED_DIR="${OUT_DIR}/_baked_goals/${TASK}/protocol_${PROTOCOL}/seed_$(printf '%06d' "${BASE_SEED}")"
CAND_DIR="${SEED_DIR}/candidate_debug"

echo
echo "[done] out_dir=${OUT_DIR}"
echo "[done] seed_dir=${SEED_DIR}"
echo "[done] log_path=${LOG_PATH}"
echo
echo "[candidate dirs]"
find "${CAND_DIR}" -maxdepth 1 -mindepth 1 -type d | sort | rg '/00[135]_' || true
echo
echo "[key files]"
for d in "${CAND_DIR}"/001_* "${CAND_DIR}"/003_* "${CAND_DIR}"/005_*; do
  [[ -d "${d}" ]] || continue
  echo "--- ${d}"
  for f in goal_image.png goal_mask_overlay.png goal_bbox_overlay.png candidate_pov.png candidate_pov_raw_mask_overlay.png metadata.json; do
    [[ -f "${d}/${f}" ]] && echo "${d}/${f}"
  done
done
