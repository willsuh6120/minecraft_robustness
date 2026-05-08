#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
require_xvfb_run
ensure_minestudio_engine

RUN_ROOT="${RUN_ROOT:-/home/gyulab/minecraft/Minestudio/outputs/evaluate_rocket/p_obstacle_maze_t6_main_local_fresh_20260507_004042}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_uniform}"
TASK="${TASK:-mine_coal}"
EPISODES="${EPISODES:-4}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-0.0}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-0}"
ASSET_DIR="${ASSET_DIR:-}"
BANK_NAME="${BANK_NAME:-}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "missing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi

if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find "${RUN_ROOT}/assets" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${ASSET_DIR}" && -f "${RUN_ROOT}/bank_views/splits/full16/bank_manifest.json" ]]; then
  ASSET_DIR="${RUN_ROOT}/bank_views/splits"
  BANK_NAME="${BANK_NAME:-full16}"
fi
if [[ -z "${BANK_NAME}" ]]; then
  BANK_NAME="eval_bank"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "failed to resolve ASSET_DIR under ${RUN_ROOT}" >&2
  exit 1
fi

MODEL_PATH="$(cat "${RUN_ROOT}/${EXPERIMENT_NAME}/latest_model_path.txt")"
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "invalid latest model path: ${MODEL_PATH}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/${EXPERIMENT_NAME}/final_model_video_probe/full16_x${EPISODES}}"
mkdir -p "${OUT_ROOT}"

echo "[maze-t6-final-video] run_root=${RUN_ROOT}"
echo "[maze-t6-final-video] experiment=${EXPERIMENT_NAME}"
echo "[maze-t6-final-video] model=${MODEL_PATH}"
echo "[maze-t6-final-video] asset_dir=${ASSET_DIR}"
echo "[maze-t6-final-video] bank_name=${BANK_NAME}"
echo "[maze-t6-final-video] out_root=${OUT_ROOT}"

ASSET_DIR="${ASSET_DIR}" \
BANK_NAME="${BANK_NAME}" \
TASK="${TASK}" \
MODEL_PATH="${MODEL_PATH}" \
CFG_COEF="${CFG_COEF}" \
CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
ENV_REWARD_SCALE="${ENV_REWARD_SCALE}" \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
BASE_SEED="${BASE_SEED}" \
SEED_STEP="${SEED_STEP}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
EPISODE_RETRIES="${EPISODE_RETRIES}" \
STEP_BUDGET="${STEP_BUDGET}" \
STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
SKIP_VIDEO="${SKIP_VIDEO}" \
OUT_ROOT="${OUT_ROOT}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/maze_t6_uniform_reward_tools.py" summarize-probe \
  --summary-json "${OUT_ROOT}/summary.json" \
  --label "${EXPERIMENT_NAME} final_model full16_x${EPISODES}" \
  --out-json "${OUT_ROOT}/split_report.json" \
  --out-md "${OUT_ROOT}/split_report.md" >/dev/null

echo "[maze-t6-final-video] done out_root=${OUT_ROOT}"
