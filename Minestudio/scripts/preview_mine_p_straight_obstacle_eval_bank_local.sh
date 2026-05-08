#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-8}"
EPISODES="${EPISODES:-2}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"

ASSET_DIR="${ASSET_DIR:-}"
ASSET_OUT_DIR="${ASSET_OUT_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_preview_assets}"
PROBE_OUT_ROOT="${PROBE_OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_eval_bank_video_preview/$(date +%Y%m%d_%H%M%S)}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
DEFAULT_GOAL_SPEC="${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/${TASK}/${GOAL_PROTOCOL}/goal_spec.json"
GOAL_SPEC="${GOAL_SPEC:-}"
GOAL_OVERWRITE="${GOAL_OVERWRITE:-0}"
if [[ -z "${GOAL_SPEC}" ]]; then
  GOAL_SPEC="${DEFAULT_GOAL_SPEC}"
fi
if [[ "${GOAL_OVERWRITE}" == "1" || ! -f "${GOAL_SPEC}" ]]; then
  echo "[p-obstacle-preview] ensuring fixed goal bank goal_spec=${GOAL_SPEC} overwrite=${GOAL_OVERWRITE}"
  TASK="${TASK}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_BANK_ROOT="$(dirname "${GOAL_SPEC}")" \
  OVERWRITE="${GOAL_OVERWRITE}" \
  bash "${ROOT_DIR}/scripts/ensure_fixed_clean_goal_bank.sh"
fi
BAKE_GOALS="${BAKE_GOALS:-}"
if [[ -z "${BAKE_GOALS}" ]]; then
  if [[ -n "${GOAL_SPEC}" ]]; then
    BAKE_GOALS=0
  else
    BAKE_GOALS=1
  fi
fi
GOAL_BAKE_WORLD_MODE="${GOAL_BAKE_WORLD_MODE:-clean_path}"

if [[ -z "${ASSET_DIR}" ]]; then
  echo "[p-obstacle-preview] generating eval bank assets"
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  FINAL_INSTANCES=0 \
  SKIP_FINAL_BANK=1 \
  BAKE_GOALS="${BAKE_GOALS}" \
  GOAL_BAKE_WORLD_MODE="${GOAL_BAKE_WORLD_MODE}" \
  BASE_SEED="${BASE_SEED}" \
  OUT_DIR="${ASSET_OUT_DIR}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_straight_obstacle_assets.sh"

  ASSET_DIR="$(find "${ASSET_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi

echo "[p-obstacle-preview] asset_dir=${ASSET_DIR}"
echo "[p-obstacle-preview] episodes_per_world=${EPISODES}"
echo "[p-obstacle-preview] world_workers=${WORLD_WORKERS}"
echo "[p-obstacle-preview] video=enabled"
echo "[p-obstacle-preview] bake_goals=${BAKE_GOALS}"
echo "[p-obstacle-preview] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"
echo "[p-obstacle-preview] out_root=${PROBE_OUT_ROOT}"

ASSET_DIR="${ASSET_DIR}" \
BANK_NAME=eval_bank \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
BASE_SEED="${BASE_SEED}" \
SEED_STEP="${SEED_STEP}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
EPISODE_RETRIES="${EPISODE_RETRIES}" \
STEP_BUDGET="${STEP_BUDGET}" \
MODEL_PATH="${MODEL_PATH}" \
CFG_COEF="${CFG_COEF}" \
CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
GOAL_SPEC="${GOAL_SPEC}" \
STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
SKIP_VIDEO=0 \
OUT_ROOT="${PROBE_OUT_ROOT}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

find "${PROBE_OUT_ROOT}" -type f -name "*_annotated.mp4" | sort > "${PROBE_OUT_ROOT}/annotated_video_manifest.txt"
find "${PROBE_OUT_ROOT}" -type f -name "*.mp4" | sort > "${PROBE_OUT_ROOT}/video_manifest.txt"

echo "[p-obstacle-preview] summary=${PROBE_OUT_ROOT}/summary.json"
echo "[p-obstacle-preview] annotated_videos=${PROBE_OUT_ROOT}/annotated_video_manifest.txt"
echo "[p-obstacle-preview] videos=${PROBE_OUT_ROOT}/video_manifest.txt"
