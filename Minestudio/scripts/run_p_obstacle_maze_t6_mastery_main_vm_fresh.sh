#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
ensure_minestudio_engine

RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_mastery_main_vm_fresh_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
ASSET_ROOT="${ASSET_ROOT:-${OUT_ROOT}/assets}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${WORKSPACE_ROOT}/ROCKET-2/env_conf}"

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
FINAL_INSTANCES="${FINAL_INSTANCES:-0}"
SKIP_FINAL_BANK="${SKIP_FINAL_BANK:-1}"
BASE_SEED="${BASE_SEED:-1}"
STEP_BUDGET="${STEP_BUDGET:-150}"
BANK_WORKERS="${BANK_WORKERS:-4}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL:-center_z3}"
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v6_maze_t6}"
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL:-}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-0}"
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE:-0.125}"

TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
TRAIN_ITERS="${TRAIN_ITERS:-24}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-1}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
VF_WARMUP_ITERS="${VF_WARMUP_ITERS:-1}"
ZERO_INITIAL_VF="${ZERO_INITIAL_VF:-0}"
CALIBRATE_VALUE_NORMALIZER="${CALIBRATE_VALUE_NORMALIZER:-1}"

find_latest_subdir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

if [[ ! -d "${SOURCE_ENV_CONF_DIR}" ]]; then
  echo "SOURCE_ENV_CONF_DIR not found: ${SOURCE_ENV_CONF_DIR}" >&2
  echo "Set SOURCE_ENV_CONF_DIR explicitly to a valid env_conf directory before running." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${ASSET_ROOT}"

echo "[maze-t6-mastery-vm-fresh] run_tag=${RUN_TAG}"
echo "[maze-t6-mastery-vm-fresh] out_root=${OUT_ROOT}"
echo "[maze-t6-mastery-vm-fresh] asset_root=${ASSET_ROOT}"
echo "[maze-t6-mastery-vm-fresh] source_env_conf_dir=${SOURCE_ENV_CONF_DIR}"
echo "[maze-t6-mastery-vm-fresh] variant_set=${P_OBSTACLE_VARIANT_SET}"
echo "[maze-t6-mastery-vm-fresh] step_budget=${STEP_BUDGET}"

TASK="${TASK}" \
EVAL_INSTANCES="${EVAL_INSTANCES}" \
FINAL_INSTANCES="${FINAL_INSTANCES}" \
SKIP_FINAL_BANK="${SKIP_FINAL_BANK}" \
BASE_SEED="${BASE_SEED}" \
BANK_WORKERS="${BANK_WORKERS}" \
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED}" \
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES}" \
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL}" \
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL}" \
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT}" \
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE}" \
OUT_DIR="${ASSET_ROOT}" \
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh"

ASSET_DIR="$(find_latest_subdir "${ASSET_ROOT}")"
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}/eval_bank" ]]; then
  echo "failed to resolve freshly generated ASSET_DIR under ${ASSET_ROOT}" >&2
  exit 1
fi

echo "[maze-t6-mastery-vm-fresh] asset_dir=${ASSET_DIR}"

ASSET_DIR="${ASSET_DIR}" \
RUN_TAG="${RUN_TAG}" \
OUT_ROOT="${OUT_ROOT}" \
TRAINABLE_SCOPE="${TRAINABLE_SCOPE}" \
TRAIN_ITERS="${TRAIN_ITERS}" \
COLLECT_EPISODES="${COLLECT_EPISODES}" \
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE}" \
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL}" \
STEP_BUDGET="${STEP_BUDGET}" \
VF_WARMUP_ITERS="${VF_WARMUP_ITERS}" \
ZERO_INITIAL_VF="${ZERO_INITIAL_VF}" \
CALIBRATE_VALUE_NORMALIZER="${CALIBRATE_VALUE_NORMALIZER}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_mastery_main_vm.sh"
