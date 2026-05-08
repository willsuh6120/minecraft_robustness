#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK="${TASK:-mine_coal}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
GOAL_SPEC="${GOAL_SPEC:-${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/${TASK}/${GOAL_PROTOCOL}/goal_spec.json}"
GOAL_CAMERA_HORIZONTAL_DISTANCE="${GOAL_CAMERA_HORIZONTAL_DISTANCE:-2.25}"

SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_worker_timing" \
      -path "*/assets/*/eval_bank" -type d 2>/dev/null \
      | sed 's#/eval_bank$##' \
      | sort \
      | tail -n 1 || true
  )"
fi

SUITE_TAG="${SUITE_TAG:-p_obstacle_overnight_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_overnight_learning/${SUITE_TAG}}"

BASE_SEEDS="${BASE_SEEDS:-1}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"

# balanced:
#   stage 1: order ablation with one PPO iteration per obstacle world
#   stage 2: same teacher curriculum with two PPO iterations per obstacle world
# wide:
#   stage 1 plus all_forward and best_single controls
# minimal:
#   only teacher_to_hard and no_update_teacher_to_hard
PLAN="${PLAN:-balanced}"

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid. Set ASSET_DIR to the P-obstacle asset dir." >&2
  exit 1
fi
if [[ ! -d "${ASSET_DIR}/eval_bank" || ! -d "${ASSET_DIR}/collect_block_plans" ]]; then
  echo "ASSET_DIR does not look like a P-obstacle asset bank: ${ASSET_DIR}" >&2
  exit 1
fi

mkdir -p "${MASTER_ROOT}"
cat > "${MASTER_ROOT}/README.txt" <<EOF
P obstacle overnight learning suite

plan=${PLAN}
asset_dir=${ASSET_DIR}
goal_spec=${GOAL_SPEC}
base_seeds=${BASE_SEEDS}
collect_episodes=${COLLECT_EPISODES}
eval_episodes=${EVAL_EPISODES}
collect_workers=${COLLECT_WORKERS}
eval_workers=${EVAL_WORKERS}

Block ids:
1 bar_z2_full
2 side_posts_z3
3 left_gate
4 right_gate
5 left_zigzag
6 right_zigzag
7 left_wall
8 right_wall

teacher_to_hard=4,1,8,5,7,2,3,6
reverse=6,3,2,7,5,8,1,4
all_forward=1,2,3,4,5,6,7,8
best_single=4,4,4,4,4,4,4,4
EOF

run_stage() {
  local stage_name="$1"
  local variants="$2"
  local iters_per_block="$3"

  echo
  echo "==================== ${stage_name} ===================="
  echo "[p-overnight] variants=${variants}"
  echo "[p-overnight] iters_per_block=${iters_per_block}"

  TASK="${TASK}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  GOAL_CAMERA_HORIZONTAL_DISTANCE="${GOAL_CAMERA_HORIZONTAL_DISTANCE}" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  ASSET_DIR="${ASSET_DIR}" \
  GENERATE_ASSETS=0 \
  SUITE_ROOT="${MASTER_ROOT}/${stage_name}" \
  VARIANTS="${variants}" \
  BASE_SEEDS="${BASE_SEEDS}" \
  COLLECT_EPISODES="${COLLECT_EPISODES}" \
  EVAL_EPISODES="${EVAL_EPISODES}" \
  COLLECT_WORKERS="${COLLECT_WORKERS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  ITERS_PER_BLOCK="${iters_per_block}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS}" \
  UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE}" \
  PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
  KL_COEF="${KL_COEF}" \
  SKIP_VIDEO=1 \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_learning_signal_suite_local.sh"
}

case "${PLAN}" in
  minimal)
    run_stage "stage_01_order_1iter" "teacher_to_hard,no_update_teacher_to_hard" 1
    ;;
  balanced)
    run_stage "stage_01_order_1iter" "teacher_to_hard,no_update_teacher_to_hard,reverse" 1
    run_stage "stage_02_teacher_2iter" "teacher_to_hard" 2
    ;;
  wide)
    run_stage "stage_01_order_1iter" "teacher_to_hard,no_update_teacher_to_hard,reverse,all_forward,best_single" 1
    run_stage "stage_02_teacher_2iter" "teacher_to_hard" 2
    ;;
  *)
    echo "Unknown PLAN=${PLAN}; expected minimal, balanced, or wide." >&2
    exit 1
    ;;
esac

echo
echo "==================== overnight done ===================="
echo "[p-overnight] master_root=${MASTER_ROOT}"
find "${MASTER_ROOT}" -name suite_summary.md -print | sort
