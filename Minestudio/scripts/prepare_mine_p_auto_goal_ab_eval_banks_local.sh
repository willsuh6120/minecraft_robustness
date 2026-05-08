#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"
BASE_SEED="${BASE_SEED:-1}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_auto_goal_ab_assets/$(date +%Y%m%d_%H%M%S)}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"

mkdir -p "${OUT_ROOT}"

run_protocol() {
  local label="$1"
  local protocol="$2"
  local out_dir="${OUT_ROOT}/${label}"
  local log_path="${OUT_ROOT}/${label}.log"

  echo
  echo "==================== ${label} protocol=${protocol} ===================="
  echo "[p-auto-goal-ab] out_dir=${out_dir}"
  mkdir -p "${out_dir}"

  set +e
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  BANK_WORKERS="${BANK_WORKERS}" \
  BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES}" \
  BASE_SEED="${BASE_SEED}" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  P_GOAL_POSE_PROTOCOL="${protocol}" \
  OUT_DIR="${out_dir}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh" 2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e

  echo "${status}" > "${OUT_ROOT}/${label}.status"
  if (( status == 0 )); then
    local asset_dir
    asset_dir="$(find "${out_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    echo "${asset_dir}" > "${OUT_ROOT}/${label}.asset_dir"
    echo "[p-auto-goal-ab] ${label} success asset_dir=${asset_dir}"
  else
    echo "[p-auto-goal-ab] ${label} failed status=${status}; log=${log_path}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
  fi
}

cat > "${OUT_ROOT}/README.txt" <<EOF
P obstacle same-world auto-goal A/B eval-bank generation

A front_close:
  Camera hints use the clear target approach cell near local (0, 3).
  More stable, likely close-up.

B legacy_scale:
  Camera hints use the older corridor_center_close/mid geometry near local z ~= 2.
  More comparable target scale to previous O2 auto-goal, but may collide/fail for obstacle variants.

task=${TASK}
eval_instances=${EVAL_INSTANCES}
bank_workers=${BANK_WORKERS}
base_seed=${BASE_SEED}
source_env_conf_dir=${SOURCE_ENV_CONF_DIR}
EOF

run_protocol "A_front_close" "front_close"
run_protocol "B_legacy_scale" "legacy_scale"

echo
echo "==================== A/B done ===================="
echo "[p-auto-goal-ab] out_root=${OUT_ROOT}"
for label in A_front_close B_legacy_scale; do
  printf '%s status=' "${label}"
  cat "${OUT_ROOT}/${label}.status" 2>/dev/null || true
  if [[ -f "${OUT_ROOT}/${label}.asset_dir" ]]; then
    printf '%s asset_dir=' "${label}"
    cat "${OUT_ROOT}/${label}.asset_dir"
  fi
done

failed=0
for label in A_front_close B_legacy_scale; do
  if [[ "$(cat "${OUT_ROOT}/${label}.status" 2>/dev/null || echo 1)" != "0" ]]; then
    failed=1
  fi
done
exit "${failed}"
