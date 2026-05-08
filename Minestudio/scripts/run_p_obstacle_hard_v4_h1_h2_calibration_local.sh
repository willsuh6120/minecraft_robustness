#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cat >&2 <<'EOF'
scripts/run_p_obstacle_hard_v4_h1_h2_calibration_local.sh is deprecated.
Use one of these instead:
  bash scripts/run_p_obstacle_hard_v5_open_path_calibration_local.sh
  bash scripts/run_p_obstacle_hard_v6_maze_t6_calibration_local.sh
EOF
exit 2

TASK="${TASK:-mine_coal}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v4_h1_h2_calibration/$(date +%Y%m%d_%H%M%S)}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
BANK_WORKERS="${BANK_WORKERS:-4}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
BASELINE_EPISODES="${BASELINE_EPISODES:-16}"
BASELINE_WORKERS="${BASELINE_WORKERS:-4}"
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v4_o2_target_funnel}"
HEIGHTS="${HEIGHTS:-1 2}"

mkdir -p "${MASTER_ROOT}"

log() {
  echo "[p-hard-v4-h1-h2] $*"
}

cleanup_leftovers() {
  bash "${ROOT_DIR}/scripts/cleanup_minestudio_leftovers.sh" --kill || true
}

trap 'log "interrupted; cleaning leftovers"; cleanup_leftovers' INT TERM EXIT

log "master_root=${MASTER_ROOT}"
log "heights=${HEIGHTS}"
cleanup_leftovers

for height in ${HEIGHTS}; do
  run_tag="h${height}_$(date +%Y%m%d_%H%M%S)"
  log "starting height=${height} run_tag=${run_tag}"

  TASK="${TASK}" \
  RUN_TAG="${run_tag}" \
  MASTER_ROOT="${MASTER_ROOT}/${run_tag}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  BANK_WORKERS="${BANK_WORKERS}" \
  VIDEO_EPISODES="${VIDEO_EPISODES}" \
  VIDEO_WORKERS="${VIDEO_WORKERS}" \
  BASELINE_EPISODES="${BASELINE_EPISODES}" \
  BASELINE_WORKERS="${BASELINE_WORKERS}" \
  P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
  P_OBSTACLE_HEIGHT="${height}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_hard_v4_calibration_local.sh"

  log "finished height=${height}; cleaning leftovers"
  cleanup_leftovers
done

log "done master_root=${MASTER_ROOT}"
find "${MASTER_ROOT}" -maxdepth 2 -name calibration_summary.md -print | sort
