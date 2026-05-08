#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cat >&2 <<'EOF'
scripts/run_p_obstacle_hard_v4_resume_h1_video_h2_full_local.sh is deprecated.
Use one of these instead:
  bash scripts/run_p_obstacle_hard_v5_open_path_calibration_local.sh
  bash scripts/run_p_obstacle_hard_v6_maze_t6_calibration_local.sh
EOF
exit 2

TASK="${TASK:-mine_coal}"
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v4_o2_target_funnel}"
BANK_WORKERS="${BANK_WORKERS:-4}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
BASELINE_EPISODES="${BASELINE_EPISODES:-16}"
BASELINE_WORKERS="${BASELINE_WORKERS:-4}"

RUN_FAMILY_ROOT="${RUN_FAMILY_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v4_h1_h2_calibration}"
if [[ -z "${MASTER_ROOT:-}" ]]; then
  MASTER_ROOT="$(find "${RUN_FAMILY_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${MASTER_ROOT:-}" || ! -d "${MASTER_ROOT}" ]]; then
  echo "Could not find MASTER_ROOT under ${RUN_FAMILY_ROOT}; set MASTER_ROOT explicitly." >&2
  exit 1
fi

if [[ -z "${H1_RUN_DIR:-}" ]]; then
  H1_RUN_DIR="$(find "${MASTER_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'h1_*' 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${H1_RUN_DIR:-}" || ! -d "${H1_RUN_DIR}" ]]; then
  echo "Could not find H1_RUN_DIR under ${MASTER_ROOT}; set H1_RUN_DIR explicitly." >&2
  exit 1
fi

if [[ -z "${H1_ASSET_DIR:-}" ]]; then
  H1_ASSET_DIR="$(find "${H1_RUN_DIR}/assets" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${H1_ASSET_DIR:-}" || ! -d "${H1_ASSET_DIR}" ]]; then
  echo "Could not find H1_ASSET_DIR under ${H1_RUN_DIR}/assets; set H1_ASSET_DIR explicitly." >&2
  exit 1
fi

H2_RUN_DIR="${H2_RUN_DIR:-${MASTER_ROOT}/h2_$(date +%Y%m%d_%H%M%S)}"

log() {
  echo "[p-hard-v4-resume] $*"
}

cleanup_leftovers() {
  bash "${ROOT_DIR}/scripts/cleanup_minestudio_leftovers.sh" --kill || true
}

run_calib() {
  local label="$1"
  shift
  log "start ${label}"
  "$@"
  log "done ${label}"
  cleanup_leftovers
}

trap 'log "interrupted; cleaning leftovers"; cleanup_leftovers' INT TERM EXIT

log "master_root=${MASTER_ROOT}"
log "h1_run_dir=${H1_RUN_DIR}"
log "h1_asset_dir=${H1_ASSET_DIR}"
log "h2_run_dir=${H2_RUN_DIR}"
cleanup_leftovers

run_calib "h1 video only" \
  env TASK="${TASK}" \
    MASTER_ROOT="${H1_RUN_DIR}" \
    ASSET_DIR="${H1_ASSET_DIR}" \
    GENERATE_ASSETS=0 \
    RUN_VIDEO=1 \
    RUN_BASELINE=0 \
    VIDEO_EPISODES="${VIDEO_EPISODES}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" \
    BASELINE_EPISODES="${BASELINE_EPISODES}" \
    BASELINE_WORKERS="${BASELINE_WORKERS}" \
    P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
    P_OBSTACLE_HEIGHT=1 \
    bash "${ROOT_DIR}/scripts/run_p_obstacle_hard_v4_calibration_local.sh"

run_calib "h2 bank and video" \
  env TASK="${TASK}" \
    MASTER_ROOT="${H2_RUN_DIR}" \
    GENERATE_ASSETS=1 \
    BANK_WORKERS="${BANK_WORKERS}" \
    RUN_VIDEO=1 \
    RUN_BASELINE=0 \
    VIDEO_EPISODES="${VIDEO_EPISODES}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" \
    BASELINE_EPISODES="${BASELINE_EPISODES}" \
    BASELINE_WORKERS="${BASELINE_WORKERS}" \
    P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
    P_OBSTACLE_HEIGHT=2 \
    bash "${ROOT_DIR}/scripts/run_p_obstacle_hard_v4_calibration_local.sh"

H2_ASSET_DIR="${H2_ASSET_DIR:-$(find "${H2_RUN_DIR}/assets" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)}"
if [[ -z "${H2_ASSET_DIR}" || ! -d "${H2_ASSET_DIR}" ]]; then
  echo "Could not find generated H2_ASSET_DIR under ${H2_RUN_DIR}/assets" >&2
  exit 1
fi

run_calib "h1 baseline" \
  env TASK="${TASK}" \
    MASTER_ROOT="${H1_RUN_DIR}" \
    ASSET_DIR="${H1_ASSET_DIR}" \
    GENERATE_ASSETS=0 \
    RUN_VIDEO=0 \
    RUN_BASELINE=1 \
    VIDEO_EPISODES="${VIDEO_EPISODES}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" \
    BASELINE_EPISODES="${BASELINE_EPISODES}" \
    BASELINE_WORKERS="${BASELINE_WORKERS}" \
    P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
    P_OBSTACLE_HEIGHT=1 \
    bash "${ROOT_DIR}/scripts/run_p_obstacle_hard_v4_calibration_local.sh"

run_calib "h2 baseline" \
  env TASK="${TASK}" \
    MASTER_ROOT="${H2_RUN_DIR}" \
    ASSET_DIR="${H2_ASSET_DIR}" \
    GENERATE_ASSETS=0 \
    RUN_VIDEO=0 \
    RUN_BASELINE=1 \
    VIDEO_EPISODES="${VIDEO_EPISODES}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" \
    BASELINE_EPISODES="${BASELINE_EPISODES}" \
    BASELINE_WORKERS="${BASELINE_WORKERS}" \
    P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
    P_OBSTACLE_HEIGHT=2 \
    bash "${ROOT_DIR}/scripts/run_p_obstacle_hard_v4_calibration_local.sh"

log "done"
log "h1_summary=${H1_RUN_DIR}/calibration_summary.md"
log "h1_video_manifest=${H1_RUN_DIR}/video_rocket2_${VIDEO_EPISODES}ep/annotated_video_manifest.txt"
log "h2_summary=${H2_RUN_DIR}/calibration_summary.md"
log "h2_video_manifest=${H2_RUN_DIR}/video_rocket2_${VIDEO_EPISODES}ep/annotated_video_manifest.txt"
