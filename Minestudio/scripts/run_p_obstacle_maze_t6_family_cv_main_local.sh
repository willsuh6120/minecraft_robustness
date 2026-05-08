#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  SOURCE_RUN_ROOT="$(
    {
      find "${ROOT_DIR}/outputs/evaluate_rocket" -maxdepth 1 -type d -name 'p_obstacle_maze_t6_main_local_fresh_*' 2>/dev/null || true
      find "${HOME}/minecraft_metrics" -maxdepth 1 -type d -name 'p_obstacle_maze_t6_main_local_fresh_*' 2>/dev/null || true
    } | sort | tail -n 1
  )"
fi

FOLD_NAME="${FOLD_NAME:-fold_a}"
RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_family_cv_main_local_${FOLD_NAME}_$(date +%Y%m%d_%H%M%S)}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
FOLD_NAME="${FOLD_NAME}" \
RUN_TAG="${RUN_TAG}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_cv_${FOLD_NAME}}" \
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-1.0}" \
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-1}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_family_cv_suite.sh"
