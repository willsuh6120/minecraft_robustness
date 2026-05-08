#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REFERENCE_SOURCE_RUN_ROOT="${REFERENCE_SOURCE_RUN_ROOT:-${ROOT_DIR}/../references/maze_t6_vm_rebuild}"

if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  for candidate in \
    "${REFERENCE_SOURCE_RUN_ROOT}" \
    "/home/willsuh1114/minecraft/Minestudio/outputs/evaluate_rocket/p_obstacle_maze_t6_control_vm_rebuild_20260506_183915"; do
    if [[ -d "${candidate}/bank_views" ]]; then
      SOURCE_RUN_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  SOURCE_RUN_ROOT="$(
    {
      find "${ROOT_DIR}/outputs/evaluate_rocket" -maxdepth 1 -type d -name 'p_obstacle_maze_t6_main_local_fresh_*' 2>/dev/null || true
      find "/home/willsuh1114/minecraft/Minestudio/outputs/evaluate_rocket" -maxdepth 1 -type d -name 'p_obstacle_maze_t6_control_vm_rebuild_*' 2>/dev/null || true
      find "${HOME}/minecraft_metrics" -maxdepth 1 -type d -name 'p_obstacle_maze_t6_main_local_fresh_*' 2>/dev/null || true
    } | sort | tail -n 1
  )"
fi

RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_mastery_rr_main_local_$(date +%Y%m%d_%H%M%S)}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
RUN_TAG="${RUN_TAG}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_mastery_rr_full16}" \
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-1.0}" \
COLLECT_MODE="${COLLECT_MODE:-round_robin_full16}" \
TRAIN_ITERS="${TRAIN_ITERS:-24}" \
COLLECT_EPISODES="${COLLECT_EPISODES:-64}" \
COLLECT_WORLD_INSTANCES=1 \
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-1}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_mastery_suite.sh"
