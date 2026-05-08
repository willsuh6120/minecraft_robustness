#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUN_ROOT="${RUN_ROOT:-/home/gyulab/minecraft/Minestudio/outputs/evaluate_rocket/p_obstacle_maze_t6_main_local_fresh_20260507_004042}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_uniform}"
EPISODES="${EPISODES:-16}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"

RUN_ROOT="${RUN_ROOT}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
SKIP_VIDEO="${SKIP_VIDEO}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_best_checkpoint_probe.sh"
