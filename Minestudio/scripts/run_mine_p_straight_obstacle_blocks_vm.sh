#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_assets}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing. Generate P obstacle assets first with scripts/prepare_mine_p_straight_obstacle_assets.sh." >&2
  exit 1
fi

if [[ -z "${OUT_ROOT:-}" ]]; then
  OUT_ROOT="${ROOT_DIR}/outputs/evaluate_rocket/ppo_mine_p_straight_obstacle_blocks_vm_$(date +%Y%m%d_%H%M%S)"
fi

TASK_GROUP_NAME="${TASK_GROUP_NAME:-mine_p_straight_obstacle_vm}" \
ASSET_DIR="${ASSET_DIR}" \
OUT_ROOT="${OUT_ROOT}" \
EXPERIMENT_TAG="${EXPERIMENT_TAG:-p_straight_obstacle}" \
COLLECT_VIDEO_MODE="${COLLECT_VIDEO_MODE:-skip}" \
BASELINE_VIDEO_MODE="${BASELINE_VIDEO_MODE:-skip}" \
EVAL_VIDEO_MODE="${EVAL_VIDEO_MODE:-skip}" \
FINAL_EVAL_VIDEO_MODE="${FINAL_EVAL_VIDEO_MODE:-skip}" \
SKIP_VIDEO="${SKIP_VIDEO:-1}" \
bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
