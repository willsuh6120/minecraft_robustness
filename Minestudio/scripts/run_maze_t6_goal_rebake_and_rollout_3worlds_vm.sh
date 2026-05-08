#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"
setup_minestudio_runtime
require_xvfb_run

find_latest_vm_asset_dir() {
  find "${ROOT_DIR}/outputs/evaluate_rocket" \
    -maxdepth 3 \
    -type d \
    -path "*/p_obstacle_maze_t6_uniform_reward_vm_*/assets/*" \
    | sort \
    | tail -n 1
}

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find_latest_vm_asset_dir)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}/eval_bank" ]]; then
  echo "Missing maze_t6 VM asset bank. Set ASSET_DIR to an existing assets/<timestamp> dir." >&2
  exit 1
fi

TASK="${TASK:-mine_coal}"
PROTOCOL="${PROTOCOL:-ours_v1}"
BASE_SEED="${BASE_SEED:-1}"
EPISODES="${EPISODES:-1}"
STEP_BUDGET="${STEP_BUDGET:-150}"
EPISODE_RETRIES="${EPISODE_RETRIES:-1}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-0.0}"
RUN_TAG="${RUN_TAG:-maze_t6_spawn_debug_3worlds_vm_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
TASK_GROUP_WORK_ROOT="${TASK_GROUP_WORK_ROOT:-${OUT_ROOT}/task_groups}"
WORLD_NAMES_CSV="${WORLD_NAMES_CSV:-left_lane_z1_t6,right_chicane_t6,right_outer_detour_t6}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${TASK_GROUP_WORK_ROOT}"
SELECTED_TSV="${OUT_ROOT}/selected_worlds.tsv"

"${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}" "${WORLD_NAMES_CSV}" "${SELECTED_TSV}"
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1]).resolve()
world_names = [item.strip() for item in str(sys.argv[2]).split(",") if item.strip()]
selected_tsv = Path(sys.argv[3]).resolve()

manifest = json.loads((asset_dir / "eval_bank" / "bank_manifest.json").read_text(encoding="utf-8"))
rows = []
for item in manifest.get("instance_worlds") or []:
    variant = str(item.get("path_obstacle_variant_id") or "").strip()
    task_group = str(item.get("generated_task_group_dir") or item.get("task_group_path") or "").strip()
    if not variant or not task_group:
        continue
    rows.append((variant, task_group))

mapping = {variant: task_group for variant, task_group in rows}
missing = [name for name in world_names if name not in mapping]
if missing:
    raise SystemExit(f"Missing world(s) in bank manifest: {missing}")

selected_tsv.write_text(
    "".join(f"{name}\t{mapping[name]}\n" for name in world_names),
    encoding="utf-8",
)
print(f"asset_dir={asset_dir}")
print(f"selected={len(world_names)}")
for name in world_names:
    print(f"{name}\t{mapping[name]}")
PY

echo "[maze-3world-debug] asset_dir=${ASSET_DIR}"
echo "[maze-3world-debug] out_root=${OUT_ROOT}"
echo "[maze-3world-debug] task_group_work_root=${TASK_GROUP_WORK_ROOT}"
echo "[maze-3world-debug] worlds=${WORLD_NAMES_CSV}"
echo "[maze-3world-debug] model=${MODEL_PATH}"

while IFS=$'\t' read -r world_name task_group_path; do
  [[ -n "${world_name}" ]] || continue
  work_task_group_path="${TASK_GROUP_WORK_ROOT}/${world_name}/$(basename "${task_group_path}")"
  echo "[maze-3world-debug] rebake goal ${world_name}"
  TASK_GROUP_PATH="${task_group_path}" \
  WORK_TASK_GROUP_PATH="${work_task_group_path}" \
  TASK="${TASK}" \
  BASE_SEED="${BASE_SEED}" \
  PROTOCOL="${PROTOCOL}" \
  bash "${ROOT_DIR}/scripts/rebake_goal_from_task_group_vm.sh"

  echo "[maze-3world-debug] rollout ${world_name}"
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" \
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
    --env-source rocket2_official \
    --protocol "${PROTOCOL}" \
    --task-group "maze_t6_debug_${world_name}" \
    --task-group-path "${work_task_group_path}" \
    --tasks "${TASK}" \
    --episodes-per-task "${EPISODES}" \
    --base-seed "${BASE_SEED}" \
    --seed-step 1 \
    --sampling-base-seed "${BASE_SEED}" \
    --sampling-seed-step 1 \
    --episode-retries "${EPISODE_RETRIES}" \
    --step-budget-override "${STEP_BUDGET}" \
    --model-path "${MODEL_PATH}" \
    --out-dir "${OUT_ROOT}/${world_name}" \
    --cfg-coef "${CFG_COEF}" \
    --env-reward-scale "${ENV_REWARD_SCALE}" \
    --auto-goal
done < "${SELECTED_TSV}"

echo "[maze-3world-debug] done"
echo "[maze-3world-debug] selected_tsv=${SELECTED_TSV}"
echo "[maze-3world-debug] out_root=${OUT_ROOT}"
