#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"
setup_minestudio_runtime
require_xvfb_run

TASK_GROUP_PATH="${TASK_GROUP_PATH:-}"
WORK_TASK_GROUP_PATH="${WORK_TASK_GROUP_PATH:-}"
AUTO_WORK_COPY="${AUTO_WORK_COPY:-1}"
DEBUG_COPY_ROOT="${DEBUG_COPY_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/_debug_task_groups/$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-mine_coal}"
PROTOCOL="${PROTOCOL:-ours_v1}"
BASE_SEED="${BASE_SEED:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-1}"
GOAL_SUBDIR="${GOAL_SUBDIR:-_baked_goals}"
SAVE_DEBUG_ASSETS="${SAVE_DEBUG_ASSETS:-1}"
OVERWRITE="${OVERWRITE:-1}"
PATCH_TARGET_CENTER="${PATCH_TARGET_CENTER:-1}"
SKIP_BAKE="${SKIP_BAKE:-0}"

if [[ -z "${TASK_GROUP_PATH}" ]]; then
  echo "TASK_GROUP_PATH is required." >&2
  echo "Example:" >&2
  echo "  TASK_GROUP_PATH=/abs/path/to/generated_task_groups/<id> bash scripts/rebake_goal_from_task_group_local.sh" >&2
  exit 2
fi

if [[ ! -d "${TASK_GROUP_PATH}" ]]; then
  echo "TASK_GROUP_PATH does not exist: ${TASK_GROUP_PATH}" >&2
  exit 2
fi

SOURCE_TASK_GROUP_PATH="${TASK_GROUP_PATH}"
if [[ -z "${WORK_TASK_GROUP_PATH}" && "${AUTO_WORK_COPY}" == "1" ]]; then
  instance_dir="$(basename "$(dirname "$(dirname "${TASK_GROUP_PATH}")")")"
  group_dir="$(basename "${TASK_GROUP_PATH}")"
  WORK_TASK_GROUP_PATH="${DEBUG_COPY_ROOT}/${instance_dir}/${group_dir}"
fi

if [[ -n "${WORK_TASK_GROUP_PATH}" ]]; then
  if [[ ! -d "${WORK_TASK_GROUP_PATH}" ]]; then
    mkdir -p "$(dirname "${WORK_TASK_GROUP_PATH}")"
    cp -a "${TASK_GROUP_PATH}" "${WORK_TASK_GROUP_PATH}"
  fi
  TASK_GROUP_PATH="${WORK_TASK_GROUP_PATH}"
fi

TASK_GROUP="${TASK_GROUP:-$(basename "${TASK_GROUP_PATH}")}"

if [[ "${PATCH_TARGET_CENTER}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${TASK_GROUP_PATH}" "${TASK}"
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, '/home/willsuh1114/minecraft/Minestudio')
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import canonical_task_key
from minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen import (
    _absolute_tp_command,
    build_goal_pose_hints,
)

task_group_path = Path(sys.argv[1])
task_name = str(sys.argv[2])
task_yaml_path = task_group_path / f"{task_name}.yaml"
if not task_yaml_path.exists():
    raise SystemExit(f"task yaml missing: {task_yaml_path}")

data = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8")) or {}
spawn_positions = list(data.get("spawn_positions") or [])
spawn_position = list((spawn_positions[0] or {}).get("position") or []) if spawn_positions else []
procedural_layout = data.get("procedural_layout") if isinstance(data.get("procedural_layout"), dict) else {}
target_local = procedural_layout.get("target_local") if isinstance(procedural_layout.get("target_local"), (list, tuple)) else None
target_type = str(data.get("target_type") or "coal_ore")

if len(spawn_position) < 3 or not (isinstance(target_local, (list, tuple)) and len(target_local) >= 2):
    raise SystemExit("task yaml does not look like a procedural mine task with target_local metadata")

spawn_x, spawn_y, spawn_z = [float(v) for v in spawn_position[:3]]
target_center = [
    round(spawn_x + float(target_local[0]), 4),
    round(spawn_y - 0.5, 4),
    round(spawn_z + float(target_local[1]), 4),
]
goal_camera_look_target_world_center = [
    round(spawn_x + float(target_local[0]), 4),
    round(spawn_y + 0.5, 4),
    round(spawn_z + float(target_local[1]), 4),
]
data["target_world_center"] = list(target_center)
data["target_blocks"] = [{"type": target_type, "world_center": list(target_center)}]
if isinstance(procedural_layout.get("realized_metrics"), dict):
    procedural_layout["realized_metrics"]["target_world_center"] = list(target_center)
    procedural_layout["goal_camera_look_target_world_center"] = list(goal_camera_look_target_world_center)
    procedural_layout["realized_metrics"].pop("build_anchor_position", None)
    procedural_layout.pop("build_anchor_position", None)
    procedural_layout.pop("build_anchor_mode", None)
    data["procedural_layout"] = procedural_layout
else:
    procedural_layout["goal_camera_look_target_world_center"] = list(goal_camera_look_target_world_center)
    data["procedural_layout"] = procedural_layout
if isinstance(data.get("realized_factor_metrics"), dict):
    data["realized_factor_metrics"]["target_world_center"] = list(target_center)
spawn_yaw = float((spawn_positions[0] or {}).get("yaw", 0.0) or 0.0) if spawn_positions else 0.0
spawn_pitch = float((spawn_positions[0] or {}).get("pitch", 0.0) or 0.0) if spawn_positions else 0.0
tp_command = _absolute_tp_command(spawn_x, spawn_y, spawn_z, spawn_yaw, spawn_pitch)
anchor_prefix = (
    f"/execute positioned {float(spawn_x):.3f} {float(spawn_y):.3f} {float(spawn_z):.3f} run "
)
relative_prefix = "/execute as @p at @s run "
commands = []
for raw_command in list(data.get("custom_init_commands") or []):
    stripped = str(raw_command or "").strip()
    if stripped.startswith(anchor_prefix):
        commands.append(relative_prefix + stripped[len(anchor_prefix):])
    else:
        commands.append(str(raw_command))
if not commands or str(commands[0]).strip() != tp_command:
    commands.insert(0, tp_command)
if not commands or str(commands[-1]).strip() != tp_command:
    commands.append(tp_command)
data["custom_init_commands"] = commands
data["spawn_support_pad"] = {"cleanup_after_build": True}
data["goal_pose_hints"] = build_goal_pose_hints(canonical_task_key(task_name), data)
task_yaml_path.write_text(
    yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

manifest_path = task_group_path / "worldgen_manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in list(manifest.get("tasks") or []):
        if str(row.get("task_config_name") or "").strip() != task_name:
            continue
        row["target_world_center"] = list(target_center)
        break
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({
    "task_yaml_path": str(task_yaml_path.resolve()),
    "patched_target_world_center": target_center,
}, ensure_ascii=False))
PY
fi

echo "[rebake-goal] python=${PYTHON_BIN}"
echo "[rebake-goal] minestudio_dir=${MINESTUDIO_DIR}"
echo "[rebake-goal] source_task_group_path=${SOURCE_TASK_GROUP_PATH}"
echo "[rebake-goal] task_group_path=${TASK_GROUP_PATH}"
echo "[rebake-goal] auto_work_copy=${AUTO_WORK_COPY}"
echo "[rebake-goal] task=${TASK} protocol=${PROTOCOL} base_seed=${BASE_SEED}"

if [[ "${SKIP_BAKE}" == "1" ]]; then
  echo "[rebake-goal] skip_bake=1 patched task yaml only."
  exit 0
fi

cmd=(
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}"
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets
  --task-group "${TASK_GROUP}"
  --task-group-path "${TASK_GROUP_PATH}"
  --protocol "${PROTOCOL}"
  --tasks "${TASK}"
  --base-seed "${BASE_SEED}"
  --episode-retries "${EPISODE_RETRIES}"
  --goal-subdir "${GOAL_SUBDIR}"
)

if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
  cmd+=( --save-debug-assets )
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=( --overwrite )
fi

"${cmd[@]}"

goal_dir="${TASK_GROUP_PATH}/${GOAL_SUBDIR}/${TASK}/protocol_${PROTOCOL}/seed_$(printf '%06d' "${BASE_SEED}")"
echo "[rebake-goal] goal_dir=${goal_dir}"
echo "[rebake-goal] goal_image=${goal_dir}/goal_image.png"
echo "[rebake-goal] goal_mask_overlay=${goal_dir}/goal_mask_overlay.png"
echo "[rebake-goal] goal_spec=${goal_dir}/goal_spec.json"
