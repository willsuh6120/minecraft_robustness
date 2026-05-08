#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${HOME}/miniconda3/envs/minestudio/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

XVFB_RUN="${XVFB_RUN:-$(command -v xvfb-run || true)}"
XVFB_SCREEN_ARGS="${XVFB_SCREEN_ARGS:--screen 0 1280x1024x24}"

TASK="${TASK:-mine_coal}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
GOAL_BANK_ROOT="${GOAL_BANK_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/${TASK}/${GOAL_PROTOCOL}}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
BASE_SEED="${BASE_SEED:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_DEBUG_ASSETS="${SAVE_DEBUG_ASSETS:-0}"
GOAL_CAMERA_HORIZONTAL_DISTANCE="${GOAL_CAMERA_HORIZONTAL_DISTANCE:-2.25}"

GOAL_SPEC="${GOAL_SPEC:-${GOAL_BANK_ROOT}/goal_spec.json}"

if [[ "${OVERWRITE}" != "1" && -f "${GOAL_SPEC}" && -f "${GOAL_BANK_ROOT}/goal_image.png" && -f "${GOAL_BANK_ROOT}/goal_mask.png" ]]; then
  echo "[ensure-fixed-goal] reuse goal_spec=${GOAL_SPEC}"
  exit 0
fi
if [[ ! -d "${SOURCE_ENV_CONF_DIR}" ]]; then
  echo "SOURCE_ENV_CONF_DIR not found: ${SOURCE_ENV_CONF_DIR}" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

mkdir -p "${GOAL_BANK_ROOT}"
WORK_ROOT="${GOAL_BANK_ROOT}/_source_world"
mkdir -p "${WORK_ROOT}"
PLAN_JSON="${WORK_ROOT}/worldgen_plan.json"
GENERATED_ROOT="${WORK_ROOT}/generated_task_groups"
mkdir -p "${GENERATED_ROOT}"

cat > "${PLAN_JSON}" <<'JSON'
[
  {
    "task_config_name": "mine_coal",
    "task_key": "mine_coal",
    "primary_failure_mode_majority": "fixed_clean_goal_front_close",
    "primary_factors_majority": [],
    "severity_majority": 0,
    "factor_levels": {
      "R": 0,
      "H": 0,
      "O": 0,
      "C": 0,
      "P": 0,
      "A": 0
    },
    "layout_seed": 103183093,
    "template_index": 0,
    "requested_split_label": "fixed_clean_goal",
    "computed_split_label": "fixed_clean_goal",
    "hard_factor_count": 0,
    "requested_layout_case": "straight",
    "trainable_with_rl_majority": true,
    "world_generation_suggestions": {
      "visibility": "keep",
      "distractors": "keep",
      "path_difficulty": "keep",
      "view_difficulty": "keep",
      "mine_blueprint_id": "straight_tunnel",
      "mine_target_local": [0, 4],
      "notes": "Fixed clean front-close goal source world. World factors are neutral; target_local is fixed at R0 distance while the goal camera uses the front-close pose."
    }
  }
]
JSON

echo "[ensure-fixed-goal] generating clean source world"
"${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
  --plan-json "${PLAN_JSON}" \
  --env-conf-dir "${SOURCE_ENV_CONF_DIR}" \
  --mine-layout-backend "${MINE_LAYOUT_BACKEND}" \
  --mine-anchor-mode "${MINE_ANCHOR_MODE}" \
  --out-dir "${GENERATED_ROOT}"

GENERATED_DIR="$(find "${GENERATED_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "${GENERATED_DIR}" || ! -d "${GENERATED_DIR}" ]]; then
  echo "Failed to locate generated task group under ${GENERATED_ROOT}" >&2
  exit 1
fi

echo "[ensure-fixed-goal] forcing corridor_center_close goal pose"
"${PYTHON_BIN}" - <<'PY' "${GENERATED_DIR}" "${TASK}" "${GOAL_CAMERA_HORIZONTAL_DISTANCE}"
import math
import sys
from pathlib import Path

import yaml

generated_dir = Path(sys.argv[1])
task = str(sys.argv[2])
goal_camera_horizontal_distance = float(sys.argv[3])
yaml_path = generated_dir / f"{task}.yaml"
if not yaml_path.exists():
    raise SystemExit(f"Generated YAML not found: {yaml_path}")
data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
hints = data.get("goal_pose_hints") or []
if not isinstance(hints, list):
    hints = []
selected = [hint for hint in hints if isinstance(hint, dict) and str(hint.get("name") or "") == "corridor_center_close"]
if not selected:
    names = [str(hint.get("name") or "") for hint in hints if isinstance(hint, dict)]
    raise SystemExit(f"corridor_center_close goal pose hint not found in {yaml_path}; available={names}")
target_center = data.get("target_world_center")
if not (isinstance(target_center, (list, tuple)) and len(target_center) >= 3):
    raise SystemExit(f"target_world_center missing in {yaml_path}")
procedural_layout = data.get("procedural_layout") if isinstance(data.get("procedural_layout"), dict) else {}
face_label = str(
    data.get("goal_camera_preferred_face")
    or data.get("goal_preferred_face")
    or procedural_layout.get("goal_camera_preferred_face")
    or procedural_layout.get("goal_preferred_face")
    or "-z"
).strip()
face_axes = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+z": (0.0, 1.0),
    "-z": (0.0, -1.0),
}
if face_label not in face_axes:
    raise SystemExit(f"Unsupported goal face label for fixed goal: {face_label}")
spawn_positions = data.get("spawn_positions") or []
spawn_position = (spawn_positions[0] or {}).get("position") if spawn_positions else None
feet_y = float(spawn_position[1]) if isinstance(spawn_position, (list, tuple)) and len(spawn_position) >= 2 else math.floor(float(target_center[1]))
out_x, out_z = face_axes[face_label]
target_x, _, target_z = [float(value) for value in target_center[:3]]
target_y = float(target_center[1])
camera_x = target_x + out_x * goal_camera_horizontal_distance
camera_y = feet_y
camera_z = target_z + out_z * goal_camera_horizontal_distance
dx = target_x - camera_x
dy = target_y - (camera_y + 1.62)
dz = target_z - camera_z
horizontal = max(1e-6, math.sqrt(dx * dx + dz * dz))
yaw = math.degrees(math.atan2(-dx, dz))
pitch = math.degrees(math.atan2(-dy, horizontal))
forced_hint = dict(selected[0])
forced_hint.update(
    {
        "name": "corridor_center_close",
        "position": [camera_x, camera_y, camera_z],
        "yaw": float(yaw),
        "pitch": float(pitch),
        "height_offset": 0.0,
        "pose_family": "corridor_center",
        "selection_bias": 10000.0,
        "distance": goal_camera_horizontal_distance,
        "fixed_goal_camera_horizontal_distance": goal_camera_horizontal_distance,
        "fixed_goal_face_label": face_label,
    }
)
data["goal_pose_hints"] = [forced_hint]
data["goal_pose_candidate_mode"] = "hints_only"
data["auto_goal_pose_candidate_mode"] = "hints_only"
data["fixed_goal_protocol_pose_name"] = "corridor_center_close"
data["fixed_goal_camera_horizontal_distance"] = goal_camera_horizontal_distance
yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(
    "[ensure-fixed-goal] forced pose corridor_center_close "
    f"distance={goal_camera_horizontal_distance} face={face_label} in {yaml_path}"
)
PY

BAKE_CMD=(
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets
  --task-group "fixed_goal_${TASK}_${GOAL_PROTOCOL}"
  --task-group-path "${GENERATED_DIR}"
  --protocol ours_v1
  --tasks "${TASK}"
  --base-seed "${BASE_SEED}"
  --episode-retries "${EPISODE_RETRIES}"
  --overwrite
)
if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
  BAKE_CMD+=(--save-debug-assets)
fi

echo "[ensure-fixed-goal] baking goal from ${GENERATED_DIR}"
if [[ -n "${XVFB_RUN}" ]]; then
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${BAKE_CMD[@]}"
else
  "${BAKE_CMD[@]}"
fi

BAKED_DIR="${GENERATED_DIR}/_baked_goals/${TASK}/protocol_ours_v1/seed_$(printf '%06d' "${BASE_SEED}")"
BAKED_SPEC="${BAKED_DIR}/goal_spec.json"
if [[ ! -f "${BAKED_SPEC}" ]]; then
  echo "Baked goal_spec not found: ${BAKED_SPEC}" >&2
  exit 1
fi

echo "[ensure-fixed-goal] normalizing goal bank at ${GOAL_BANK_ROOT}"
"${PYTHON_BIN}" - <<'PY' "${BAKED_SPEC}" "${GOAL_BANK_ROOT}" "${GOAL_PROTOCOL}" "${GENERATED_DIR}" "${PLAN_JSON}" "${GOAL_CAMERA_HORIZONTAL_DISTANCE}"
import hashlib
import json
import shutil
import sys
from pathlib import Path

baked_spec_path = Path(sys.argv[1])
goal_bank_root = Path(sys.argv[2]).resolve()
goal_protocol = str(sys.argv[3])
generated_dir = Path(sys.argv[4]).resolve()
plan_json = Path(sys.argv[5]).resolve()
goal_camera_horizontal_distance = float(sys.argv[6])

spec = json.loads(baked_spec_path.read_text(encoding="utf-8"))
goal_bank_root.mkdir(parents=True, exist_ok=True)

def copy_asset(key: str, filename: str) -> None:
    raw = str(spec.get(key) or "").strip()
    src = Path(raw) if raw else baked_spec_path.parent / filename
    if not src.exists():
        return
    dst = goal_bank_root / filename
    shutil.copy2(src, dst)
    spec[key] = str(dst.resolve())

copy_asset("goal_image_path", "goal_image.png")
copy_asset("goal_mask_path", "goal_mask.png")
copy_asset("goal_mask_overlay_path", "goal_mask_overlay.png")
copy_asset("goal_bbox_overlay_path", "goal_bbox_overlay.png")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

goal_image_path = Path(spec["goal_image_path"])
goal_mask_path = Path(spec["goal_mask_path"])
spec.update(
    {
        "goal_protocol": goal_protocol,
        "goal_protocol_fixed": True,
        "goal_protocol_notes": "Fixed clean front-close goal prompt for world-factor robustness experiments.",
        "goal_source_world": "clean_straight_R0_O0_P0_H0_C0_A0_target_local_0_4",
        "goal_source_factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0},
        "goal_source_target_local": [0, 4],
        "goal_protocol_pose_name": "corridor_center_close",
        "goal_protocol_pose_candidate_mode": "hints_only",
        "goal_protocol_camera_horizontal_distance": goal_camera_horizontal_distance,
        "goal_source_generated_task_group_dir": str(generated_dir),
        "goal_source_plan_json": str(plan_json),
        "goal_image_sha256": sha256_file(goal_image_path),
        "goal_mask_sha256": sha256_file(goal_mask_path),
    }
)
(goal_bank_root / "goal_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"goal_spec": str(goal_bank_root / "goal_spec.json"), "goal_image_sha256": spec["goal_image_sha256"]}, ensure_ascii=False))
PY

echo "[ensure-fixed-goal] done goal_spec=${GOAL_SPEC}"
