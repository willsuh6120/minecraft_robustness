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

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
FINAL_INSTANCES="${FINAL_INSTANCES:-0}"
SKIP_FINAL_BANK="${SKIP_FINAL_BANK:-1}"
BASE_SEED="${BASE_SEED:-1}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL:-center_z3}"
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-center_z3}"
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL:-}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-0}"
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE:-0.25}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_auto_goal_assets}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

echo "[prepare-p-auto-goal] out_dir=${OUT_DIR}"
echo "[prepare-p-auto-goal] eval_instances=${EVAL_INSTANCES}"
echo "[prepare-p-auto-goal] bank_workers=${BANK_WORKERS}"
echo "[prepare-p-auto-goal] goal_bake_world_mode=same"
echo "[prepare-p-auto-goal] p_goal_pose_protocol=${P_GOAL_POSE_PROTOCOL}"
echo "[prepare-p-auto-goal] p_obstacle_variant_set=${P_OBSTACLE_VARIANT_SET}"
echo "[prepare-p-auto-goal] p_obstacle_material=${P_OBSTACLE_MATERIAL:-variant_default}"
echo "[prepare-p-auto-goal] p_obstacle_height=${P_OBSTACLE_HEIGHT}"
echo "[prepare-p-auto-goal] path_progress_reward_per_zone=${PATH_PROGRESS_REWARD_PER_ZONE}"

TASK="${TASK}" \
EVAL_INSTANCES="${EVAL_INSTANCES}" \
FINAL_INSTANCES="${FINAL_INSTANCES}" \
SKIP_FINAL_BANK="${SKIP_FINAL_BANK}" \
BASE_SEED="${BASE_SEED}" \
BANK_WORKERS="${BANK_WORKERS}" \
BAKE_GOALS=1 \
GOAL_BAKE_WORLD_MODE=same \
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED}" \
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES}" \
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL}" \
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL}" \
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT}" \
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE}" \
OUT_DIR="${OUT_DIR}" \
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
bash "${ROOT_DIR}/scripts/prepare_mine_p_straight_obstacle_assets.sh"

ASSET_DIR="$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "Could not locate generated asset dir under ${OUT_DIR}" >&2
  exit 1
fi

echo "[prepare-p-auto-goal] asset_dir=${ASSET_DIR}"

"${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}" "${TASK}"
import json
import sys
from pathlib import Path

import yaml

asset_dir = Path(sys.argv[1])
task_name = str(sys.argv[2])
manifest_path = asset_dir / "eval_bank" / "bank_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

rows = []
failed = []
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(rows)))
    variant = str(item.get("path_obstacle_variant_id") or f"instance_{idx:03d}")
    positions = []
    height = None
    material = None
    plan_rows = item.get("plan_rows") or []
    if plan_rows:
        suggestions = plan_rows[0].get("world_generation_suggestions") or {}
        positions = suggestions.get("mine_path_obstacle_positions") or []
        height = suggestions.get("mine_path_obstacle_height")
        material = suggestions.get("mine_path_obstacle_material")
    task_group = Path(str(item.get("generated_task_group_dir") or ""))
    yaml_path = task_group / f"{task_name}.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    goal_spec_path = Path(str(data.get("baked_goal_spec_path") or ""))
    ok = goal_spec_path.is_file()
    metadata = {}
    if ok:
        goal_spec = json.loads(goal_spec_path.read_text(encoding="utf-8"))
        metadata = goal_spec.get("metadata") or {}
    row = {
        "instance_idx": idx,
        "variant_id": variant,
        "positions": positions,
        "height": height,
        "material": material,
        "yaml": str(yaml_path),
        "goal_spec": str(goal_spec_path),
        "goal_ok": ok,
        "probe_name": metadata.get("probe_name"),
        "pose_family": metadata.get("pose_family"),
        "sampled_face_distance": metadata.get("sampled_face_distance"),
        "clearance_backoff": metadata.get("clearance_backoff"),
        "collision_warning": metadata.get("collision_warning"),
        "bbox_area_frac": metadata.get("bbox_area_frac"),
    }
    rows.append(row)
    if not ok:
        failed.append(row)

audit_path = asset_dir / "p_auto_goal_audit.json"
audit_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print("| idx | variant | height | material | ok | probe | face_dist | backoff | collision | bbox_area |")
print("|---:|---|---:|---|---|---|---:|---:|---|---:|")
for row in rows:
    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    print(
        f"| {row['instance_idx']} | {row['variant_id']} | {fmt(row.get('height'))} | "
        f"{row.get('material') or ''} | {row.get('goal_ok')} | "
        f"{row.get('probe_name') or ''} | {fmt(row.get('sampled_face_distance'))} | "
        f"{fmt(row.get('clearance_backoff'))} | {fmt(row.get('collision_warning'))} | "
        f"{fmt(row.get('bbox_area_frac'))} |"
    )
print(f"[prepare-p-auto-goal] audit={audit_path}")
if failed:
    raise SystemExit(f"{len(failed)} eval_bank instances are missing baked goal specs.")
PY

echo "[prepare-p-auto-goal] done asset_dir=${ASSET_DIR}"
