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
EVAL_INSTANCES="${EVAL_INSTANCES:-1}"
BANK_WORKERS="${BANK_WORKERS:-1}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_auto_goal_preview}"

TASK="${TASK}" \
EVAL_INSTANCES="${EVAL_INSTANCES}" \
FINAL_INSTANCES=0 \
SKIP_FINAL_BANK=1 \
BANK_WORKERS="${BANK_WORKERS}" \
OUT_DIR="${OUT_DIR}" \
bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh"

ASSET_DIR="$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
"${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}" "${TASK}"
import json
import shutil
import sys
from pathlib import Path

import yaml

asset_dir = Path(sys.argv[1])
task = str(sys.argv[2])
preview_dir = asset_dir / "goal_preview"
preview_dir.mkdir(parents=True, exist_ok=True)

manifest = json.loads((asset_dir / "eval_bank" / "bank_manifest.json").read_text(encoding="utf-8"))
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", 0))
    variant = str(item.get("path_obstacle_variant_id") or f"instance_{idx:03d}")
    task_group = Path(str(item.get("generated_task_group_dir") or ""))
    data = yaml.safe_load((task_group / f"{task}.yaml").read_text(encoding="utf-8")) or {}
    goal_spec_path = Path(str(data.get("baked_goal_spec_path") or ""))
    if not goal_spec_path.is_file():
        continue
    spec = json.loads(goal_spec_path.read_text(encoding="utf-8"))
    metadata = spec.get("metadata") or {}
    for key in ("goal_image_path", "goal_mask_path", "goal_overlay_path", "raw_image_path"):
        src = Path(str(spec.get(key) or metadata.get(key) or ""))
        if src.is_file():
            dst = preview_dir / f"instance_{idx:03d}_{variant}_{key}{src.suffix}"
            shutil.copy2(src, dst)
    summary = {
        "instance_idx": idx,
        "variant": variant,
        "goal_spec_path": str(goal_spec_path),
        "metadata": {
            "probe_name": metadata.get("probe_name"),
            "pose_family": metadata.get("pose_family"),
            "sampled_camera_position": metadata.get("sampled_camera_position"),
            "sampled_face_distance": metadata.get("sampled_face_distance"),
            "sampled_center_distance": metadata.get("sampled_center_distance"),
            "sampled_pitch": metadata.get("sampled_pitch"),
            "clearance_backoff": metadata.get("clearance_backoff"),
            "bbox_width_px": metadata.get("bbox_width_px"),
            "bbox_height_px": metadata.get("bbox_height_px"),
            "bbox_area_frac": metadata.get("bbox_area_frac"),
            "collision_warning": metadata.get("collision_warning"),
        },
    }
    (preview_dir / f"instance_{idx:03d}_{variant}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

print(f"[preview-p-auto-goal] preview_dir={preview_dir}")
PY
