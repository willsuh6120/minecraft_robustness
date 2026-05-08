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
RUN_TAG="${RUN_TAG:-p_obstacle_hard_v3_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v3_calibration/${RUN_TAG}}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

ASSET_DIR="${ASSET_DIR:-}"
ASSET_ROOT="${ASSET_ROOT:-${MASTER_ROOT}/assets}"
GENERATE_ASSETS="${GENERATE_ASSETS:-1}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"

P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v3}"
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL:-}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-0}"
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL:-center_z3}"

RUN_BASELINE="${RUN_BASELINE:-1}"
BASELINE_EPISODES="${BASELINE_EPISODES:-16}"
BASELINE_WORKERS="${BASELINE_WORKERS:-4}"
RUN_VIDEO="${RUN_VIDEO:-1}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"

STEP_BUDGET="${STEP_BUDGET:-150}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${MASTER_ROOT}"

log() {
  echo "[p-hard-v3-calibration] $*"
}

resolve_latest_asset_dir() {
  find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

generate_assets() {
  if [[ "${GENERATE_ASSETS}" != "1" && -n "${ASSET_DIR}" && -d "${ASSET_DIR}" ]]; then
    log "using existing asset_dir=${ASSET_DIR}"
    return
  fi

  mkdir -p "${ASSET_ROOT}"
  log "generating P-hard-v3 auto-goal bank under ${ASSET_ROOT}"
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  FINAL_INSTANCES=0 \
  SKIP_FINAL_BANK=1 \
  BANK_WORKERS="${BANK_WORKERS}" \
  BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES}" \
  BASE_SEED=1 \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL}" \
  P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
  P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL}" \
  P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT}" \
  OUT_DIR="${ASSET_ROOT}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh"

  ASSET_DIR="$(resolve_latest_asset_dir)"
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "Could not resolve generated ASSET_DIR under ${ASSET_ROOT}" >&2
    exit 1
  fi
  log "asset_dir=${ASSET_DIR}"
}

validate_and_render_geometry() {
  "${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}" "${MASTER_ROOT}/hard_v3_geometry.md"
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
manifest_path = asset_dir / "eval_bank" / "bank_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
worlds = manifest.get("instance_worlds") or []
if len(worlds) != 16:
    raise SystemExit(f"Expected 16 eval worlds, found {len(worlds)}")

reserved = {(0, 3), (0, 4), (0, 5)}
required = {(0, 1), (0, 2)}
bad = []
lines = [
    "# P-Hard-V3 Geometry",
    "",
    "Top-down local grid. `S` spawn, `T` target, `G` reserved centered goal camera / clean sightline cell, `#` obstacle.",
    "",
]
for item in sorted(worlds, key=lambda row: int(row.get("instance_idx", 0))):
    idx = int(item.get("instance_idx", 0))
    variant = str(item.get("path_obstacle_variant_id") or f"instance_{idx:03d}")
    rows = item.get("plan_rows") or []
    suggestions = (rows[0].get("world_generation_suggestions") or {}) if rows else {}
    target = tuple(int(v) for v in suggestions.get("mine_target_local") or [0, 5])
    positions = {tuple(int(v) for v in pos[:2]) for pos in suggestions.get("mine_path_obstacle_positions") or []}
    height = int(suggestions.get("mine_path_obstacle_height") or 1)
    material = str(suggestions.get("mine_path_obstacle_material") or "")
    arena_mode = str(suggestions.get("mine_arena_mode") or "")
    disable_jitter = bool(suggestions.get("mine_disable_spawn_jitter"))
    half_width = int(suggestions.get("mine_arena_half_width") or 5)
    back_z = int(suggestions.get("mine_arena_back_z") or -3)
    front_z = target[1] + int(suggestions.get("mine_arena_front_buffer") or 3)
    if target != (0, 5):
        bad.append((idx, variant, "target_not_0_5", target))
    if not positions & required:
        bad.append((idx, variant, "missing_center_blocker", sorted(positions)))
    if positions & reserved:
        bad.append((idx, variant, "reserved_cell_blocked", sorted(positions & reserved)))
    if height < 2:
        bad.append((idx, variant, "height_lt_2", height))
    if arena_mode != "wide_symmetric":
        bad.append((idx, variant, "arena_not_wide_symmetric", arena_mode))
    if not disable_jitter:
        bad.append((idx, variant, "spawn_jitter_not_disabled", disable_jitter))

    lines += [
        f"## {idx:02d} {variant}",
        "",
        f"positions: `{sorted(positions)}`  ",
        f"height: `{height}`  ",
        f"material: `{material}`  ",
        f"arena: `{arena_mode}`, half_width=`{half_width}`, back_z=`{back_z}`, front_z=`{front_z}`",
        "",
        "```text",
    ]
    wall_x = half_width + 1
    for z in range(front_z, back_z - 1, -1):
        cells = []
        for x in range(-wall_x, wall_x + 1):
            if (x, z) == (0, 0):
                char = "S"
            elif (x, z) == (0, 5):
                char = "T"
            elif (x, z) in {(0, 3), (0, 4)}:
                char = "G"
            elif (x, z) in positions:
                char = "#"
            elif x in {-wall_x, wall_x} or z == back_z:
                char = "W"
            else:
                char = "."
            cells.append(char)
        lines.append(f"z={z}  " + " ".join(cells))
    lines += [
        "     " + " ".join(f"{x:>2}" for x in range(-wall_x, wall_x + 1)),
        "```",
        "",
    ]

if bad:
    raise SystemExit(f"Invalid P-hard-v3 geometry: {bad}")
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"geometry={out_path}")
PY
}

run_baseline() {
  if [[ "${RUN_BASELINE}" != "1" ]]; then
    log "skipping baseline"
    return
  fi

  local baseline_out="${MASTER_ROOT}/baseline_rocket2_${BASELINE_EPISODES}ep"
  log "running baseline: 16 worlds x ${BASELINE_EPISODES} episodes"
  ASSET_DIR="${ASSET_DIR}" \
  TASK="${TASK}" \
  BANK_NAME=eval_bank \
  EPISODES="${BASELINE_EPISODES}" \
  WORLD_WORKERS="${BASELINE_WORKERS}" \
  MODEL_PATH="${MODEL_PATH}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  GOAL_SPEC="" \
  BASE_SEED=1 \
  SAMPLING_BASE_SEED=1 \
  EPISODE_RETRIES=2 \
  STEP_BUDGET="${STEP_BUDGET}" \
  STOP_ON_SUCCESS=1 \
  SKIP_VIDEO=1 \
  OUT_ROOT="${baseline_out}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"
}

run_video_probe() {
  if [[ "${RUN_VIDEO}" != "1" ]]; then
    log "skipping video probe"
    return
  fi

  local video_out="${MASTER_ROOT}/video_rocket2_${VIDEO_EPISODES}ep"
  log "running video probe: 16 worlds x ${VIDEO_EPISODES} episodes"
  ASSET_DIR="${ASSET_DIR}" \
  TASK="${TASK}" \
  BANK_NAME=eval_bank \
  EPISODES="${VIDEO_EPISODES}" \
  WORLD_WORKERS="${VIDEO_WORKERS}" \
  MODEL_PATH="${MODEL_PATH}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  GOAL_SPEC="" \
  BASE_SEED=1001 \
  SAMPLING_BASE_SEED=1001 \
  EPISODE_RETRIES=2 \
  STEP_BUDGET="${STEP_BUDGET}" \
  STOP_ON_SUCCESS=1 \
  SKIP_VIDEO=0 \
  OUT_ROOT="${video_out}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

  find "${video_out}" -type f -name "*_annotated.mp4" | sort > "${video_out}/annotated_video_manifest.txt"
  find "${video_out}" -type f -name "*.mp4" | sort > "${video_out}/video_manifest.txt"
}

write_summary() {
  "${PYTHON_BIN}" - <<'PY' "${MASTER_ROOT}" "${BASELINE_EPISODES}" "${VIDEO_EPISODES}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = str(sys.argv[2])
video_episodes = str(sys.argv[3])
summary_path = root / f"baseline_rocket2_{episodes}ep" / "summary.json"
lines = [
    "# P-Hard-V3 Calibration Summary",
    "",
    f"root: `{root}`",
    f"asset_dir: `{next((root / 'assets').glob('*'), '')}`",
    f"geometry: `{root / 'hard_v3_geometry.md'}`",
    "",
]
if summary_path.exists():
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = sorted(data.get("variants") or [], key=lambda row: int(row.get("instance_idx", 0)))
    total_eps = sum(int(row.get("episodes") or 0) for row in rows)
    total_succ = sum(int(row.get("successful_episodes") or 0) for row in rows)
    rate = total_succ / total_eps if total_eps else 0.0
    lines += [
        "## Baseline",
        "",
        f"overall: `{total_succ}/{total_eps}` ({100.0 * rate:.1f}%)",
        "",
        "| idx | variant | success | mean steps | height | material | positions |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        eps = int(row.get("episodes") or 0)
        succ = int(row.get("successful_episodes") or 0)
        sr = float(row.get("success_rate") or 0.0)
        mean_steps = row.get("mean_steps")
        mean_text = "" if mean_steps is None else f"{float(mean_steps):.1f}"
        lines.append(
            f"| {int(row.get('instance_idx', 0))} | {row.get('variant_id', '')} | "
            f"{succ}/{eps} ({100.0 * sr:.1f}%) | {mean_text} | "
            f"{row.get('height') or ''} | {row.get('material') or ''} | `{row.get('positions', [])}` |"
        )
else:
    lines.append("baseline summary missing")

video_manifest = root / f"video_rocket2_{video_episodes}ep" / "annotated_video_manifest.txt"
if video_manifest.exists():
    lines += [
        "",
        "## Video",
        "",
        f"annotated manifest: `{video_manifest}`",
    ]

out_path = root / "calibration_summary.md"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path)
PY
}

cat > "${MASTER_ROOT}/README.txt" <<EOF
P-hard-v3 calibration

task=${TASK}
master_root=${MASTER_ROOT}
variant_set=${P_OBSTACLE_VARIANT_SET}
material_override=${P_OBSTACLE_MATERIAL:-variant_default}
height_override=${P_OBSTACLE_HEIGHT}
goal_pose_protocol=${P_GOAL_POSE_PROTOCOL}
baseline_episodes=${BASELINE_EPISODES}
baseline_workers=${BASELINE_WORKERS}
run_video=${RUN_VIDEO}
video_episodes=${VIDEO_EPISODES}
EOF

generate_assets
validate_and_render_geometry
run_video_probe
run_baseline
write_summary

log "done"
log "master_root=${MASTER_ROOT}"
log "summary=${MASTER_ROOT}/calibration_summary.md"
log "geometry=${MASTER_ROOT}/hard_v3_geometry.md"
