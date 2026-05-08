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
RUN_TAG="${RUN_TAG:-p_obstacle_hard_v4_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v4_calibration/${RUN_TAG}}"
GEOMETRY_MD="${GEOMETRY_MD:-${MASTER_ROOT}/p_obstacle_geometry.md}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

ASSET_DIR="${ASSET_DIR:-}"
ASSET_ROOT="${ASSET_ROOT:-${MASTER_ROOT}/assets}"
GENERATE_ASSETS="${GENERATE_ASSETS:-1}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"

P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v4_o2_target_funnel}"
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL:-}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-0}"
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL:-center_z3}"

RUN_VIDEO="${RUN_VIDEO:-1}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
BASELINE_EPISODES="${BASELINE_EPISODES:-16}"
BASELINE_WORKERS="${BASELINE_WORKERS:-4}"

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
  echo "[p-hard-v4-calibration] $*"
}

ensure_supported_variant_set() {
  local normalized
  normalized="$(printf '%s' "${P_OBSTACLE_VARIANT_SET}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    hard_v4|hard_v4_o2_target_funnel|o2_target_funnel)
      cat >&2 <<'EOF'
Deprecated P obstacle bank: hard_v4.
Use one of these instead:
  bash scripts/run_p_obstacle_hard_v5_open_path_calibration_local.sh
  bash scripts/run_p_obstacle_hard_v6_maze_t6_calibration_local.sh
EOF
      exit 2
      ;;
  esac
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
  log "generating P-hard-v4 auto-goal bank under ${ASSET_ROOT}"
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
  "${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}" "${GEOMETRY_MD}"
import json
import sys
from collections import deque
from pathlib import Path

from minestudio.tutorials.inference.evaluate_rocket.path_progress_reward import (
    build_path_progress_reward_config,
    build_zone_label_lookup,
)

asset_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
manifest_path = asset_dir / "eval_bank" / "bank_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
worlds = manifest.get("instance_worlds") or []
if len(worlds) != 16:
    raise SystemExit(f"Expected 16 eval worlds, found {len(worlds)}")

bad = []
lines = [
    "# P Obstacle O2 Target Funnel Geometry",
    "",
    "Top-down local grid. `S` spawn, `T` target, `G` reserved goal-camera/approach center cell, `#` obstacle, `W` stone/wall.",
    "Path-shaped reward zones are shown as `1`, `2`, `3`, and `A` when they lie on a valid simple path to the funnel.",
    "",
    "Validation requires a connected open path from `S=(0,0)` to the funnel entrance row.",
    "The target-adjacent row is reserved for O-factor composition and must not contain P obstacles.",
    "",
    "Target cross-section at `z=5` after O2-style action openings:",
    "",
    "```text",
    "y=3  W W W",
    "y=2  W W W",
    "y=1  W . W",
    "y=0  . T .",
    "     x=-1 0 +1",
    "```",
    "",
]

def is_wall_cell(x, z, *, target_z, half_width, back_z, front_z, funnel_start_z):
    wall_x = half_width + 1
    if x in {-wall_x, wall_x} or z == back_z:
        return True
    if z >= funnel_start_z:
        if z > target_z:
            return True
        if z == target_z:
            return x not in {-1, 0, 1}
        return abs(x) > 1
    return False


def has_open_path_to_funnel(positions, *, target_z, half_width, back_z, front_z, funnel_start_z):
    wall_x = half_width + 1
    start = (0, 0)
    goals = {(-1, funnel_start_z), (0, funnel_start_z), (1, funnel_start_z)}
    blocked = set(positions)
    if start in blocked:
        return False, []
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        if cell in goals:
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return True, path
        x, z = cell
        for nxt in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            nx, nz = nxt
            if nx < -wall_x or nx > wall_x or nz < back_z or nz > front_z:
                continue
            if nxt in blocked or is_wall_cell(
                nx,
                nz,
                target_z=target_z,
                half_width=half_width,
                back_z=back_z,
                front_z=front_z,
                funnel_start_z=funnel_start_z,
            ):
                continue
            if nxt in parent:
                continue
            parent[nxt] = cell
            q.append(nxt)
    return False, []


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
    half_width = int(suggestions.get("mine_arena_half_width") or 4)
    back_z = int(suggestions.get("mine_arena_back_z") or -3)
    front_buffer = int(suggestions.get("mine_arena_front_buffer") or 3)
    front_z = target[1] + front_buffer
    funnel_start_z = int(suggestions.get("mine_arena_funnel_start_z") or 3)
    reward_config = build_path_progress_reward_config(
        target_local=target,
        obstacle_positions=sorted(positions),
        arena_settings={
            "funnel_start_z": funnel_start_z,
            "half_width": half_width,
            "back_z": back_z,
        },
    )
    reward_labels = build_zone_label_lookup(reward_config)

    target_x, target_z = target
    center_blocker_candidates = {(0, z) for z in range(1, max(2, int(target_z) - 2))}
    reserved_center_approach = {(0, int(target_z) - 2)}
    reserved_occlusion_row_z = int(target_z) - 1

    if target_x != 0 or target_z < 5:
        bad.append((idx, variant, "unsupported_target_local", target))
    if arena_mode != "o2_target_funnel":
        bad.append((idx, variant, "arena_not_o2_target_funnel", arena_mode))
    if not disable_jitter:
        bad.append((idx, variant, "spawn_jitter_not_disabled", disable_jitter))
    if not positions & center_blocker_candidates:
        bad.append((idx, variant, "missing_center_blocker", sorted(positions)))
    if any((int(x), int(z)) in reserved_center_approach for x, z in positions):
        bad.append((idx, variant, "obstacle_in_reserved_center_approach", sorted(positions)))
    if any(int(z) >= reserved_occlusion_row_z for _, z in positions):
        bad.append((idx, variant, "obstacle_in_reserved_occlusion_or_target_row", sorted(positions)))
    path_ok, open_path = has_open_path_to_funnel(
        positions,
        target_z=target[1],
        half_width=half_width,
        back_z=back_z,
        front_z=front_z,
        funnel_start_z=funnel_start_z,
    )
    if not path_ok:
        bad.append((idx, variant, "no_open_path_to_funnel", sorted(positions)))
    wall_x = half_width + 1
    lines += [
        f"## {idx:02d} {variant}",
        "",
        f"positions: `{sorted(positions)}`  ",
        f"validated_open_path: `{open_path}`  ",
        f"height: `{height}`  ",
        f"material: `{material}`  ",
        f"target: `{target}`  ",
        f"arena: `{arena_mode}`, half_width=`{half_width}`, back_z=`{back_z}`, funnel_start_z=`{funnel_start_z}`, reserved_o_z=`{reserved_occlusion_row_z}`, front_z=`{front_z}`",
        "",
        "```text",
        "x:   " + " ".join(f"{x:>2}" for x in range(-wall_x, wall_x + 1)),
    ]

    for z in range(front_z, back_z - 1, -1):
        cells = []
        for x in range(-wall_x, wall_x + 1):
            if (x, z) == (0, 0):
                char = "S"
            elif (x, z) in positions:
                char = "#"
            elif x in {-wall_x, wall_x} or z == back_z:
                char = "W"
            elif (x, z) in reward_labels:
                char = reward_labels[(x, z)]
            elif z >= funnel_start_z:
                if z > target[1]:
                    char = "W"
                elif z == target[1]:
                    if x == 0:
                        char = "T"
                    elif abs(x) == 1:
                        char = "."
                    else:
                        char = "W"
                elif abs(x) <= 1:
                    if (x, z) == (0, target[1] - 2):
                        char = "G"
                    elif z == target[1] - 1:
                        char = "O"
                    else:
                        char = "."
                else:
                    char = "W"
            else:
                char = "."
            cells.append(char)
        lines.append(f"z={z:<2} " + " ".join(cells))
    lines += ["```", ""]

if bad:
    raise SystemExit(f"Invalid P obstacle geometry: {bad}")
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"geometry={out_path}")
PY
}

run_video_probe() {
  if [[ "${RUN_VIDEO}" != "1" ]]; then
    log "skipping video probe"
    return
  fi

  local video_out="${MASTER_ROOT}/video_rocket2_${VIDEO_EPISODES}ep"
  log "running video probe first: 16 worlds x ${VIDEO_EPISODES} episodes"
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

run_baseline() {
  if [[ "${RUN_BASELINE}" != "1" ]]; then
    log "skipping baseline"
    return
  fi

  local baseline_out="${MASTER_ROOT}/baseline_rocket2_${BASELINE_EPISODES}ep"
  log "running baseline after videos: 16 worlds x ${BASELINE_EPISODES} episodes"
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

write_summary() {
  "${PYTHON_BIN}" - <<'PY' "${MASTER_ROOT}" "${BASELINE_EPISODES}" "${VIDEO_EPISODES}" "${GEOMETRY_MD}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = str(sys.argv[2])
video_episodes = str(sys.argv[3])
geometry_path = Path(sys.argv[4])
summary_path = root / f"baseline_rocket2_{episodes}ep" / "summary.json"
lines = [
    "# P Obstacle Calibration Summary",
    "",
    f"root: `{root}`",
    f"asset_dir: `{next((root / 'assets').glob('*'), '')}`",
    f"geometry: `{geometry_path}`",
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
P obstacle O2 target-funnel calibration

task=${TASK}
master_root=${MASTER_ROOT}
variant_set=${P_OBSTACLE_VARIANT_SET}
material_override=${P_OBSTACLE_MATERIAL:-variant_default}
height_override=${P_OBSTACLE_HEIGHT}
goal_pose_protocol=${P_GOAL_POSE_PROTOCOL}
video_episodes=${VIDEO_EPISODES}
video_workers=${VIDEO_WORKERS}
baseline_episodes=${BASELINE_EPISODES}
baseline_workers=${BASELINE_WORKERS}
EOF

ensure_supported_variant_set
generate_assets
validate_and_render_geometry
run_video_probe
run_baseline
write_summary

log "done"
log "master_root=${MASTER_ROOT}"
log "summary=${MASTER_ROOT}/calibration_summary.md"
log "geometry=${GEOMETRY_MD}"
