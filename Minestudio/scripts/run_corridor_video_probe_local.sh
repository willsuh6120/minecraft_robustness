#!/usr/bin/env bash
# Standalone corridor video probe.
#
# Runs a small number of episodes on one P-maze obstacle variant with video
# recording enabled, then post-processes every resulting episode directory
# with corridor_annotate_episode.py to add lane / causal-segment overlays.
#
# Required env var:
#   ASSET_DIR   — path to the maze_t6 asset bank (contains eval_bank/ and collect_block_plans/)
#
# Optional overrides (all have sensible defaults):
#   WORLD_IDX           — 1-based block index to probe (default: 9 = left_funnel_t6)
#   MODEL_PATH          — model to run (default: hf:phython96/ROCKET-2-1x-22w baseline)
#   EPISODES            — episodes to collect (default: 4)
#   STEP_BUDGET         — steps per episode (default: 150)
#   FOCUS_WEIGHT        — weight to highlight in annotation (default: 4.0)
#   OUT_ROOT            — output directory (default: outputs/.../corridor_video_probe/<timestamp>)
#   SKIP_ANNOTATE       — set to 1 to skip corridor annotation post-processing (default: 0)
#
# Example:
#   ASSET_DIR=/path/to/assets bash scripts/run_corridor_video_probe_local.sh
#   ASSET_DIR=/path/to/assets WORLD_IDX=11 bash scripts/run_corridor_video_probe_local.sh
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_ROCKET_DIR="${ROOT_DIR}/minestudio/tutorials/inference/evaluate_rocket"

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

# ── Parameters ────────────────────────────────────────────────────────────────
ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v6_maze_t6_calibration" \
      -path "*/assets/*/eval_bank" -type d 2>/dev/null \
      | sed 's#/eval_bank$##' | sort | tail -n 1 || true
  )"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR missing or invalid. Set ASSET_DIR to the maze_t6 asset bank." >&2
  exit 1
fi

WORLD_IDX="${WORLD_IDX:-9}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
EPISODES="${EPISODES:-4}"
STEP_BUDGET="${STEP_BUDGET:-150}"
FOCUS_WEIGHT="${FOCUS_WEIGHT:-4.0}"
SKIP_ANNOTATE="${SKIP_ANNOTATE:-0}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/corridor_video_probe/$(date +%Y%m%d_%H%M%S)_world${WORLD_IDX}}"

BANK_DIR="${ASSET_DIR}/eval_bank"
BANK_MANIFEST="${BANK_DIR}/bank_manifest.json"
if [[ ! -f "${BANK_MANIFEST}" ]]; then
  echo "Missing bank manifest: ${BANK_MANIFEST}" >&2; exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "[corridor-probe] asset_dir=${ASSET_DIR}"
echo "[corridor-probe] world_idx=${WORLD_IDX}"
echo "[corridor-probe] model=${MODEL_PATH}"
echo "[corridor-probe] episodes=${EPISODES}"
echo "[corridor-probe] out_root=${OUT_ROOT}"

# ── Extract task_group_path for the chosen world ──────────────────────────────
INSTANCE_JSON="${OUT_ROOT}/instance_info.json"
"${PYTHON_BIN}" - <<'PY' "${BANK_MANIFEST}" "${WORLD_IDX}" "${INSTANCE_JSON}"
import json, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
world_idx = int(sys.argv[2])
out_path = Path(sys.argv[3])

worlds = manifest.get("instance_worlds") or []
chosen = next((w for w in worlds if int(w.get("instance_idx", -1)) == world_idx - 1), None)
if chosen is None:
    # Fall back: pick by position
    if 0 < world_idx <= len(worlds):
        chosen = worlds[world_idx - 1]
if chosen is None:
    raise SystemExit(f"No world at index {world_idx} in {sys.argv[1]}")

out_path.write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(chosen, ensure_ascii=False))
PY

TASK_GROUP_PATH="$("${PYTHON_BIN}" -c "
import json; from pathlib import Path
d = json.loads(Path('${INSTANCE_JSON}').read_text())
print(d.get('generated_task_group_dir') or d.get('task_group_path') or '')
")"
VARIANT_ID="$("${PYTHON_BIN}" -c "
import json; from pathlib import Path
d = json.loads(Path('${INSTANCE_JSON}').read_text())
print(d.get('path_obstacle_variant_id') or f'world_{${WORLD_IDX}}')
")"

if [[ -z "${TASK_GROUP_PATH}" || ! -d "${TASK_GROUP_PATH}" ]]; then
  echo "[corridor-probe] could not resolve task_group_path for world ${WORLD_IDX}" >&2
  exit 1
fi

echo "[corridor-probe] variant=${VARIANT_ID}  task_group_path=${TASK_GROUP_PATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

# ── Run rollout with video enabled ────────────────────────────────────────────
ROLLOUT_CMD=(
  "${PYTHON_BIN}"
  "${EVAL_ROCKET_DIR}/interaction_crossview_rollout.py"
  --model-path "${MODEL_PATH}"
  --tasks mine_coal
  --task-group "corridor_probe_w${WORLD_IDX}"
  --task-group-path "${TASK_GROUP_PATH}"
  --env-source rocket2_official
  --episodes-per-task "${EPISODES}"
  --base-seed 42
  --step-budget-override "${STEP_BUDGET}"
  --stop-on-success
  --auto-goal
  --out-dir "${OUT_ROOT}/rollout"
  # video ON: no --skip-video flag
)

if [[ -n "${XVFB_RUN}" ]]; then
  "${XVFB_RUN}" ${XVFB_SCREEN_ARGS} "${ROLLOUT_CMD[@]}"
else
  "${ROLLOUT_CMD[@]}"
fi

echo "[corridor-probe] rollout done"

# ── Post-process with corridor annotation ────────────────────────────────────
if [[ "${SKIP_ANNOTATE}" != "1" ]]; then
  echo "[corridor-probe] annotating episodes..."
  "${PYTHON_BIN}" \
    "${EVAL_ROCKET_DIR}/corridor_annotate_episode.py" \
    "${OUT_ROOT}/rollout" \
    --recurse \
    --focus-weight "${FOCUS_WEIGHT}" \
    --overwrite
  echo "[corridor-probe] annotation complete"
fi

echo "[corridor-probe] done  out_root=${OUT_ROOT}"
