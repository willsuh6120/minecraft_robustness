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
BANK_NAME="${BANK_NAME:-eval_bank}"
ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_assets}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing. Generate assets with scripts/prepare_mine_p_straight_obstacle_assets.sh first." >&2
  exit 1
fi

BANK_DIR="${ASSET_DIR}/${BANK_NAME}"
BANK_MANIFEST="${BANK_DIR}/bank_manifest.json"
if [[ ! -f "${BANK_MANIFEST}" ]]; then
  echo "Missing bank manifest: ${BANK_MANIFEST}" >&2
  exit 1
fi

EPISODES="${EPISODES:-16}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-1.0}"
GOAL_SPEC="${GOAL_SPEC:-}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_baseline_variant_probe/$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

mkdir -p "${OUT_ROOT}"

INSTANCE_TSV="${OUT_ROOT}/instances.tsv"
"${PYTHON_BIN}" - <<'PY' "${BANK_MANIFEST}" "${INSTANCE_TSV}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
lines = []
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(lines)))
    task_group_path = str(item.get("generated_task_group_dir") or item.get("task_group_path") or "").strip()
    if not task_group_path:
        continue
    variant = str(item.get("path_obstacle_variant_id") or "").strip()
    if not variant:
        rows = item.get("plan_rows") or []
        if rows:
            suggestions = (rows[0].get("world_generation_suggestions") or {})
            variant = str(suggestions.get("mine_path_obstacle_variant_id") or "").strip()
    lines.append(f"{idx}\t{variant or f'instance_{idx:03d}'}\t{task_group_path}")
out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"instances={len(lines)}")
PY

if [[ ! -s "${INSTANCE_TSV}" ]]; then
  echo "No fixed-bank instances found in ${BANK_MANIFEST}" >&2
  exit 1
fi

echo "[p-obstacle-baseline-probe] asset_dir=${ASSET_DIR}"
echo "[p-obstacle-baseline-probe] bank=${BANK_NAME}"
echo "[p-obstacle-baseline-probe] episodes_per_variant=${EPISODES}"
echo "[p-obstacle-baseline-probe] world_workers=${WORLD_WORKERS}"
echo "[p-obstacle-baseline-probe] model=${MODEL_PATH}"
echo "[p-obstacle-baseline-probe] env_reward_scale=${ENV_REWARD_SCALE}"
echo "[p-obstacle-baseline-probe] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"
echo "[p-obstacle-baseline-probe] out_root=${OUT_ROOT}"

run_instance() {
  local instance_idx="$1"
  local variant="$2"
  local task_group_path="$3"
  local instance_tag
  instance_tag="$(printf 'instance_%03d' "${instance_idx}")"
  local out_dir="${OUT_ROOT}/${instance_tag}_${variant}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout
    --env-source rocket2_official
    --protocol ours_v1
    --task-group "p_obstacle_${BANK_NAME}_${instance_tag}_${variant}"
    --task-group-path "${task_group_path}"
    --tasks "${TASK}"
    --episodes-per-task "${EPISODES}"
    --base-seed "${BASE_SEED}"
    --seed-step "${SEED_STEP}"
    --sampling-base-seed "${SAMPLING_BASE_SEED}"
    --sampling-seed-step "${SAMPLING_SEED_STEP}"
    --episode-retries "${EPISODE_RETRIES}"
    --step-budget-override "${STEP_BUDGET}"
    --model-path "${MODEL_PATH}"
    --out-dir "${out_dir}"
    --cfg-coef "${CFG_COEF}"
    --env-reward-scale "${ENV_REWARD_SCALE}"
    --cfg-policy-mode "${CFG_POLICY_MODE}"
    --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}"
  )
  if [[ -n "${GOAL_SPEC}" ]]; then
    cmd+=( --goal-spec "${GOAL_SPEC}" )
  else
    cmd+=( --auto-goal )
  fi
  if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
    cmd+=( --stop-on-success )
  fi
  if [[ "${SKIP_VIDEO}" == "1" ]]; then
    cmd+=( --skip-video )
  fi

  echo "[p-obstacle-baseline-probe] start ${instance_tag} variant=${variant}"
  if [[ -n "${XVFB_RUN}" ]]; then
    "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
  echo "[p-obstacle-baseline-probe] done ${instance_tag} variant=${variant}"
}

wait_batch() {
  local pid
  local failed=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "At least one P obstacle probe worker failed." >&2
    exit 1
  fi
}

pids=()
while IFS=$'\t' read -r instance_idx variant task_group_path; do
  [[ -n "${instance_idx}" ]] || continue
  run_instance "${instance_idx}" "${variant}" "${task_group_path}" &
  pids+=( "$!" )
  if (( ${#pids[@]} >= WORLD_WORKERS )); then
    wait_batch "${pids[@]}"
    pids=()
  fi
done < "${INSTANCE_TSV}"
if (( ${#pids[@]} > 0 )); then
  wait_batch "${pids[@]}"
fi

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}" "${BANK_MANIFEST}"
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
variant_by_idx = {}
positions_by_idx = {}
height_by_idx = {}
material_by_idx = {}
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(variant_by_idx)))
    variant = str(item.get("path_obstacle_variant_id") or "")
    positions = []
    rows = item.get("plan_rows") or []
    if rows:
        suggestions = rows[0].get("world_generation_suggestions") or {}
        variant = variant or str(suggestions.get("mine_path_obstacle_variant_id") or "")
        positions = suggestions.get("mine_path_obstacle_positions") or []
        height_by_idx[idx] = suggestions.get("mine_path_obstacle_height")
        material_by_idx[idx] = suggestions.get("mine_path_obstacle_material")
    variant_by_idx[idx] = variant or f"instance_{idx:03d}"
    positions_by_idx[idx] = positions

rows = []
for instance_dir in sorted(out_root.glob("instance_*")):
    name = instance_dir.name
    try:
        idx = int(name.split("_")[1])
    except Exception:
        continue
    episode_rows = []
    for ep_csv in sorted(instance_dir.glob("*/episodes.csv")):
        with ep_csv.open(newline="", encoding="utf-8") as handle:
            episode_rows.extend(list(csv.DictReader(handle)))
    episodes = len(episode_rows)
    successes = sum(str(row.get("auto_success", "")).lower() == "true" for row in episode_rows)
    mean_steps = (
        sum(float(row.get("num_steps") or 0.0) for row in episode_rows) / episodes
        if episodes
        else None
    )
    rows.append(
        {
            "instance_idx": idx,
            "variant_id": variant_by_idx.get(idx, name),
            "positions": positions_by_idx.get(idx, []),
            "height": height_by_idx.get(idx),
            "material": material_by_idx.get(idx),
            "episodes": episodes,
            "successful_episodes": successes,
            "success_rate": successes / episodes if episodes else None,
            "mean_steps": mean_steps,
            "instance_dir": str(instance_dir),
        }
    )

rows.sort(key=lambda item: (item["success_rate"] is None, item["success_rate"] if item["success_rate"] is not None else 999, item["instance_idx"]))
payload = {
    "bank_manifest": str(Path(sys.argv[2])),
    "variants": rows,
}
(out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[p-obstacle-baseline-probe] done out_root=${OUT_ROOT}"
