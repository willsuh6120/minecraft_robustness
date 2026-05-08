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
RUN_TAG="${RUN_TAG:-h_heading_offset_calibration_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/h_heading_offset_calibration/${RUN_TAG}}"

ASSET_DIR="${ASSET_DIR:-}"
LAYOUT_CASES="${LAYOUT_CASES:-straight}"
COLLECT_HEADINGS="${COLLECT_HEADINGS:-0,90,120,150,180}"
EVAL_HEADINGS="${EVAL_HEADINGS:-0,90,120,150,180}"
FINAL_HEADINGS="${FINAL_HEADINGS:-${EVAL_HEADINGS}}"
EVAL_INSTANCES="${EVAL_INSTANCES:-0}"
FINAL_INSTANCES="${FINAL_INSTANCES:-0}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOALS="${BAKE_GOALS:-1}"

BANK_NAME="${BANK_NAME:-eval_bank}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
EPISODES="${EPISODES:-16}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
GOAL_SPEC="${GOAL_SPEC:-}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
RESULT_TAG="${RESULT_TAG:-baseline_rocket2_${EPISODES}ep}"
OUT_ROOT="${OUT_ROOT:-${MASTER_ROOT}/${RESULT_TAG}}"

RUN_VIDEO="${RUN_VIDEO:-0}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
VIDEO_LIMIT_INSTANCES="${VIDEO_LIMIT_INSTANCES:-3}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

mkdir -p "${MASTER_ROOT}"

log() {
  echo "[h-heading-calibration] $*"
}

resolve_latest_asset_dir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

ensure_assets() {
  if [[ -n "${ASSET_DIR}" && -d "${ASSET_DIR}" ]]; then
    log "using existing asset_dir=${ASSET_DIR}"
    return
  fi

  local asset_root="${MASTER_ROOT}/assets"
  mkdir -p "${asset_root}"
  log "generating H heading assets under ${asset_root}"

  gen_cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_h_heading_assets
    --tasks "${TASK}"
    --layout-cases "${LAYOUT_CASES}"
    --collect-headings "${COLLECT_HEADINGS}"
    --eval-headings "${EVAL_HEADINGS}"
    --final-headings "${FINAL_HEADINGS}"
    --eval-instances "${EVAL_INSTANCES}"
    --final-instances "${FINAL_INSTANCES}"
    --bank-workers "${BANK_WORKERS}"
    --out-dir "${asset_root}"
  )
  if [[ "${BAKE_GOALS}" == "1" ]]; then
    gen_cmd+=( --bake-goals )
  fi
  "${gen_cmd[@]}"

  ASSET_DIR="$(resolve_latest_asset_dir "${asset_root}")"
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "Could not resolve generated ASSET_DIR under ${asset_root}" >&2
    exit 1
  fi
  log "asset_dir=${ASSET_DIR}"
}

prepare_instance_tsv() {
  local bank_manifest="$1"
  local out_tsv="$2"
  local limit_instances="$3"
  "${PYTHON_BIN}" - <<'PY' "${bank_manifest}" "${out_tsv}" "${limit_instances}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
limit_instances = int(sys.argv[3])
lines = []
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(lines)))
    task_group_path = str(item.get("generated_task_group_dir") or item.get("task_group_path") or "").strip()
    if not task_group_path:
        continue
    variant = str(item.get("heading_variant_id") or "").strip()
    if not variant:
        rows = item.get("plan_rows") or []
        if rows:
            suggestions = rows[0].get("world_generation_suggestions") or {}
            variant = str(suggestions.get("mine_heading_variant_id") or "").strip()
    lines.append(f"{idx}\t{variant or f'instance_{idx:03d}'}\t{task_group_path}")
    if limit_instances > 0 and len(lines) >= limit_instances:
        break
out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"instances={len(lines)}")
PY
}

run_instance() {
  local instance_idx="$1"
  local variant="$2"
  local task_group_path="$3"
  local episodes="$4"
  local skip_video="$5"
  local out_root="$6"
  local instance_tag
  instance_tag="$(printf 'instance_%03d' "${instance_idx}")"
  local out_dir="${out_root}/${instance_tag}_${variant}"
  mkdir -p "${out_dir}"

  cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout
    --env-source rocket2_official
    --protocol ours_v1
    --task-group "h_heading_${BANK_NAME}_${instance_tag}_${variant}"
    --task-group-path "${task_group_path}"
    --tasks "${TASK}"
    --episodes-per-task "${episodes}"
    --base-seed "${BASE_SEED}"
    --seed-step "${SEED_STEP}"
    --sampling-base-seed "${SAMPLING_BASE_SEED}"
    --sampling-seed-step "${SAMPLING_SEED_STEP}"
    --episode-retries "${EPISODE_RETRIES}"
    --step-budget-override "${STEP_BUDGET}"
    --model-path "${MODEL_PATH}"
    --out-dir "${out_dir}"
    --cfg-coef "${CFG_COEF}"
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
  if [[ "${skip_video}" == "1" ]]; then
    cmd+=( --skip-video )
  fi

  log "start ${instance_tag} variant=${variant} episodes=${episodes} skip_video=${skip_video}"
  if [[ -n "${XVFB_RUN}" ]]; then
    "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
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
    echo "At least one H heading probe worker failed." >&2
    exit 1
  fi
}

render_summary() {
  local out_root="$1"
  local bank_manifest="$2"
  "${PYTHON_BIN}" - <<'PY' "${out_root}" "${bank_manifest}"
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rows = []
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(rows)))
    variant = str(item.get("heading_variant_id") or "")
    layout_case = str(item.get("layout_case") or "")
    heading_deg = item.get("heading_deg")
    heading_sign_label = str(item.get("heading_sign_label") or "")
    if not variant:
        plan_rows = item.get("plan_rows") or []
        if plan_rows:
            suggestions = plan_rows[0].get("world_generation_suggestions") or {}
            variant = str(suggestions.get("mine_heading_variant_id") or "")
    episode_rows = []
    for ep_csv in sorted(out_root.glob(f"instance_{idx:03d}_*/**/episodes.csv")):
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
            "variant_id": variant or f"instance_{idx:03d}",
            "layout_case": layout_case,
            "heading_deg": int(heading_deg or 0),
            "heading_sign_label": heading_sign_label,
            "episodes": episodes,
            "successful_episodes": successes,
            "success_rate": (successes / episodes) if episodes else None,
            "mean_steps": mean_steps,
        }
    )

rows.sort(key=lambda row: int(row["instance_idx"]))
overall_eps = sum(int(row["episodes"]) for row in rows)
overall_success = sum(int(row["successful_episodes"]) for row in rows)
payload = {
    "asset_dir": str(manifest.get("bank_dir") or ""),
    "bank_manifest": str(Path(sys.argv[2]).resolve()),
    "variants": rows,
    "overall_episodes": overall_eps,
    "overall_successful_episodes": overall_success,
    "overall_success_rate": (overall_success / overall_eps) if overall_eps else None,
}
(out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def pct(success, eps):
    return "" if not eps else f"{100.0 * float(success) / float(eps):.1f}%"

lines = [
    "# H Heading Offset Calibration Summary",
    "",
    f"bank_manifest: `{Path(sys.argv[2]).resolve()}`",
    "",
    f"overall: `{overall_success}/{overall_eps}` ({pct(overall_success, overall_eps)})",
    "",
    "| idx | variant | layout | heading | sign | success | mean steps |",
    "|---:|---|---|---:|---|---:|---:|",
]
for row in rows:
    mean_steps = "" if row["mean_steps"] is None else f"{float(row['mean_steps']):.1f}"
    lines.append(
        f"| {row['instance_idx']} | {row['variant_id']} | {row['layout_case']} | "
        f"{row['heading_deg']} | {row['heading_sign_label'] or 'opp'} | "
        f"{row['successful_episodes']}/{row['episodes']} ({pct(row['successful_episodes'], row['episodes'])}) | {mean_steps} |"
    )
(out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_root / "summary.md")
PY
}

run_probe() {
  local bank_name="$1"
  local episodes="$2"
  local skip_video="$3"
  local out_root="$4"
  local limit_instances="$5"
  local bank_dir="${ASSET_DIR}/${bank_name}"
  local bank_manifest="${bank_dir}/bank_manifest.json"
  local instance_tsv="${out_root}/instances.tsv"
  mkdir -p "${out_root}"
  prepare_instance_tsv "${bank_manifest}" "${instance_tsv}" "${limit_instances}"
  if [[ ! -s "${instance_tsv}" ]]; then
    echo "No fixed-bank instances found in ${bank_manifest}" >&2
    exit 1
  fi
  pids=()
  while IFS=$'\t' read -r instance_idx variant task_group_path; do
    [[ -n "${instance_idx}" ]] || continue
    run_instance "${instance_idx}" "${variant}" "${task_group_path}" "${episodes}" "${skip_video}" "${out_root}" &
    pids+=( "$!" )
    if (( ${#pids[@]} >= WORLD_WORKERS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done < "${instance_tsv}"
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
  render_summary "${out_root}" "${bank_manifest}"
}

ensure_assets

if [[ ! -d "${ASSET_DIR}/${BANK_NAME}" ]]; then
  echo "Missing bank under ASSET_DIR: ${ASSET_DIR}/${BANK_NAME}" >&2
  exit 1
fi

cat > "${MASTER_ROOT}/README.txt" <<EOF
H heading offset calibration

task=${TASK}
asset_dir=${ASSET_DIR}
bank_name=${BANK_NAME}
model_path=${MODEL_PATH}
episodes=${EPISODES}
world_workers=${WORLD_WORKERS}
goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}
layout_cases=${LAYOUT_CASES}
collect_headings=${COLLECT_HEADINGS}
eval_headings=${EVAL_HEADINGS}
final_headings=${FINAL_HEADINGS}
EOF

log "master_root=${MASTER_ROOT}"
log "asset_dir=${ASSET_DIR}"
log "bank_name=${BANK_NAME}"
log "out_root=${OUT_ROOT}"
run_probe "${BANK_NAME}" "${EPISODES}" "${SKIP_VIDEO}" "${OUT_ROOT}" 0

if [[ "${RUN_VIDEO}" == "1" ]]; then
  VIDEO_OUT="${MASTER_ROOT}/video_rocket2_${VIDEO_EPISODES}ep"
  log "running small video probe out_root=${VIDEO_OUT}"
  run_probe "${BANK_NAME}" "${VIDEO_EPISODES}" 0 "${VIDEO_OUT}" "${VIDEO_LIMIT_INSTANCES}"
  find "${VIDEO_OUT}" -type f -name "*_annotated.mp4" | sort > "${VIDEO_OUT}/annotated_video_manifest.txt"
  find "${VIDEO_OUT}" -type f -name "*.mp4" | sort > "${VIDEO_OUT}/video_manifest.txt"
fi

log "done master_root=${MASTER_ROOT}"
log "summary=${OUT_ROOT}/summary.md"
