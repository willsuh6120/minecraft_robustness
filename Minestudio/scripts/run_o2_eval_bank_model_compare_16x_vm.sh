#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/minestudio/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

XVFB_RUN="${XVFB_RUN:-$(command -v xvfb-run || true)}"
XVFB_SCREEN_ARGS="${XVFB_SCREEN_ARGS:--screen 0 1280x1024x24}"
if [[ -z "${XVFB_RUN}" ]]; then
  echo "xvfb-run not found. Install xvfb first." >&2
  exit 1
fi

TASK="${TASK:-mine_coal}"
TASK_GROUP_PREFIX="${TASK_GROUP_PREFIX:-mine_o2_eval_bank_16x_compare}"
RUN_ROOT="${RUN_ROOT:-}"
ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi

BANK_NAME="${BANK_NAME:-eval_bank}"
BANK_DIR="${BANK_DIR:-${ASSET_DIR}/${BANK_NAME}}"
BANK_MANIFEST="${BANK_MANIFEST:-${BANK_DIR}/bank_manifest.json}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

EPISODES_PER_WORLD="${EPISODES_PER_WORLD:-16}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/o2_eval_bank_16x_model_compare/$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
usage:
  ASSET_DIR=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm/20260427_054807 \
  RUN_ROOT=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330 \
  bash scripts/run_o2_eval_bank_model_compare_16x_vm.sh

optional env vars:
  TRAINED_MODEL_PATH=/path/to/model.pt   # if omitted, auto-select best eval checkpoint from RUN_ROOT
  EPISODES_PER_WORLD=16
  WORLD_WORKERS=4
  SKIP_VIDEO=1
  OUT_ROOT=/path/to/output
EOF
}

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  usage >&2
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BANK_MANIFEST}" ]]; then
  usage >&2
  echo "BANK_MANIFEST not found: ${BANK_MANIFEST}" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

configure_torch_cuda_runtime() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi
  local python_version
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_pkg="${CONDA_PREFIX}/lib/python${python_version}/site-packages/nvidia"
  local lib_paths=()
  local subdir
  for subdir in \
    cublas/lib \
    cudnn/lib \
    cuda_runtime/lib \
    curand/lib \
    cusolver/lib \
    cusparse/lib \
    nvjitlink/lib
  do
    if [[ -d "${site_pkg}/${subdir}" ]]; then
      lib_paths+=("${site_pkg}/${subdir}")
    fi
  done
  if (( ${#lib_paths[@]} == 0 )); then
    return 0
  fi
  local joined
  joined="$(IFS=:; echo "${lib_paths[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

configure_torch_cuda_runtime
mkdir -p "${OUT_ROOT}"

if [[ -z "${TRAINED_MODEL_PATH}" ]]; then
  if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
    usage >&2
    echo "TRAINED_MODEL_PATH is empty, so RUN_ROOT must point to the original O2 chain run." >&2
    exit 1
  fi
  TRAINED_MODEL_PATH="$("${PYTHON_BIN}" - <<'PY' "${RUN_ROOT}"
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
candidates = []

def add_candidate(path: Path, source: str, payload: dict):
    model_path = str(payload.get("model_path") or "").strip()
    if not model_path:
        return
    try:
        rate = float(payload.get("auto_success_rate", -1.0) or -1.0)
    except Exception:
        rate = -1.0
    try:
        iteration = int(payload.get("iteration", -1) or -1)
    except Exception:
        iteration = -1
    candidates.append((rate, iteration, str(path), source, model_path))

for chain_summary_path in sorted(run_root.rglob("chain_summary.json")):
    try:
        chain = json.loads(chain_summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for block in chain.get("blocks") or []:
        sel = block.get("best_eval_selection") or {}
        add_candidate(chain_summary_path, "chain_summary.best_eval_selection", sel)

for pilot_state_path in sorted(run_root.rglob("pilot_state.json")):
    try:
        state = json.loads(pilot_state_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    sel = state.get("best_eval_selection") or {}
    add_candidate(pilot_state_path, "pilot_state.best_eval_selection", sel)
    sel = state.get("final_eval_model_selection") or {}
    add_candidate(pilot_state_path, "pilot_state.final_eval_model_selection", sel)

if not candidates:
    raise SystemExit(f"No best-eval checkpoint candidates found under {run_root}")

candidates.sort(key=lambda item: (item[0], item[1], item[2]))
print(candidates[-1][4])
PY
)"
fi

if [[ "${TRAINED_MODEL_PATH}" != hf:* && ! -f "${TRAINED_MODEL_PATH}" ]]; then
  echo "TRAINED_MODEL_PATH does not exist: ${TRAINED_MODEL_PATH}" >&2
  exit 1
fi

INSTANCE_TSV="${OUT_ROOT}/instances.tsv"
"${PYTHON_BIN}" - <<'PY' "${BANK_MANIFEST}" "${INSTANCE_TSV}"
import json
import re
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])

def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"

lines = []
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(lines)))
    task_group_path = str(item.get("generated_task_group_dir") or item.get("task_group_path") or "").strip()
    if not task_group_path:
        continue
    rows = item.get("plan_rows") or []
    suggestions = (rows[0].get("world_generation_suggestions") or {}) if rows else {}
    variant = str(suggestions.get("mine_occluder_variant_id") or item.get("occluder_variant_id") or "").strip()
    top_row = "".join(str(x) for x in suggestions.get("mine_occluder_top_row") or [])
    bottom_row = "".join(str(x) for x in suggestions.get("mine_occluder_bottom_row") or [])
    if not variant:
        variant = f"instance_{idx:03d}"
    pattern = f"{top_row}/{bottom_row}" if top_row or bottom_row else ""
    lines.append(f"{idx}\t{slug(variant)}\t{pattern}\t{task_group_path}")

out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"instances={len(lines)}")
PY

if [[ ! -s "${INSTANCE_TSV}" ]]; then
  echo "No fixed-bank instances found in ${BANK_MANIFEST}" >&2
  exit 1
fi

echo "[o2-eval16-compare] asset_dir=${ASSET_DIR}"
echo "[o2-eval16-compare] bank_manifest=${BANK_MANIFEST}"
echo "[o2-eval16-compare] episodes_per_world=${EPISODES_PER_WORLD}"
echo "[o2-eval16-compare] world_workers=${WORLD_WORKERS}"
echo "[o2-eval16-compare] base_model=${BASE_MODEL_PATH}"
echo "[o2-eval16-compare] trained_model=${TRAINED_MODEL_PATH}"
echo "[o2-eval16-compare] skip_video=${SKIP_VIDEO}"
echo "[o2-eval16-compare] out_root=${OUT_ROOT}"

run_instance() {
  local model_label="$1"
  local model_path="$2"
  local instance_idx="$3"
  local variant="$4"
  local pattern="$5"
  local task_group_path="$6"
  local instance_tag
  instance_tag="$(printf 'instance_%03d' "${instance_idx}")"
  local out_dir="${OUT_ROOT}/${model_label}/${instance_tag}_${variant}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout
    --env-source rocket2_official
    --protocol ours_v1
    --task-group "${TASK_GROUP_PREFIX}_${model_label}_${instance_tag}_${variant}"
    --task-group-path "${task_group_path}"
    --tasks "${TASK}"
    --episodes-per-task "${EPISODES_PER_WORLD}"
    --base-seed "${BASE_SEED}"
    --seed-step "${SEED_STEP}"
    --sampling-base-seed "${SAMPLING_BASE_SEED}"
    --sampling-seed-step "${SAMPLING_SEED_STEP}"
    --episode-retries "${EPISODE_RETRIES}"
    --step-budget-override "${STEP_BUDGET}"
    --model-path "${model_path}"
    --out-dir "${out_dir}"
    --cfg-coef "${CFG_COEF}"
    --cfg-policy-mode "${CFG_POLICY_MODE}"
    --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}"
    --auto-goal
  )
  if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
    cmd+=( --stop-on-success )
  fi
  if [[ "${SKIP_VIDEO}" == "1" ]]; then
    cmd+=( --skip-video )
  fi

  echo "[o2-eval16-compare] start model=${model_label} ${instance_tag} variant=${variant} pattern=${pattern}"
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${cmd[@]}"
  echo "[o2-eval16-compare] done model=${model_label} ${instance_tag} variant=${variant}"
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
    echo "At least one O2 eval-bank probe worker failed." >&2
    exit 1
  fi
}

run_model() {
  local model_label="$1"
  local model_path="$2"
  local pids=()
  while IFS=$'\t' read -r instance_idx variant pattern task_group_path; do
    [[ -n "${instance_idx}" ]] || continue
    run_instance "${model_label}" "${model_path}" "${instance_idx}" "${variant}" "${pattern}" "${task_group_path}" &
    pids+=( "$!" )
    if (( ${#pids[@]} >= WORLD_WORKERS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done < "${INSTANCE_TSV}"
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

run_model "base" "${BASE_MODEL_PATH}"
run_model "trained" "${TRAINED_MODEL_PATH}"

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}" "${BANK_MANIFEST}" "${BASE_MODEL_PATH}" "${TRAINED_MODEL_PATH}" "${EPISODES_PER_WORLD}"
import csv
import json
import math
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
base_model_path = sys.argv[3]
trained_model_path = sys.argv[4]
episodes_per_world = int(sys.argv[5])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

meta_by_idx = {}
for item in manifest.get("instance_worlds") or []:
    idx = int(item.get("instance_idx", len(meta_by_idx)))
    rows = item.get("plan_rows") or []
    suggestions = (rows[0].get("world_generation_suggestions") or {}) if rows else {}
    top_row = "".join(str(x) for x in suggestions.get("mine_occluder_top_row") or [])
    bottom_row = "".join(str(x) for x in suggestions.get("mine_occluder_bottom_row") or [])
    meta_by_idx[idx] = {
        "variant_id": str(suggestions.get("mine_occluder_variant_id") or f"instance_{idx:03d}"),
        "pattern": f"{top_row}/{bottom_row}" if top_row or bottom_row else "",
        "task_group_path": str(item.get("generated_task_group_dir") or item.get("task_group_path") or ""),
    }

def load_rows(model_label: str):
    rows = {}
    for instance_dir in sorted((out_root / model_label).glob("instance_*")):
        parts = instance_dir.name.split("_")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[1])
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
        rows[idx] = {
            "episodes": episodes,
            "successful_episodes": successes,
            "success_rate": successes / episodes if episodes else None,
            "mean_steps": mean_steps,
            "instance_dir": str(instance_dir),
        }
    return rows

base = load_rows("base")
trained = load_rows("trained")
indices = sorted(set(meta_by_idx) | set(base) | set(trained))

def aggregate(rows):
    episodes = sum(int(v.get("episodes") or 0) for v in rows.values())
    successes = sum(int(v.get("successful_episodes") or 0) for v in rows.values())
    mean_steps_n = sum((float(v.get("mean_steps") or 0.0) * int(v.get("episodes") or 0)) for v in rows.values())
    return {
        "episodes": episodes,
        "successful_episodes": successes,
        "success_rate": successes / episodes if episodes else None,
        "mean_steps": mean_steps_n / episodes if episodes else None,
    }

def pct(x):
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.1f}%"

def pp(x):
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.1f}pp"

def wilson(successes, episodes):
    if episodes <= 0:
        return (None, None)
    z = 1.96
    p = successes / episodes
    denom = 1.0 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * episodes)) / episodes) / denom
    return (max(0.0, center - half), min(1.0, center + half))

world_rows = []
for idx in indices:
    b = base.get(idx, {})
    t = trained.get(idx, {})
    b_rate = b.get("success_rate")
    t_rate = t.get("success_rate")
    world_rows.append({
        "instance_idx": idx,
        **meta_by_idx.get(idx, {}),
        "base": b,
        "trained": t,
        "delta_success_rate": (t_rate - b_rate) if b_rate is not None and t_rate is not None else None,
    })

base_agg = aggregate(base)
trained_agg = aggregate(trained)
delta = None
if base_agg["success_rate"] is not None and trained_agg["success_rate"] is not None:
    delta = trained_agg["success_rate"] - base_agg["success_rate"]
base_ci = wilson(base_agg["successful_episodes"], base_agg["episodes"])
trained_ci = wilson(trained_agg["successful_episodes"], trained_agg["episodes"])

payload = {
    "bank_manifest": str(manifest_path),
    "episodes_per_world_requested": episodes_per_world,
    "base_model_path": base_model_path,
    "trained_model_path": trained_model_path,
    "base": base_agg,
    "trained": trained_agg,
    "delta_success_rate": delta,
    "base_wilson95": base_ci,
    "trained_wilson95": trained_ci,
    "worlds": world_rows,
}
(out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# O2 Eval Bank 16x Model Compare")
md.append("")
md.append(f"- bank_manifest: `{manifest_path}`")
md.append(f"- base_model: `{base_model_path}`")
md.append(f"- trained_model: `{trained_model_path}`")
md.append(f"- protocol: existing baked/auto goal from each O2 eval-bank task group")
md.append(f"- episodes: `{len(indices)}` worlds x `{episodes_per_world}` repeats/model")
md.append("")
md.append("## Overall")
md.append("")
md.append("| model | success | episodes | success rate | Wilson 95% CI | mean steps |")
md.append("|---|---:|---:|---:|---:|---:|")
md.append(
    f"| ROCKET-2 base | {base_agg['successful_episodes']} | {base_agg['episodes']} | {pct(base_agg['success_rate'])} | "
    f"{pct(base_ci[0])}-{pct(base_ci[1])} | {base_agg['mean_steps']:.2f} |"
)
md.append(
    f"| trained best | {trained_agg['successful_episodes']} | {trained_agg['episodes']} | {pct(trained_agg['success_rate'])} | "
    f"{pct(trained_ci[0])}-{pct(trained_ci[1])} | {trained_agg['mean_steps']:.2f} |"
)
md.append("")
md.append(f"Delta trained-base: `{pp(delta)}`")
md.append("")
md.append("## Per World")
md.append("")
md.append("| idx | variant | pattern | base | trained | delta | base mean steps | trained mean steps |")
md.append("|---:|---|---|---:|---:|---:|---:|---:|")
for row in sorted(world_rows, key=lambda r: r["instance_idx"]):
    b = row.get("base") or {}
    t = row.get("trained") or {}
    b_steps = b.get("mean_steps")
    t_steps = t.get("mean_steps")
    b_steps_text = f"{b_steps:.2f}" if b_steps is not None else "n/a"
    t_steps_text = f"{t_steps:.2f}" if t_steps is not None else "n/a"
    md.append(
        f"| {row['instance_idx']} | {row.get('variant_id', '')} | {row.get('pattern', '')} | "
        f"{int(b.get('successful_episodes') or 0)}/{int(b.get('episodes') or 0)} ({pct(b.get('success_rate'))}) | "
        f"{int(t.get('successful_episodes') or 0)}/{int(t.get('episodes') or 0)} ({pct(t.get('success_rate'))}) | "
        f"{pp(row.get('delta_success_rate'))} | "
        f"{b_steps_text} | "
        f"{t_steps_text} |"
    )

(out_root / "trained_vs_baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[o2-eval16-compare] comparison=${OUT_ROOT}/trained_vs_baseline.md"
echo "[o2-eval16-compare] summary=${OUT_ROOT}/summary.json"
echo "[o2-eval16-compare] done out_root=${OUT_ROOT}"
