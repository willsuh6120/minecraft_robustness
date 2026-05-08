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
OVERNIGHT_ROOT="${OVERNIGHT_ROOT:-}"
if [[ -z "${OVERNIGHT_ROOT}" ]]; then
  OVERNIGHT_ROOT="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_center_z3_overnight" \
      -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
fi
if [[ -z "${OVERNIGHT_ROOT}" || ! -d "${OVERNIGHT_ROOT}" ]]; then
  echo "OVERNIGHT_ROOT is missing. Set it to the completed P center_z3 overnight run." >&2
  exit 1
fi

ASSET_DIR="${ASSET_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
CHAIN_SUMMARY="${CHAIN_SUMMARY:-${OVERNIGHT_ROOT}/training/teacher_to_hard/seed_001/chain_summary.json}"
if [[ -z "${ASSET_DIR}" || -z "${MODEL_PATH}" ]]; then
  read -r detected_asset detected_model < <(
    "${PYTHON_BIN}" - <<'PY' "${OVERNIGHT_ROOT}" "${CHAIN_SUMMARY}"
import json
import sys
from pathlib import Path

overnight_root = Path(sys.argv[1])
chain_summary_path = Path(sys.argv[2])

asset_dir = ""
asset_candidates = sorted((overnight_root / "assets").glob("*"))
if asset_candidates:
    asset_dir = str(asset_candidates[-1])

model_path = ""
if chain_summary_path.exists():
    chain = json.loads(chain_summary_path.read_text(encoding="utf-8"))
    blocks = chain.get("blocks") or []
    if blocks:
        model_path = str(blocks[-1].get("next_model_path") or "")
    if not model_path:
        for block in reversed(blocks):
            sel = block.get("best_eval_selection") or {}
            model_path = str(sel.get("model_path") or "")
            if model_path:
                break
print(asset_dir, model_path)
PY
  )
  ASSET_DIR="${ASSET_DIR:-${detected_asset}}"
  MODEL_PATH="${MODEL_PATH:-${detected_model}}"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is missing or invalid: ${MODEL_PATH}" >&2
  exit 1
fi

EPISODES="${EPISODES:-32}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
STEP_BUDGET="${STEP_BUDGET:-150}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_center_z3_final_model_probe/$(date +%Y%m%d_%H%M%S)}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${OVERNIGHT_ROOT}/baseline_rocket2_32ep/summary.json}"

echo "[p-center-z3-final-probe] overnight_root=${OVERNIGHT_ROOT}"
echo "[p-center-z3-final-probe] asset_dir=${ASSET_DIR}"
echo "[p-center-z3-final-probe] model_path=${MODEL_PATH}"
echo "[p-center-z3-final-probe] episodes_per_world=${EPISODES}"
echo "[p-center-z3-final-probe] world_workers=${WORLD_WORKERS}"
echo "[p-center-z3-final-probe] out_root=${OUT_ROOT}"

ASSET_DIR="${ASSET_DIR}" \
TASK="${TASK}" \
BANK_NAME=eval_bank \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
MODEL_PATH="${MODEL_PATH}" \
CFG_COEF="${CFG_COEF:-0.0}" \
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}" \
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}" \
GOAL_SPEC="" \
BASE_SEED="${BASE_SEED:-1}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}" \
EPISODE_RETRIES="${EPISODE_RETRIES:-2}" \
STEP_BUDGET="${STEP_BUDGET}" \
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}" \
SKIP_VIDEO="${SKIP_VIDEO:-1}" \
OUT_ROOT="${OUT_ROOT}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}/summary.json" "${BASELINE_SUMMARY}" "${OUT_ROOT}/trained_vs_baseline.md"
import json
import sys
from pathlib import Path

trained_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

trained = json.loads(trained_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {"variants": []}

base_by_idx = {int(row["instance_idx"]): row for row in baseline.get("variants") or []}
trained_rows = sorted(trained.get("variants") or [], key=lambda row: int(row.get("instance_idx", 0)))

base_total_eps = sum(int(row.get("episodes") or 0) for row in base_by_idx.values())
base_total_success = sum(int(row.get("successful_episodes") or 0) for row in base_by_idx.values())
trained_total_eps = sum(int(row.get("episodes") or 0) for row in trained_rows)
trained_total_success = sum(int(row.get("successful_episodes") or 0) for row in trained_rows)

def pct(success, eps):
    return "" if not eps else f"{100.0 * float(success) / float(eps):.1f}%"

lines = [
    "# P Center-Z3 Trained vs Baseline Probe",
    "",
    f"trained_summary: `{trained_path}`",
    f"baseline_summary: `{baseline_path}`" if baseline_path.exists() else "baseline_summary: missing",
    "",
    "| model | success | rate |",
    "|---|---:|---:|",
    f"| baseline | {base_total_success}/{base_total_eps} | {pct(base_total_success, base_total_eps)} |",
    f"| trained | {trained_total_success}/{trained_total_eps} | {pct(trained_total_success, trained_total_eps)} |",
    "",
    "| idx | variant | baseline | trained | delta pp |",
    "|---:|---|---:|---:|---:|",
]

for row in trained_rows:
    idx = int(row.get("instance_idx", 0))
    base = base_by_idx.get(idx, {})
    variant = str(row.get("variant_id") or base.get("variant_id") or f"instance_{idx:03d}")
    b_eps = int(base.get("episodes") or 0)
    b_succ = int(base.get("successful_episodes") or 0)
    t_eps = int(row.get("episodes") or 0)
    t_succ = int(row.get("successful_episodes") or 0)
    b_rate = b_succ / b_eps if b_eps else None
    t_rate = t_succ / t_eps if t_eps else None
    delta = "" if b_rate is None or t_rate is None else f"{100.0 * (t_rate - b_rate):+.1f}"
    lines.append(
        f"| {idx} | {variant} | {b_succ}/{b_eps} ({pct(b_succ, b_eps)}) | "
        f"{t_succ}/{t_eps} ({pct(t_succ, t_eps)}) | {delta} |"
    )

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path)
PY

echo "[p-center-z3-final-probe] comparison=${OUT_ROOT}/trained_vs_baseline.md"
