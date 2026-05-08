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
RUN_TAG="${RUN_TAG:-h_heading_offset_learning_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/h_heading_offset_learning_signal/${RUN_TAG}}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/h_heading_offset_calibration" \
      -name asset_manifest.json -type f 2>/dev/null \
      | sort \
      | tail -n 1 \
      | xargs -r dirname || true
  )"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid. Run run_h_heading_offset_calibration_local.sh first or set ASSET_DIR." >&2
  exit 1
fi
if [[ ! -d "${ASSET_DIR}/eval_bank" || ! -d "${ASSET_DIR}/collect_block_plans" ]]; then
  echo "ASSET_DIR does not look like an H heading asset bank: ${ASSET_DIR}" >&2
  exit 1
fi

BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
if [[ -z "${BASELINE_SUMMARY}" ]]; then
  CALIB_ROOT="$(dirname "$(dirname "${ASSET_DIR}")")"
  BASELINE_SUMMARY="${CALIB_ROOT}/baseline_rocket2_16ep/summary.json"
fi

BASE_SEEDS="${BASE_SEEDS:-1}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PROBE_WORKERS="${PROBE_WORKERS:-4}"
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES:-16}"
RUN_FINAL_PROBES="${RUN_FINAL_PROBES:-1}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"
VF_COEF="${VF_COEF:-0.25}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE:-advantage_top_k}"
LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K:-32}"
LOSS_FOCUS_CONTEXT_LEN="${LOSS_FOCUS_CONTEXT_LEN:-96}"
LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT:-4.0}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal_same_world_baked}"
GOAL_SPEC="${GOAL_SPEC:-}"
PLAN="${PLAN:-focused}"
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL:-1}"

mkdir -p "${MASTER_ROOT}"

log() {
  echo "[h-heading-learning] $*"
}

variant_order_names() {
  case "$1" in
    smoke) printf '%s\n' "straight_H090_pos" ;;
    focused) printf '%s\n' "straight_H090_pos,straight_H090_neg,straight_H120_pos,straight_H120_neg,straight_H150_pos,straight_H150_neg,straight_H090_pos,straight_H120_pos" ;;
    single_straight_H90) printf '%s\n' "straight_H090_pos,straight_H090_pos,straight_H090_pos,straight_H090_pos,straight_H090_pos,straight_H090_pos,straight_H090_pos,straight_H090_pos" ;;
    no_update_focused) printf '%s\n' "straight_H090_pos,straight_H090_neg,straight_H120_pos,straight_H120_neg,straight_H150_pos,straight_H150_neg,straight_H090_pos,straight_H120_pos" ;;
    *) echo "Unknown PLAN=${PLAN}" >&2; return 1 ;;
  esac
}

variant_disable_updates() {
  case "$1" in
    no_update_*) printf '%s\n' "1" ;;
    *) printf '%s\n' "0" ;;
  esac
}

resolve_block_order_csv() {
  local variant_csv="$1"
  "${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}/asset_manifest.json" "${variant_csv}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
requested = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
mapping = {}
for row in manifest.get("collect_block_variant_map") or []:
    mapping[str(row.get("variant_id") or "").strip()] = int(row.get("block_index") or 0)
missing = [name for name in requested if name not in mapping]
if missing:
    raise SystemExit(f"Missing collect block variants in asset_manifest: {missing}")
print(",".join(str(mapping[name]) for name in requested))
PY
}

resolve_collect_block_map_md() {
  "${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}/asset_manifest.json"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in manifest.get("collect_block_variant_map") or []:
    print(f"{int(row.get('block_index') or 0):02d} {row.get('variant_id')}")
PY
}

run_chain() {
  local variant="$1"
  local seed="$2"
  local requested_variants
  local order_csv
  local disable_updates
  requested_variants="$(variant_order_names "${variant}")"
  order_csv="$(resolve_block_order_csv "${requested_variants}")"
  disable_updates="$(variant_disable_updates "${variant}")"

  local seed_tag
  seed_tag="$(printf 'seed_%03d' "${seed}")"
  local out_root="${MASTER_ROOT}/training/${variant}/${seed_tag}"

  log "variant=${variant} seed=${seed_tag} order_names=${requested_variants}"
  log "variant=${variant} seed=${seed_tag} order_csv=${order_csv}"

  TASK="${TASK}" \
  TASK_GROUP_NAME="mine_h_heading_${variant}_${seed_tag}" \
  ASSET_DIR="${ASSET_DIR}" \
  OUT_ROOT="${out_root}" \
  EXPERIMENT_TAG="h_heading_${variant}_${seed_tag}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  BLOCK_COUNT="$("${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}/asset_manifest.json"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(manifest.get("collect_block_variant_map") or []))
PY
)" \
  BLOCK_ORDER="${order_csv}" \
  DISABLE_UPDATES="${disable_updates}" \
  SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL}" \
  BASE_SEED="${seed}" \
  SAMPLING_BASE_SEED="${seed}" \
  BAKE_GOAL_BASE_SEED="${seed}" \
  COLLECT_EPISODES="${COLLECT_EPISODES}" \
  EVAL_EPISODES="${EVAL_EPISODES}" \
  COLLECT_WORKERS="${COLLECT_WORKERS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  FINAL_EVAL_WORKERS=1 \
  ITERS_PER_BLOCK="${ITERS_PER_BLOCK}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS}" \
  UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE}" \
  PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
  KL_COEF="${KL_COEF}" \
  VF_COEF="${VF_COEF}" \
  TRAINABLE_SCOPE="${TRAINABLE_SCOPE}" \
  LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE}" \
  LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K}" \
  LOSS_FOCUS_CONTEXT_LEN="${LOSS_FOCUS_CONTEXT_LEN}" \
  LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT}" \
  COLLECT_VIDEO_MODE=skip \
  BASELINE_VIDEO_MODE=skip \
  EVAL_VIDEO_MODE=skip \
  FINAL_EVAL_VIDEO_MODE=skip \
  SKIP_VIDEO=1 \
  bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
}

run_final_probe() {
  local variant="$1"
  local seed="$2"
  local seed_tag
  seed_tag="$(printf 'seed_%03d' "${seed}")"
  local chain_root="${MASTER_ROOT}/training/${variant}/${seed_tag}"
  local chain_summary="${chain_root}/chain_summary.json"
  if [[ ! -f "${chain_summary}" ]]; then
    echo "[h-heading-learning] missing chain summary for final probe: ${chain_summary}" >&2
    return 1
  fi

  local model_path
  model_path="$("${PYTHON_BIN}" - <<'PY' "${chain_summary}"
import json
import sys
from pathlib import Path

chain = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocks = chain.get("blocks") or []
if not blocks:
    raise SystemExit("empty chain")
print(blocks[-1].get("next_model_path") or "")
PY
)"
  if [[ -z "${model_path}" || ! -f "${model_path}" ]]; then
    echo "[h-heading-learning] invalid final model path for ${variant}/${seed_tag}: ${model_path}" >&2
    return 1
  fi

  local probe_root="${MASTER_ROOT}/final_probe/${variant}/${seed_tag}"
  local trained_root="${probe_root}/trained_eval"
  local trained_summary="${trained_root}/trained_eval_bank_${FINAL_PROBE_EPISODES}ep/summary.json"

  MASTER_ROOT="${trained_root}" \
  ASSET_DIR="${ASSET_DIR}" \
  BANK_NAME=eval_bank \
  MODEL_PATH="${model_path}" \
  EPISODES="${FINAL_PROBE_EPISODES}" \
  WORLD_WORKERS="${PROBE_WORKERS}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  BASE_SEED="${BASE_SEED:-1}" \
  SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}" \
  EPISODE_RETRIES="${EPISODE_RETRIES:-2}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  SKIP_VIDEO=1 \
  RESULT_TAG="trained_eval_bank_${FINAL_PROBE_EPISODES}ep" \
  bash "${ROOT_DIR}/scripts/run_h_heading_offset_calibration_local.sh"

  "${PYTHON_BIN}" - <<'PY' "${trained_summary}" "${BASELINE_SUMMARY}" "${probe_root}/trained_vs_baseline.md" "${variant}" "${seed_tag}"
import json
import sys
from pathlib import Path

trained_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
variant = sys.argv[4]
seed_tag = sys.argv[5]

trained = json.loads(trained_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {"variants": []}

base_by_idx = {int(row["instance_idx"]): row for row in baseline.get("variants") or []}
trained_rows = sorted(trained.get("variants") or [], key=lambda row: int(row.get("instance_idx", 0)))

def pct(success, eps):
    return "" if not eps else f"{100.0 * float(success) / float(eps):.1f}%"

base_total_eps = sum(int(row.get("episodes") or 0) for row in base_by_idx.values())
base_total_success = sum(int(row.get("successful_episodes") or 0) for row in base_by_idx.values())
trained_total_eps = sum(int(row.get("episodes") or 0) for row in trained_rows)
trained_total_success = sum(int(row.get("successful_episodes") or 0) for row in trained_rows)

lines = [
    f"# H Heading Offset Trained vs Baseline Probe: {variant}/{seed_tag}",
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
    label = str(row.get("variant_id") or base.get("variant_id") or f"instance_{idx:03d}")
    b_eps = int(base.get("episodes") or 0)
    b_succ = int(base.get("successful_episodes") or 0)
    t_eps = int(row.get("episodes") or 0)
    t_succ = int(row.get("successful_episodes") or 0)
    b_rate = b_succ / b_eps if b_eps else None
    t_rate = t_succ / t_eps if t_eps else None
    delta = "" if b_rate is None or t_rate is None else f"{100.0 * (t_rate - b_rate):+.1f}"
    lines.append(
        f"| {idx} | {label} | {b_succ}/{b_eps} ({pct(b_succ, b_eps)}) | "
        f"{t_succ}/{t_eps} ({pct(t_succ, t_eps)}) | {delta} |"
    )

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path)
PY

  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/render_h_heading_offset_report.py" \
    --calibration-summary "${BASELINE_SUMMARY}" \
    --chain-summary "${chain_summary}" \
    --final-probe-md "${probe_root}/trained_vs_baseline.md" \
    --out-path "${probe_root}/h_heading_summary.md"
}

cat > "${MASTER_ROOT}/README.txt" <<EOF
H heading offset learning signal suite

asset_dir=${ASSET_DIR}
baseline_summary=${BASELINE_SUMMARY}
goal_protocol=${GOAL_PROTOCOL}
goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}
plan=${PLAN}
base_seeds=${BASE_SEEDS}
collect_episodes=${COLLECT_EPISODES}
eval_episodes=${EVAL_EPISODES}
iters_per_block=${ITERS_PER_BLOCK}
final_probe_episodes=${FINAL_PROBE_EPISODES}
loss_focus_mode=${LOSS_FOCUS_MODE}
trainable_scope=${TRAINABLE_SCOPE}
vf_coef=${VF_COEF}
kl_coef=${KL_COEF}

Collect block map:
$(resolve_collect_block_map_md)
EOF

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

is_known_variant() {
  case "$1" in
    focused|single_straight_H90|no_update_focused|smoke) return 0 ;;
    *) return 1 ;;
  esac
}

VARIANTS="${VARIANTS:-${PLAN}}"
split_csv "${VARIANTS}"
VARIANT_ITEMS=( "${SPLIT_ITEMS[@]}" )
if (( ${#VARIANT_ITEMS[@]} == 0 )); then
  VARIANT_ITEMS=( "${PLAN}" )
fi
for variant in "${VARIANT_ITEMS[@]}"; do
  if ! is_known_variant "${variant}"; then
    echo "Unknown variant=${variant}" >&2
    exit 1
  fi
done
split_csv "${BASE_SEEDS}"
SEED_ITEMS=( "${SPLIT_ITEMS[@]}" )

log "master_root=${MASTER_ROOT}"
log "asset_dir=${ASSET_DIR}"
log "baseline_summary=${BASELINE_SUMMARY}"
log "plan=${PLAN}"
log "variants=${VARIANTS}"

for variant in "${VARIANT_ITEMS[@]}"; do
  case "${variant}" in
    focused|single_straight_H90|no_update_focused|smoke) ;;
    *) echo "Unknown variant=${variant}" >&2; exit 1 ;;
  esac
  for seed in "${SEED_ITEMS[@]}"; do
    run_chain "${variant}" "${seed}"
    if [[ "${RUN_FINAL_PROBES}" == "1" ]]; then
      run_final_probe "${variant}" "${seed}"
    fi
  done
done

log "done master_root=${MASTER_ROOT}"
