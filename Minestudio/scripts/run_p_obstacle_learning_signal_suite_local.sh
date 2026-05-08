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
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
GOAL_SPEC="${GOAL_SPEC:-${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/${TASK}/${GOAL_PROTOCOL}/goal_spec.json}"
GOAL_OVERWRITE="${GOAL_OVERWRITE:-0}"
GOAL_CAMERA_HORIZONTAL_DISTANCE="${GOAL_CAMERA_HORIZONTAL_DISTANCE:-2.25}"

SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
ASSET_DIR="${ASSET_DIR:-}"
ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_learning_assets}"
GENERATE_ASSETS="${GENERATE_ASSETS:-auto}"
EVAL_INSTANCES="${EVAL_INSTANCES:-8}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BASE_SEEDS="${BASE_SEEDS:-1}"
VARIANTS="${VARIANTS:-teacher_to_hard,no_update_teacher_to_hard}"
SUITE_TAG="${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_learning_signal_suite/${SUITE_TAG}}"

COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-1}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"

if [[ "${GOAL_OVERWRITE}" == "1" || ! -f "${GOAL_SPEC}" ]]; then
  echo "[p-learning-suite] ensuring fixed goal bank goal_spec=${GOAL_SPEC} overwrite=${GOAL_OVERWRITE}"
  TASK="${TASK}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_BANK_ROOT="$(dirname "${GOAL_SPEC}")" \
  GOAL_CAMERA_HORIZONTAL_DISTANCE="${GOAL_CAMERA_HORIZONTAL_DISTANCE}" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  OVERWRITE="${GOAL_OVERWRITE}" \
  bash "${ROOT_DIR}/scripts/ensure_fixed_clean_goal_bank.sh"
fi

if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ "${GENERATE_ASSETS}" == "1" || ( "${GENERATE_ASSETS}" == "auto" && ( -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ) ) ]]; then
  mkdir -p "${ASSET_ROOT}"
  echo "[p-learning-suite] generating P obstacle assets asset_root=${ASSET_ROOT}"
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  FINAL_INSTANCES=0 \
  SKIP_FINAL_BANK=1 \
  BAKE_GOALS=0 \
  BASE_SEED=1 \
  BANK_WORKERS="${BANK_WORKERS}" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  OUT_DIR="${ASSET_ROOT}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_straight_obstacle_assets.sh"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${ASSET_DIR}/eval_bank" || ! -d "${ASSET_DIR}/collect_block_plans" ]]; then
  echo "ASSET_DIR does not look like a P obstacle asset bank: ${ASSET_DIR}" >&2
  exit 1
fi

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

variant_order() {
  case "$1" in
    # Baseline probe suggests right_gate/right_wall/bar_z2 are teachable, while
    # left_zigzag/left_wall are medium and side_posts/left_gate/right_zigzag are hard.
    teacher_to_hard) printf '%s\n' "4,1,8,5,7,2,3,6" ;;
    reverse) printf '%s\n' "6,3,2,7,5,8,1,4" ;;
    hard_to_teacher) printf '%s\n' "2,3,6,5,7,4,1,8" ;;
    all_forward) printf '%s\n' "1,2,3,4,5,6,7,8" ;;
    best_single) printf '%s\n' "4,4,4,4,4,4,4,4" ;;
    medium_only) printf '%s\n' "5,7,5,7,5,7,5,7" ;;
    hard_single) printf '%s\n' "2,2,2,2,2,2,2,2" ;;
    no_update_teacher_to_hard) printf '%s\n' "4,1,8,5,7,2,3,6" ;;
    no_update_all_forward) printf '%s\n' "1,2,3,4,5,6,7,8" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_disable_updates() {
  case "$1" in
    no_update_*) printf '%s\n' "1" ;;
    *) printf '%s\n' "0" ;;
  esac
}

mkdir -p "${SUITE_ROOT}"
cat > "${SUITE_ROOT}/README.txt" <<EOF
P obstacle learning signal suite

asset_dir=${ASSET_DIR}
goal_spec=${GOAL_SPEC}
variants=${VARIANTS}
base_seeds=${BASE_SEEDS}
collect_episodes=${COLLECT_EPISODES}
eval_episodes=${EVAL_EPISODES}
iters_per_block=${ITERS_PER_BLOCK}
collect_workers=${COLLECT_WORKERS}
eval_workers=${EVAL_WORKERS}

Block ids:
1 bar_z2_full
2 side_posts_z3
3 left_gate
4 right_gate
5 left_zigzag
6 right_zigzag
7 left_wall
8 right_wall
EOF

echo "[p-learning-suite] suite_root=${SUITE_ROOT}"
echo "[p-learning-suite] asset_dir=${ASSET_DIR}"
echo "[p-learning-suite] goal_spec=${GOAL_SPEC}"
echo "[p-learning-suite] variants=${VARIANTS}"
echo "[p-learning-suite] base_seeds=${BASE_SEEDS}"
echo "[p-learning-suite] iters_per_block=${ITERS_PER_BLOCK}"

split_csv "${VARIANTS}"
VARIANT_ITEMS=( "${SPLIT_ITEMS[@]}" )
split_csv "${BASE_SEEDS}"
SEED_ITEMS=( "${SPLIT_ITEMS[@]}" )

for variant in "${VARIANT_ITEMS[@]}"; do
  order="$(variant_order "${variant}")"
  disable_updates="$(variant_disable_updates "${variant}")"
  for seed in "${SEED_ITEMS[@]}"; do
    seed_tag="$(printf 'seed_%03d' "${seed}")"
    out_root="${SUITE_ROOT}/${variant}/${seed_tag}"
    mkdir -p "${out_root}"
    echo
    echo "==================== variant=${variant} ${seed_tag} ===================="
    echo "[p-learning-suite] order=${order} disable_updates=${disable_updates} out_root=${out_root}"
    TASK="${TASK}" \
    TASK_GROUP_NAME="mine_p_straight_obstacle_local_${variant}_${seed_tag}" \
    ASSET_DIR="${ASSET_DIR}" \
    OUT_ROOT="${out_root}" \
    EXPERIMENT_TAG="p_${variant}_${seed_tag}" \
    GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
    GOAL_SPEC="${GOAL_SPEC}" \
    BLOCK_COUNT=8 \
    BLOCK_ORDER="${order}" \
    DISABLE_UPDATES="${disable_updates}" \
    SKIP_ALL_FINAL_EVAL=1 \
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
    COLLECT_VIDEO_MODE=skip \
    BASELINE_VIDEO_MODE=skip \
    EVAL_VIDEO_MODE=skip \
    FINAL_EVAL_VIDEO_MODE=skip \
    SKIP_VIDEO=1 \
    bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
  done
done

"${PYTHON_BIN}" - <<'PY' "${SUITE_ROOT}"
import json
import sys
from pathlib import Path

suite_root = Path(sys.argv[1])
rows = []
for chain_path in sorted(suite_root.glob("*/seed_*")):
    summary_path = chain_path / "chain_summary.json"
    if not summary_path.exists():
        continue
    chain = json.loads(summary_path.read_text(encoding="utf-8"))
    blocks = chain.get("blocks") or []
    best_rate = None
    best_block = ""
    last_rate = None
    for block in blocks:
        best_eval = block.get("best_eval_selection") or {}
        rate = best_eval.get("success_rate")
        if rate is None:
            summary = best_eval.get("summary") if isinstance(best_eval.get("summary"), dict) else {}
            for value in summary.values():
                if isinstance(value, dict) and value.get("auto_success_rate") is not None:
                    rate = value.get("auto_success_rate")
                    break
        if rate is not None:
            last_rate = float(rate)
            if best_rate is None or float(rate) > best_rate:
                best_rate = float(rate)
                best_block = str(block.get("block_tag") or "")
    rows.append(
        {
            "chain": str(chain_path.relative_to(suite_root)),
            "blocks": len(blocks),
            "best_eval_success_rate": best_rate,
            "best_block": best_block,
            "last_eval_success_rate": last_rate,
            "final_model_path": blocks[-1].get("next_model_path", "") if blocks else "",
        }
    )

(suite_root / "suite_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
lines = ["| chain | blocks | best eval | best block | last eval |", "|---|---:|---:|---|---:|"]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{100.0 * float(value):.1f}%"
    lines.append(
        f"| {row['chain']} | {row['blocks']} | {fmt(row['best_eval_success_rate'])} | {row['best_block']} | {fmt(row['last_eval_success_rate'])} |"
    )
(suite_root / "suite_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[p-learning-suite] suite_summary={suite_root / 'suite_summary.md'}")
PY

echo "[p-learning-suite] done suite_root=${SUITE_ROOT}"
