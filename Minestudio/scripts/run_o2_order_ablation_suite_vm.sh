#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK="${TASK:-mine_coal}"
BASE_SEEDS="${BASE_SEEDS:-1}"
VARIANTS="${VARIANTS:-forward,reverse,random_a,best_single,bad_single,no_update}"
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL:-0}"
SUITE_TAG="${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/o2_order_ablation_suite/${SUITE_TAG}}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal}"
GOAL_SPEC="${GOAL_SPEC:-}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing. Set ASSET_DIR to the O2 asset directory on the VM." >&2
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
    forward) printf '%s\n' "1,2,3,4,5" ;;
    reverse) printf '%s\n' "5,4,3,2,1" ;;
    random_a) printf '%s\n' "2,5,1,4,3" ;;
    random_b) printf '%s\n' "3,1,5,2,4" ;;
    random_c) printf '%s\n' "4,2,5,1,3" ;;
    best_single) printf '%s\n' "5,5,5,5,5" ;;
    bad_single) printf '%s\n' "4,4,4,4,4" ;;
    no_update) printf '%s\n' "1,2,3,4,5" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_disable_updates() {
  case "$1" in
    no_update) printf '%s\n' "1" ;;
    *) printf '%s\n' "0" ;;
  esac
}

mkdir -p "${SUITE_ROOT}"

echo "[o2-order-suite] suite_root=${SUITE_ROOT}"
echo "[o2-order-suite] asset_dir=${ASSET_DIR}"
echo "[o2-order-suite] variants=${VARIANTS}"
echo "[o2-order-suite] base_seeds=${BASE_SEEDS}"
echo "[o2-order-suite] skip_all_final_eval=${SKIP_ALL_FINAL_EVAL}"
echo "[o2-order-suite] goal_protocol=${GOAL_PROTOCOL}"
echo "[o2-order-suite] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"

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
    echo "[o2-order-suite] order=${order} disable_updates=${disable_updates} out_root=${out_root}"
    TASK="${TASK}" \
    ASSET_DIR="${ASSET_DIR}" \
    OUT_ROOT="${out_root}" \
    EXPERIMENT_TAG="${variant}_${seed_tag}" \
    BLOCK_ORDER="${order}" \
    DISABLE_UPDATES="${disable_updates}" \
    SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL}" \
    GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
    GOAL_SPEC="${GOAL_SPEC}" \
    BASE_SEED="${seed}" \
    SAMPLING_BASE_SEED="${seed}" \
    BAKE_GOAL_BASE_SEED="${seed}" \
    COLLECT_VIDEO_MODE=skip \
    BASELINE_VIDEO_MODE=skip \
    EVAL_VIDEO_MODE=skip \
    FINAL_EVAL_VIDEO_MODE=skip \
    SKIP_VIDEO=1 \
    bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
  done
done

echo "[o2-order-suite] done suite_root=${SUITE_ROOT}"
