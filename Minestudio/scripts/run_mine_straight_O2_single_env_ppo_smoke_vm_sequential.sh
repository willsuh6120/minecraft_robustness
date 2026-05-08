#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_RUN_SCRIPT="${BASE_RUN_SCRIPT:-${ROOT_DIR}/scripts/run_mine_straight_O2_single_env_ppo_smoke.sh}"

export COLLECT_WORKERS="${COLLECT_WORKERS:-1}"
export EVAL_WORKERS="${EVAL_WORKERS:-1}"
export FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-1}"

bash "${BASE_RUN_SCRIPT}"

