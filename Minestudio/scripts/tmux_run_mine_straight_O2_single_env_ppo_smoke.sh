#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${ROOT_DIR}/scripts/run_mine_straight_O2_single_env_ppo_smoke.sh}"
SESSION_NAME="${SESSION_NAME:-ppo_straight_O2_smoke}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/ppo_straight_O2_single_env_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${OUT_ROOT}/tmux.log}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
DEFAULT_PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found in PATH." >&2
  exit 1
fi

if [[ ! -x "${RUN_SCRIPT}" && ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

LAUNCH_BODY="$(cat <<EOF
set -euo pipefail
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate minestudio || true
fi
cd "${ROOT_DIR}"
export OUT_ROOT="${OUT_ROOT}"
export PYTHON_BIN="${PYTHON_BIN}"
bash "${RUN_SCRIPT}" 2>&1 | tee "${LOG_PATH}"
EOF
)"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${LAUNCH_BODY}")"

echo "session=${SESSION_NAME}"
echo "out_root=${OUT_ROOT}"
echo "log_path=${LOG_PATH}"
echo "attach_cmd=tmux attach -t ${SESSION_NAME}"
echo "tail_cmd=tail -f ${LOG_PATH}"

