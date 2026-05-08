#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
XVFB_RUN="${XVFB_RUN:-$(command -v xvfb-run)}"
XVFB_SCREEN_ARGS="${XVFB_SCREEN_ARGS:--screen 0 1280x1024x24}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "${XVFB_RUN}" ]]; then
  echo "xvfb-run not found. Install xvfb first." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "minestudio" ]]; then
  echo "minestudio conda env is not active. Current env: '${CONDA_DEFAULT_ENV:-<none>}'" >&2
  echo "Run: conda activate minestudio" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import yaml
import torch
import minestudio
PY
then
  echo "Active Python does not have required packages (yaml/torch/minestudio)." >&2
  echo "PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "Run inside the minestudio env or override PYTHON_BIN explicitly." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

default_minestudio_dir="${HOME}/envgen2/.minestudio"
default_hf_home="${HOME}/.cache/huggingface"

if [[ -n "${MINESTUDIO_DIR:-}" && "${MINESTUDIO_DIR}" != "${default_minestudio_dir}" ]]; then
  if [[ ! -d "${MINESTUDIO_DIR}" && ! -w "$(dirname "${MINESTUDIO_DIR}")" ]]; then
    echo "[warn] overriding foreign/unwritable MINESTUDIO_DIR=${MINESTUDIO_DIR}" >&2
    unset MINESTUDIO_DIR
  fi
fi
if [[ -n "${HF_HOME:-}" && "${HF_HOME}" != "${default_hf_home}" ]]; then
  if [[ ! -d "${HF_HOME}" && ! -w "$(dirname "${HF_HOME}")" ]]; then
    echo "[warn] overriding foreign/unwritable HF_HOME=${HF_HOME}" >&2
    unset HF_HOME
  fi
fi

export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${default_minestudio_dir}}"
export HF_HOME="${HF_HOME:-${default_hf_home}}"

export LAYOUT_CASES_CSV="${LAYOUT_CASES_CSV:-offset_chamber_left}"
export FACTOR_CODES_CSV="${FACTOR_CODES_CSV:-R,H,O,C,P,A}"
export PROBE_LEVEL="${PROBE_LEVEL:-2}"
export ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-2}"
export STEP_BUDGET="${STEP_BUDGET:-150}"
export EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
export INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
export USE_XVFB="${USE_XVFB:-1}"
export OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/evaluate_rocket/mine_vm_offset_chamber_factor_probes_$(date +%Y%m%d_%H%M%S)}"

echo "[config] ROOT=${ROOT}"
echo "[config] PYTHON_BIN=${PYTHON_BIN}"
echo "[config] OUT_ROOT=${OUT_ROOT}"
echo "[config] LAYOUT_CASES_CSV=${LAYOUT_CASES_CSV}"
echo "[config] FACTOR_CODES_CSV=${FACTOR_CODES_CSV}"
echo "[config] PROBE_LEVEL=${PROBE_LEVEL}"
echo "[config] INCLUDE_BASELINE=${INCLUDE_BASELINE}"
echo "[config] ROLLOUT_EPISODES=${ROLLOUT_EPISODES}"
echo "[config] STEP_BUDGET=${STEP_BUDGET}"

exec bash "${ROOT}/scripts/run_mine_single_factor_layout_probes.sh"
