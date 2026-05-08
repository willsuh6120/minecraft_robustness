#!/usr/bin/env bash

resolve_minestudio_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    return 0
  fi
  if [[ -x "${HOME}/miniconda3/envs/minestudio/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return 0
  fi
  PYTHON_BIN="$(command -v python)"
}

resolve_conda_prefix_from_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    local candidate
    candidate="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
    if [[ -d "${candidate}" && -d "${candidate}/lib" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}" ]]; then
    printf '%s\n' "${CONDA_PREFIX}"
    return 0
  fi
  printf '%s\n' ""
}

default_minestudio_cache_dir() {
  if [[ -n "${ROOT_DIR:-}" ]]; then
    local workspace_root
    workspace_root="$(cd "${ROOT_DIR}/.." && pwd)"
    printf '%s\n' "${workspace_root}/.minestudio"
    return 0
  fi
  if [[ -d "${HOME}/envgen2/.minestudio" ]]; then
    printf '%s\n' "${HOME}/envgen2/.minestudio"
    return 0
  fi
  printf '%s\n' "${HOME}/.minestudio"
}

configure_torch_cuda_runtime() {
  local conda_prefix
  conda_prefix="$(resolve_conda_prefix_from_python_bin)"
  if [[ -z "${conda_prefix}" ]]; then
    return 0
  fi
  local python_version
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_pkg="${conda_prefix}/lib/python${python_version}/site-packages/nvidia"
  local lib_paths=()
  local subdir
  for subdir in \
    cublas/lib \
    cudnn/lib \
    cuda_runtime/lib \
    cuda_nvrtc/lib \
    cufft/lib \
    curand/lib \
    cusolver/lib \
    cusparse/lib \
    cusparselt/lib \
    nccl/lib \
    nvjitlink/lib \
    nvtx/lib; do
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

setup_minestudio_runtime() {
  resolve_minestudio_python
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export MINESTUDIO_DIR="${MINESTUDIO_DIR:-$(default_minestudio_cache_dir)}"
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
  unset LD_PRELOAD || true
  configure_torch_cuda_runtime
}

require_xvfb_run() {
  XVFB_RUN="${XVFB_RUN:-$(command -v xvfb-run || true)}"
  XVFB_SCREEN_ARGS="${XVFB_SCREEN_ARGS:--screen 0 1920x1200x24 -dpi 72 +extension RANDR +extension GLX +iglx +extension MIT-SHM +render -nolisten tcp -noreset}"
  if [[ -z "${XVFB_RUN}" ]]; then
    echo "xvfb-run not found. Install xvfb first." >&2
    return 1
  fi
}

ensure_minestudio_engine() {
  resolve_minestudio_python
  export MINESTUDIO_DIR="${MINESTUDIO_DIR:-$(default_minestudio_cache_dir)}"
  "${PYTHON_BIN}" - <<'PY'
from minestudio.simulator.entry import check_engine
check_engine(skip_confirmation=True)
print("[minestudio-runtime] engine_ready=1")
PY
}
