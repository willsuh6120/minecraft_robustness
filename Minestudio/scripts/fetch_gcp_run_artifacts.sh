#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GCLOUD="${GCLOUD:-$(command -v gcloud)}"

INSTANCE_NAME="${INSTANCE_NAME:-}"
ZONE="${ZONE:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_TMP_DIR="${REMOTE_TMP_DIR:-/tmp}"
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT:-${ROOT}/outputs/imported_vm_runs}"
OVERWRITE="${OVERWRITE:-0}"
FETCH_MODE="${FETCH_MODE:-full_no_pt}"

usage() {
  cat <<'EOF'
usage:
  INSTANCE_NAME=<vm> ZONE=<zone> REMOTE_USER=<user> \
    bash scripts/fetch_gcp_run_artifacts.sh <remote_run_dir> [local_dest_dir]

example:
  INSTANCE_NAME=instance-20260417-163859 \
  ZONE=asia-southeast1-b \
  REMOTE_USER=willsuh1114 \
  bash scripts/fetch_gcp_run_artifacts.sh \
    /home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_straight_O2_single_env_smoke_stable_vm_20260425_134003

notes:
  - full_no_pt: copies the run directory contents except *.pt and *.pt.*
  - metrics_only: copies only metric/summary files needed for experiment analysis
  - metrics_with_worldgen_refs: metrics_only + referenced worldgen/plan/yaml files
    discovered from pilot_state.json / pilot_summary.json
  - default local destination:
      ${ROOT}/outputs/imported_vm_runs/<basename(remote_run_dir)>
  - set OVERWRITE=1 to replace an existing local destination

examples:
  FETCH_MODE=metrics_only INSTANCE_NAME=... ZONE=... REMOTE_USER=... \
    bash scripts/fetch_gcp_run_artifacts.sh <remote_run_dir>
EOF
}

log() {
  echo "[fetch-gcp-run-artifacts] $*"
}

cleanup_tmp_dir() {
  local dir="${1:-}"
  [[ -n "${dir}" ]] || return 0
  rm -rf -- "${dir}"
}

format_seconds() {
  local total
  total="$(printf '%.0f' "${1:-0}")"
  if (( total < 60 )); then
    printf '%ss' "${total}"
    return
  fi
  local minutes=$(( total / 60 ))
  local seconds=$(( total % 60 ))
  if (( minutes < 60 )); then
    printf '%sm%02ds' "${minutes}" "${seconds}"
    return
  fi
  local hours=$(( minutes / 60 ))
  minutes=$(( minutes % 60 ))
  printf '%sh%02dm%02ds' "${hours}" "${minutes}" "${seconds}"
}

require_gcloud_config() {
  if [[ -z "${GCLOUD}" ]]; then
    echo "gcloud not found in PATH." >&2
    exit 1
  fi
  if [[ -z "${INSTANCE_NAME}" || -z "${ZONE}" || -z "${REMOTE_USER}" ]]; then
    echo "Required env vars: INSTANCE_NAME, ZONE, REMOTE_USER" >&2
    exit 1
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  require_gcloud_config

  local remote_run_dir="${1:-}"
  local local_dest_dir="${2:-}"
  if [[ -z "${remote_run_dir}" ]]; then
    usage >&2
    exit 1
  fi

  remote_run_dir="${remote_run_dir%/}"
  if [[ -z "${local_dest_dir}" ]]; then
    local_dest_dir="${LOCAL_IMPORT_ROOT}/$(basename "${remote_run_dir}")"
  fi

  if [[ -e "${local_dest_dir}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
      echo "Local destination already exists: ${local_dest_dir}" >&2
      echo "Set OVERWRITE=1 to replace it, or pass a different local_dest_dir." >&2
      exit 1
    fi
    rm -rf -- "${local_dest_dir}"
  fi
  mkdir -p "${local_dest_dir}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap "cleanup_tmp_dir '${tmp_dir}'" EXIT

  local run_tag
  run_tag="$(date +%Y%m%d_%H%M%S)_$$"
  local remote_archive="${REMOTE_TMP_DIR}/run_artifacts_${run_tag}.tar.gz"
  local remote_manifest="${REMOTE_TMP_DIR}/run_artifacts_${run_tag}.manifest.txt"
  local local_archive="${tmp_dir}/run_artifacts_${run_tag}.tar.gz"
  local remote_pack_cmd

  case "${FETCH_MODE}" in
    full_no_pt)
      remote_pack_cmd="set -euo pipefail; test -d '${remote_run_dir}'; tar -czf '${remote_archive}' --exclude='*.pt' --exclude='*.pt.*' -C '${remote_run_dir}' ."
      ;;
    metrics_only)
      remote_pack_cmd="set -euo pipefail; test -d '${remote_run_dir}'; cd '${remote_run_dir}'; find . -type f \\( -name 'chain_summary.json' -o -name 'suite_summary.json' -o -name 'suite_summary.md' -o -name 'pilot_summary.json' -o -name 'pilot_state.json' -o -name 'summary.json' -o -name 'summary.csv' -o -name 'episodes.csv' -o -name 'run_metadata.json' -o -name 'train_metadata.json' -o -name 'train_history.json' -o -name 'sanity_preupdate.json' \\) | sort > '${remote_manifest}'; if [[ ! -s '${remote_manifest}' ]]; then echo 'No metric files found under ${remote_run_dir}' >&2; exit 1; fi; tar -czf '${remote_archive}' -T '${remote_manifest}'; rm -f '${remote_manifest}'"
      ;;
    metrics_with_worldgen_refs)
      remote_pack_cmd="set -euo pipefail; test -d '${remote_run_dir}'; cd '${remote_run_dir}'; find . -type f \\( -name 'chain_summary.json' -o -name 'suite_summary.json' -o -name 'suite_summary.md' -o -name 'pilot_summary.json' -o -name 'pilot_state.json' -o -name 'summary.json' -o -name 'summary.csv' -o -name 'episodes.csv' -o -name 'run_metadata.json' -o -name 'train_metadata.json' -o -name 'train_history.json' -o -name 'sanity_preupdate.json' \\) | sort > '${remote_manifest}'; if [[ ! -s '${remote_manifest}' ]]; then echo 'No metric files found under ${remote_run_dir}' >&2; exit 1; fi; refs_raw='${REMOTE_TMP_DIR}/run_refs_${run_tag}.raw.txt'; refs_files='${REMOTE_TMP_DIR}/run_refs_${run_tag}.files.txt'; refs_tar='${REMOTE_TMP_DIR}/run_refs_${run_tag}.tarlist.txt'; : > \"\${refs_raw}\"; : > \"\${refs_files}\"; find . -type f \\( -name 'pilot_summary.json' -o -name 'pilot_state.json' \\) -print0 | xargs -0 -r grep -hoE '\"/home/willsuh1114/envgen2/Minestudio[^\"]*\"' | tr -d '\"' | sort -u > \"\${refs_raw}\" || true; while IFS= read -r ref_path; do [[ -n \"\${ref_path}\" ]] || continue; if [[ -d \"\${ref_path}\" ]]; then find \"\${ref_path}\" -maxdepth 1 -type f \\( -name '*.yaml' -o -name '*.yml' \\) >> \"\${refs_files}\"; continue; fi; if [[ -f \"\${ref_path}\" ]]; then case \"\${ref_path}\" in */worldgen_manifest.json|*/bank_manifest.json|*/worldgen_plan.json|*/collect_plan.json|*/block_*_collect_plan.json|*.yaml|*.yml) echo \"\${ref_path}\" >> \"\${refs_files}\" ;; esac; fi; done < \"\${refs_raw}\"; cat '${remote_manifest}' \"\${refs_files}\" | sort -u > \"\${refs_tar}\"; tar -czf '${remote_archive}' -T \"\${refs_tar}\"; rm -f '${remote_manifest}' \"\${refs_raw}\" \"\${refs_files}\" \"\${refs_tar}\""
      ;;
    *)
      echo "Unsupported FETCH_MODE: ${FETCH_MODE}" >&2
      exit 1
      ;;
  esac

  log "instance=${INSTANCE_NAME} zone=${ZONE} remote_user=${REMOTE_USER}"
  log "remote_run_dir=${remote_run_dir}"
  log "local_dest_dir=${local_dest_dir}"
  log "fetch_mode=${FETCH_MODE}"

  local started_at
  started_at="$(date +%s)"

  "${GCLOUD}" compute ssh \
    "${REMOTE_USER}@${INSTANCE_NAME}" \
    --zone "${ZONE}" \
    --command "${remote_pack_cmd}"

  "${GCLOUD}" compute scp \
    "${REMOTE_USER}@${INSTANCE_NAME}:${remote_archive}" \
    "${local_archive}" \
    --zone "${ZONE}"

  "${GCLOUD}" compute ssh \
    "${REMOTE_USER}@${INSTANCE_NAME}" \
    --zone "${ZONE}" \
    --command "rm -f '${remote_archive}'"

  tar -xzf "${local_archive}" -C "${local_dest_dir}"
  cat > "${local_dest_dir}/REMOTE_SOURCE.txt" <<EOF
instance=${INSTANCE_NAME}
zone=${ZONE}
remote_user=${REMOTE_USER}
remote_run_dir=${remote_run_dir}
fetched_at=$(date -Iseconds)
fetch_mode=${FETCH_MODE}
excluded_globs=*.pt,*.pt.*
EOF

  local ended_at
  ended_at="$(date +%s)"
  local elapsed=$(( ended_at - started_at ))

  log "done elapsed_sec=${elapsed} elapsed_human=$(format_seconds "${elapsed}")"
  log "local_dest_dir=${local_dest_dir}"
  log "tip=analyze this local path directly; REMOTE_SOURCE.txt records the original VM path."
}

main "$@"
