#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GCLOUD="${GCLOUD:-$(command -v gcloud)}"

INSTANCE_NAME="${INSTANCE_NAME:-}"
ZONE="${ZONE:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-}"
REMOTE_TMP_DIR="${REMOTE_TMP_DIR:-/tmp}"

GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
LOCAL_GOAL_DIR="${LOCAL_GOAL_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/fixed_eval_bank_mines_goalvis_v4/20260419_025333/train_id/instance_000/generated_task_groups/20260419_025333/_baked_goals/mine_coal/protocol_ours_v1/seed_000001}"

if [[ -z "${GCLOUD}" ]]; then
  echo "gcloud not found in PATH." >&2
  exit 1
fi
if [[ -z "${INSTANCE_NAME}" || -z "${ZONE}" || -z "${REMOTE_USER}" ]]; then
  echo "Required env vars: INSTANCE_NAME, ZONE, REMOTE_USER" >&2
  exit 1
fi
if [[ -z "${REMOTE_REPO_ROOT}" ]]; then
  REMOTE_REPO_ROOT="/home/${REMOTE_USER}/envgen2/Minestudio"
fi
REMOTE_GOAL_DIR="${REMOTE_GOAL_DIR:-${REMOTE_REPO_ROOT}/outputs/evaluate_rocket/fixed_goal_banks/mine_coal/${GOAL_PROTOCOL}}"

if [[ ! -d "${LOCAL_GOAL_DIR}" ]]; then
  echo "LOCAL_GOAL_DIR does not exist: ${LOCAL_GOAL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_GOAL_DIR}/goal_image.png" || ! -f "${LOCAL_GOAL_DIR}/goal_mask.png" ]]; then
  echo "LOCAL_GOAL_DIR must contain goal_image.png and goal_mask.png: ${LOCAL_GOAL_DIR}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

STAGE_DIR="${TMP_DIR}/${GOAL_PROTOCOL}"
mkdir -p "${STAGE_DIR}"
for name in goal_image.png goal_mask.png goal_mask_overlay.png goal_bbox_overlay.png; do
  if [[ -f "${LOCAL_GOAL_DIR}/${name}" ]]; then
    cp "${LOCAL_GOAL_DIR}/${name}" "${STAGE_DIR}/${name}"
  fi
done

python - <<'PY' "${LOCAL_GOAL_DIR}" "${STAGE_DIR}" "${REMOTE_GOAL_DIR}" "${GOAL_PROTOCOL}"
import hashlib
import json
import sys
from pathlib import Path

local_goal_dir = Path(sys.argv[1])
stage_dir = Path(sys.argv[2])
remote_goal_dir = sys.argv[3].rstrip("/")
goal_protocol = sys.argv[4]

src_spec_path = local_goal_dir / "goal_spec.json"
if src_spec_path.exists():
    spec = json.loads(src_spec_path.read_text(encoding="utf-8"))
else:
    spec = {}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

spec.update(
    {
        "goal_protocol": goal_protocol,
        "goal_protocol_fixed": True,
        "goal_protocol_notes": "Fixed clean goal prompt for world-factor robustness experiments.",
        "goal_image_path": f"{remote_goal_dir}/goal_image.png",
        "goal_mask_path": f"{remote_goal_dir}/goal_mask.png",
        "segment_type": str(spec.get("segment_type") or "Mine"),
        "goal_source_world": "clean_O0_P0_H0_C0",
        "goal_image_sha256": sha256_file(stage_dir / "goal_image.png"),
        "goal_mask_sha256": sha256_file(stage_dir / "goal_mask.png"),
    }
)
if (stage_dir / "goal_mask_overlay.png").exists():
    spec["goal_mask_overlay_path"] = f"{remote_goal_dir}/goal_mask_overlay.png"
if (stage_dir / "goal_bbox_overlay.png").exists():
    spec["goal_bbox_overlay_path"] = f"{remote_goal_dir}/goal_bbox_overlay.png"

(stage_dir / "goal_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"goal_protocol": goal_protocol, "remote_goal_spec": f"{remote_goal_dir}/goal_spec.json"}, ensure_ascii=False))
PY

echo "[sync-fixed-goal] local_goal_dir=${LOCAL_GOAL_DIR}"
echo "[sync-fixed-goal] remote_goal_dir=${REMOTE_GOAL_DIR}"
"${GCLOUD}" compute ssh \
  "${REMOTE_USER}@${INSTANCE_NAME}" \
  --zone "${ZONE}" \
  --command "mkdir -p '${REMOTE_GOAL_DIR}'"

"${GCLOUD}" compute scp \
  "${STAGE_DIR}"/* \
  "${REMOTE_USER}@${INSTANCE_NAME}:${REMOTE_GOAL_DIR}/" \
  --zone "${ZONE}"

echo "[sync-fixed-goal] done"
echo "[sync-fixed-goal] GOAL_SPEC=${REMOTE_GOAL_DIR}/goal_spec.json"
