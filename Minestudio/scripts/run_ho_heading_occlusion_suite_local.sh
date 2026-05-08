#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/ho_heading_occlusion_suite/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_ROOT}"

cat > "${OUT_ROOT}/README.txt" <<'EOF'
H+O heading-occlusion suite scaffold

Status:
- Phase 1 (H-only) is implemented in:
  - scripts/run_h_heading_offset_calibration_local.sh
  - scripts/run_h_heading_offset_learning_signal_local.sh
- Phase 2 (H+O) is intentionally kept separate and is not wired here yet.

Design note:
- Compose H-only heading variants with exact O2 occluder patterns after H-only calibration is stable.
- Do not mix theme/background changes here; that remains phase 3.
EOF

echo "[ho-heading-occlusion-suite] scaffold only"
echo "[ho-heading-occlusion-suite] out_root=${OUT_ROOT}"
