#!/usr/bin/env bash
# lateral_top_k focus mode — local overnight run (minimal plan, tighter k)
# Hypothesis: upweighting steps with the largest absolute change in lateral offset
# from the direct start→target line captures lane-commitment, funnel-alignment, and
# detour-entry moments — the actual path decisions in P-maze — rather than the
# straight final approach that raw distance-improvement would select.
# Ground-truth coordinate signal: no learned model needed, just trajectory.jsonl + goal_spec.json.
#
# Meaningful differences vs VM run:
#   - PLAN=minimal (warm_medium + single_left_funnel only — lighter, faster feedback)
#   - LOSS_FOCUS_TOP_K=16  (tighter focus: top-16 lateral-change steps)
#   - TRAINABLE_SCOPE=heads (pi + value only — less risk of overfitting on short runs)
#   - PPO_LEARNING_RATE=2e-5 (same as current baseline for easy comparison)
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOSS_FOCUS_MODE=lateral_top_k \
LOSS_FOCUS_TOP_K=16 \
LOSS_FOCUS_WEIGHT=4.0 \
PPO_LEARNING_RATE=2e-5 \
KL_COEF=0.1 \
PLAN=minimal \
TRAINABLE_SCOPE=heads \
COLLECT_EPISODES=64 \
EVAL_EPISODES=32 \
FINAL_PROBE_EPISODES=16 \
MIN_SUCCESSFUL_FRAGMENTS=4 \
UPDATE_FRAGMENT_BATCH_SIZE=4 \
ITERS_PER_BLOCK=1 \
STEP_BUDGET=150 \
RUN_TAG="p_obstacle_maze_t6_lateral_top_k16_minimal_$(date +%Y%m%d_%H%M%S)" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_learning_signal_local.sh"
