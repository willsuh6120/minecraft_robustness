#!/usr/bin/env bash
# advantage_top_k focus mode — local overnight run (minimal plan, tighter k)
# Hypothesis: k=16 (half of suffix baseline) forces the model to concentrate
# on fewer but higher-confidence pivotal steps, which may be better for path decisions.
#
# Meaningful differences vs VM run:
#   - PLAN=minimal (warm_medium + single_left_funnel only — lighter, faster feedback)
#   - LOSS_FOCUS_TOP_K=16  (tighter focus: only the top-16 advantage steps per episode)
#   - TRAINABLE_SCOPE=heads (pi + value only — less risk of overfitting on short runs)
#   - PPO_LEARNING_RATE=2e-5 (same as current baseline for easy comparison)
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOSS_FOCUS_MODE=advantage_top_k \
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
RUN_TAG="p_obstacle_maze_t6_adv_top_k16_minimal_$(date +%Y%m%d_%H%M%S)" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_learning_signal_local.sh"
