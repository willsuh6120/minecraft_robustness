#!/usr/bin/env bash
# advantage_top_k focus mode — VM overnight run (balanced plan, broader coverage)
# Hypothesis: weighting the top-k highest-advantage steps instead of a fixed suffix
# better captures the path-choice decisions that drove success.
#
# Meaningful differences vs local run:
#   - PLAN=balanced (4 variants: warm_medium + hardish_bridge + single_left_funnel + no_update_warm_medium)
#   - LOSS_FOCUS_TOP_K=32  (same k as current suffix_len baseline for apples-to-apples)
#   - TRAINABLE_SCOPE=crossview_small (larger trainable head: pi, value, lastlayer, final_ln, updim_cross)
#   - PPO_LEARNING_RATE=3e-5 (slightly higher to compensate for sparser gradient signal)
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOSS_FOCUS_MODE=advantage_top_k \
LOSS_FOCUS_TOP_K=32 \
LOSS_FOCUS_WEIGHT=4.0 \
PPO_LEARNING_RATE=3e-5 \
KL_COEF=0.1 \
PLAN=balanced \
TRAINABLE_SCOPE=crossview_small \
COLLECT_EPISODES=64 \
EVAL_EPISODES=32 \
FINAL_PROBE_EPISODES=16 \
MIN_SUCCESSFUL_FRAGMENTS=4 \
UPDATE_FRAGMENT_BATCH_SIZE=4 \
ITERS_PER_BLOCK=1 \
STEP_BUDGET=150 \
RUN_TAG="p_obstacle_maze_t6_adv_top_k_balanced_$(date +%Y%m%d_%H%M%S)" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_learning_signal_local.sh"
