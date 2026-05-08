# EnvGen / Minestudio Research Summary

## Scope

Current working area:

- `Minestudio/`
- `ROCKET-2/`
- `EnvGen/`
- `minecraft_envgen/`

This folder is a slim copy of `/home/gyulab/envgen2` with code, scripts, configs, and small docs only.

Excluded on purpose:

- `outputs/`
- `logs/`
- baked goals
- codepacks
- large media/image exports
- heavy helper trees not needed for current work

## Research Goal

Study whether pretrained goal-image-conditioned ROCKET-2 can be post-trained with PPO to become robust to controlled world perturbations in Minecraft.

Main factors used so far:

- `O`: occlusion
- `P`: path complexity / maze obstacles
- `H`: initial heading offset

## Current Research State

### 1. O2 occlusion result

This is the clean positive result.

- Base ROCKET-2: `24/256 = 9.4%`
- Best O2-trained model: `95/256 = 37.1%`
- Delta: `+27.7pp`

Interpretation:

- PPO can improve robustness when the curriculum is teachable.
- Sequential block curriculum worked.
- Mixed quota collection did not reproduce the gain reliably.

### 2. P maze / path-complexity result

This is the unstable part.

- Main eval bank: `maze_t6`
- Baseline: `72/256 = 28.1%`
- Warm/mixed curriculum: weak and unstable
- Focused single-source teachers can help somewhat, but gains are narrow and can forget

Interpretation:

- The main bottleneck is likely credit assignment, not just lack of PPO updates.
- O2 mostly depends on late-stage target interaction.
- Maze depends on earlier path commitment, detour choice, and re-entry.
- Sparse terminal PPO is likely weighting the wrong part of the trajectory.

### 3. H heading-offset direction

This is the new cleaner benchmark.

Question:

- Can PPO teach ROCKET-2 to recover when the target is not initially visible and the agent must turn/search/re-acquire?

The H suite has been implemented, but only smoke-tested so far.
There is no full same-bank trained-vs-baseline result yet.

## Code Changes Completed

### A. Exact heading override support

Modified:

- `Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py`

Added support for:

- `mine_heading_offset_deg`
- `mine_heading_offset_abs_deg`
- `mine_spawn_yaw_deg`
- `mine_spawn_pitch_deg`

Behavior:

- exact yaw override has highest priority
- realized world metadata records actual yaw and heading offset fields

### B. H-only asset generator

Added:

- `Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_prepare_mine_h_heading_assets.py`

This generates:

- `collect_plan.json`
- `collect_block_plans/`
- `eval_bank/`
- `final_eval_bank/`
- `asset_manifest.json`

Current default bank was simplified to straight-only 8 variants:

- `straight_H000`
- `straight_H090_pos`
- `straight_H090_neg`
- `straight_H120_pos`
- `straight_H120_neg`
- `straight_H150_pos`
- `straight_H150_neg`
- `straight_H180`

### C. H calibration / learning scripts

Added:

- `Minestudio/scripts/run_h_heading_offset_calibration_local.sh`
- `Minestudio/scripts/run_h_heading_offset_learning_signal_local.sh`
- `Minestudio/scripts/run_ho_heading_occlusion_suite_local.sh`

### D. PPO focus mode

Modified:

- `Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_crossview_ppo_pilot.py`
- `Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_ppo_update.py`

Added:

- `prefix_success`

Reason:

- heading recovery and some maze decisions are early-causal
- `suffix_success` can over-focus final mining and miss the real recovery behavior

### E. H report script

Added:

- `Minestudio/scripts/render_h_heading_offset_report.py`

### F. VM sync helper

Added:

- `Minestudio/scripts/fetch_envgen_vm1_code.sh`
- `Minestudio/scripts/fetch_gcp_code_delta.sh`

These pull code changes from VM to local, complementing the existing local-to-VM sync flow.

## Smoke Validation Done

H-only suite was smoke-tested.

- tiny H asset generation: passed
- tiny baseline rollout: passed
- tiny PPO smoke: passed through update path

Important caveat:

- this only validates wiring
- it does not yet show real H generalization improvement

## Maze Reward-Shaping Discussion

Not implemented yet. This is the current design direction.

Goal:

- replace pure sparse terminal PPO for maze collection with geometry-aware intermediate rewards

Current proposal:

- reward zone `1`: first valid path entry on `z=1`
- reward zone `2`: first valid path entry on `z=2`
- reward zone `3`: first valid path entry on `z=3`
- reward zone `A`: first valid approach-zone entry on `z=4`
- terminal reward `T`: success

Important correction:

- reward should **not** be given to every open cell on a row
- reward should only be given to open cells that still have a valid forward path into the funnel/approach region
- cells that require backing up to a smaller `z` before they can rejoin the winning corridor should not be rewarded
- example: in `left_lane_z1_t6`, the dead right branch at `x=+2` on `z=1,2,3` must stay unrewarded

This matters for variants like:

- `left_narrow_door_t6`
- `right_chicane_t6`
- `left/right_outer_detour_t6`

## Recommended Next Research Steps

1. Run full H calibration on the fixed eval bank.
2. Run H PPO with same-bank final probe.
3. If H improves with `prefix_success`, compare against sparse and suffix-based controls.
4. If maze work resumes, implement path-zone shaped reward instead of row-open reward.

## Notes

This folder was created to keep the working tree small and reduce noise from large outputs and generated artifacts in `/home/gyulab/envgen2`.
