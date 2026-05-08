# Maze T6 Redesign Plan

## Problem

The current split is:

- train: one direction from each geometry family
- held-out: the mirror direction from the same family

This confounds two different questions:

1. Does the policy transfer across left/right mirror direction?
2. Is the family itself easier or harder at baseline?

In the current results, these two effects are mixed. The split is not a clean generalization test.

## World Universe

The current `maze_t6` bank has 16 worlds, organized into 8 geometry families:

- `lane_z1`: `left_lane_z1_t6`, `right_lane_z1_t6`
- `lane_z2`: `left_lane_z2_t6`, `right_lane_z2_t6`
- `chicane`: `left_chicane_t6`, `right_chicane_t6`
- `s_curve`: `left_s_curve_t6`, `right_s_curve_t6`
- `funnel`: `left_funnel_t6`, `right_funnel_t6`
- `gate_entrance`: `left_gate_entrance_t6`, `right_gate_entrance_t6`
- `outer_detour`: `left_outer_detour_t6`, `right_outer_detour_t6`
- `narrow_door`: `left_narrow_door_t6`, `right_narrow_door_t6`

## Stage 1: Seen-Bank Mastery

Goal:

- Answer the short-horizon practical question:
  - can PPO + waypoint shaping improve performance on the 16 known worlds if the split artifact is removed?

Design:

- Train bank: all 16 worlds
- Validation / checkpoint eval: same 16 worlds
- Final probe: same 16 worlds with fresh episode seeds
- Claim:
  - seen-bank mastery only
  - no generalization claim

Collect schedule:

- Use `bank_views/splits/full16`
- Set `collect_world_mode=fixed_bank`
- Set `collect_world_instances=16`
- Set `collect_episodes_per_task=64`

Why `64`:

- In fixed-bank mode, `interaction_crossview_ppo_pilot.py` distributes episodes across active instances.
- With 16 instances and 64 total episodes, this gives exactly 4 episodes per world per iter.

Recommended hyper-run:

- 24 train iters
- checkpoint eval at iter `4, 8, 12, 16, 20, 24`
- final probe `full16 x16` or `full16 x32`
- run both `main` and `control`
- independent training seeds: at least `3`, preferably `5`

Interpretation:

- If `main` still fails to beat `control` here, the issue is not mostly the left/right split.
- If `main` improves here, the old split was a major artifact.

## Stage 2: Family-Heldout Generalization

Goal:

- Measure transfer to unseen geometry families, not unseen mirror direction.

Rule:

- Keep both left/right members of each family together.
- Hold out whole families.

### Proposed 4-fold CV

Each fold uses:

- train: 6 families = 12 worlds
- val: 1 family = 2 worlds
- test: 1 family = 2 worlds

This is the proposed assignment:

#### Fold A

- train: `lane_z1`, `chicane`, `s_curve`, `funnel`, `outer_detour`, `narrow_door`
- val: `lane_z2`
- test: `gate_entrance`

#### Fold B

- train: `lane_z1`, `lane_z2`, `s_curve`, `funnel`, `gate_entrance`, `outer_detour`
- val: `chicane`
- test: `narrow_door`

#### Fold C

- train: `lane_z1`, `lane_z2`, `chicane`, `gate_entrance`, `outer_detour`, `narrow_door`
- val: `funnel`
- test: `s_curve`

#### Fold D

- train: `lane_z2`, `chicane`, `s_curve`, `funnel`, `gate_entrance`, `narrow_door`
- val: `lane_z1`
- test: `outer_detour`

Rationale:

- Easy and hard families are spread across folds.
- Each family appears exactly once as val or test.
- Train still contains 12 worlds, so collect remains diverse.

### Collect schedule for family CV

Use:

- `bank_views/cv_folds/<fold>/train`

Set:

- `collect_world_mode=fixed_bank`
- `collect_world_instances=12`
- `collect_episodes_per_task=72`

Why `72`:

- 12 train worlds x 6 episodes each = exact balance
- avoids uneven distribution like `64 / 12`

Eval:

- checkpoint validation on `cv_folds/<fold>/val`
- choose best checkpoint using validation only
- final report on `cv_folds/<fold>/test`

Recommended counts:

- val: `x16` per world = 32 episodes
- test: `x32` per world = 64 episodes

Recommended training:

- 24 or 32 train iters
- `3` seeds minimum per arm

Statistical unit:

- one run = one `(fold, seed, arm)` tuple
- compare `main - control` as paired differences across fold/seed runs

## Stage 3: External Test

After Stage 2:

- generate new families or perturbed family variants
- keep them untouched during train and val
- use only for the final external test

Examples:

- wider or narrower `narrow_door`
- shifted `chicane`
- alternate `funnel` approach width
- additional lane-commitment layouts

## Runner Changes Needed

### For Stage 1

Current suite:

- collects from a singleton bank per iter

Need:

- collect from `full16` fixed bank every iter
- no more left-only / right-only round-robin schedule

Concretely:

- `collect_fixed_bank_dir="${BANK_VIEW_ROOT}/splits/full16"`
- `collect_world_instances=16`
- `collect_episodes_per_task=64`

### For Stage 2

Need:

- `collect_fixed_bank_dir="${BANK_VIEW_ROOT}/cv_folds/<fold>/train"`
- `eval_fixed_bank_dir="${BANK_VIEW_ROOT}/cv_folds/<fold>/val"`
- `final_eval_fixed_bank_dir="${BANK_VIEW_ROOT}/cv_folds/<fold>/test"`
- `collect_world_instances=12`
- `collect_episodes_per_task=72`

Also:

- select best checkpoint by validation success only
- never select by test

## What To Run First

Recommended next sequence:

1. Stage 1 mastery, `main` vs `control`, 3 seeds each
2. If Stage 1 shows `main > control`, proceed to Stage 2 family-heldout CV
3. If Stage 1 does not show improvement, fix stability first:
   - replay / mixed collect
   - stronger KL anchor
   - more train iters
   - broader trainable scope

## Repo Support Added

`scripts/maze_t6_uniform_reward_tools.py` now exposes:

- family names
- family bank views
- 4-fold family CV bank views

Generated bank views now include:

- `families/<family>/bank_manifest.json`
- `cv_folds/<fold>/{train,val,test}/bank_manifest.json`
