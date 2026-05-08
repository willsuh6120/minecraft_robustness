# Maze T6 Uniform Reward PPO

Workspace root: `/home/willsuh1114/minecraft`

Main repo root: `/home/willsuh1114/minecraft/Minestudio`

## Goal

Run the `hard_v6_maze_t6` PPO experiment with:

- full-trajectory uniform loss
- no `suffix_success`
- no `corridor_commit`
- train-world collect on 8 fixed worlds
- heldout evaluation on the other 8 worlds
- sparse control vs waypoint-shaped main

## Train / Heldout Split

Train 8:

- `left_lane_z1_t6`
- `right_lane_z2_t6`
- `left_chicane_t6`
- `right_s_curve_t6`
- `left_funnel_t6`
- `right_gate_entrance_t6`
- `left_outer_detour_t6`
- `right_narrow_door_t6`

Heldout 8:

- `right_lane_z1_t6`
- `left_lane_z2_t6`
- `right_chicane_t6`
- `left_s_curve_t6`
- `right_funnel_t6`
- `left_gate_entrance_t6`
- `right_outer_detour_t6`
- `left_narrow_door_t6`

Round-robin schedule, 2 cycles:

1. `left_lane_z1_t6`
2. `right_lane_z2_t6`
3. `left_outer_detour_t6`
4. `left_chicane_t6`
5. `left_funnel_t6`
6. `right_s_curve_t6`
7. `right_narrow_door_t6`
8. `right_gate_entrance_t6`
9. `left_lane_z1_t6`
10. `right_lane_z2_t6`
11. `left_outer_detour_t6`
12. `left_chicane_t6`
13. `left_funnel_t6`
14. `right_s_curve_t6`
15. `right_narrow_door_t6`
16. `right_gate_entrance_t6`

## Scripts

Common suite:

- `Minestudio/scripts/run_p_obstacle_maze_t6_uniform_reward_suite.sh`

Local wrapper:

- `Minestudio/scripts/run_p_obstacle_maze_t6_uniform_reward_suite_local.sh`
  - defaults to `main_waypoint_uniform` only
  - reuses the latest existing `p_obstacle_maze_t6_uniform_reward_local_*` asset bank
  - fails instead of regenerating a bank if none exists

VM wrapper:

- `Minestudio/scripts/run_p_obstacle_maze_t6_uniform_reward_suite_vm.sh`
  - defaults to `control_sparse_uniform` only
  - reuses the latest existing `p_obstacle_maze_t6_uniform_reward_vm_*` asset bank
  - fails instead of regenerating a bank if none exists

Shared helpers:

- `Minestudio/scripts/_minestudio_runtime.sh`
- `Minestudio/scripts/maze_t6_uniform_reward_tools.py`

## What The Suite Does

1. Reuse an existing `hard_v6_maze_t6` auto-goal eval bank via `ASSET_DIR`.
2. Create singleton fixed-bank views for each maze world plus split manifests.
3. Run a `full16 x 4` baseline eval before training.
4. Run the enabled experiment(s):
   - `control_sparse_uniform`
   - `main_waypoint_uniform`
5. Run intermediate evals:
   - after iter 4: `full16 x 2`
   - after iter 8: `full16 x 4`
   - after iter 12: `full16 x 2`
   - after iter 16: `full16 x 4`
6. Run final probe:
   - `full16 x 16`
7. Write split reports for `train8`, `heldout8`, and `full16`.

## Reward Handling

The env now supports `env_reward_scale`.

- Control A uses `env_reward_scale=0.0` during collect/update.
- Main B uses `env_reward_scale=1.0` during collect/update.
- All eval probes use `env_reward_scale=0.0`.
- Default maze shaping for this suite is `PATH_PROGRESS_REWARD_PER_ZONE=0.125`.

This keeps evaluation sparse even though the task YAMLs contain `path_progress_reward`.
Baseline video probe is now disabled by default (`RUN_VIDEO_PROBE=0`).
The local wrapper runs main only by default; the VM wrapper runs control only by default.

## Default PPO Settings

- `LOSS_FOCUS_MODE=uniform`
- `MIN_SUCCESSFUL_FRAGMENTS=0`
- `COLLECT_EPISODES=64`
- `STEP_BUDGET=150`
- `PPO_EPOCHS=1`
- `PPO_LEARNING_RATE=2e-5`
- `VF_COEF=0.25`
- `KL_COEF=0.1`
- `TRAINABLE_SCOPE=heads`
- `UPDATE_FRAGMENT_BATCH_SIZE=4`
- `NORMALIZE_ADVANTAGE=1`

## Commands

Local:

```bash
cd /home/willsuh1114/minecraft/Minestudio
bash scripts/run_p_obstacle_maze_t6_uniform_reward_suite_local.sh
```

VM:

```bash
cd /home/willsuh1114/minecraft/Minestudio
conda activate minestudio
bash scripts/run_p_obstacle_maze_t6_uniform_reward_suite_vm.sh
```

Use an existing asset bank:

```bash
cd /home/willsuh1114/minecraft/Minestudio
ASSET_DIR=/path/to/generated/assets \
bash scripts/run_p_obstacle_maze_t6_uniform_reward_suite_local.sh
```

Run only the main shaped experiment:

```bash
cd /home/willsuh1114/minecraft/Minestudio
RUN_CONTROL_A=0 RUN_MAIN_B=1 \
bash scripts/run_p_obstacle_maze_t6_uniform_reward_suite_local.sh
```

## Important Outputs

Suite root:

- `outputs/evaluate_rocket/<run_tag>/`

Shared baseline outputs:

- `baseline/video_full16_x1/`
- `baseline/full16_x4/`

Experiment outputs:

- `control_sparse_uniform/`
- `main_waypoint_uniform/`

Per-probe split reports:

- `split_report.json`
- `split_report.md`

Top-level summary:

- `SUITE_SUMMARY.md`

## Procedural Mine Goal Notes

### What Was Wrong

For the player-relative tunnel build path currently used by maze_t6, the working target center is
`target_world_center.y = spawn_y + 0.5`.

### Low-Level Fix

The fix has three pieces:

1. In [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:2046),
   procedural mine `target_world_center` is now emitted as:

   - `x = spawn_x + target_local_x`
   - `y = spawn_y + 0.5`
   - `z = spawn_z + target_local_z`

2. In [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py:2186),
   if the task has exactly one configured target, the goal baker now trusts the current configured target center over any
   stale `goal_pose_hints[].target_world_center`, and recomputes `yaw/pitch` from that repaired center.

3. Existing banks need rebake. The helper
   [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh:61)
   patches:

   - `target_world_center`
   - `target_blocks`
   - `goal_pose_hints`
   - `procedural_layout.realized_metrics.target_world_center`
   - `realized_factor_metrics.target_world_center`
   - `worldgen_manifest.json`

   before calling the goal baker.

### Important Semantics

- Task YAML `spawn_positions[*].position[1]` is player feet Y, not eye Y.
- Goal spec `sampled_camera_position[1]` is also feet Y.
- The actual first-person optical center is above that. For a normal standing player, think in terms of roughly `feet_y + 1.62`.

This matters because the goal baker and rollout metadata use feet Y, not eye Y. With player feet at `160.0`, the optical
center is still above that. In the current player-relative maze_t6 bank the configured target center is `160.5`, and a
same-world goal shot from `center_z3` can still look somewhat downward because the camera stands two cells back and the
eye point is above the feet point.

### Why The First Rollout Frame Looks Like “Floating”

For the `left_lane_z1_t6` probe we inspected, the player feet remain fixed at `y=160.0` in the stored trajectory from
the first frame onward:

- [trajectory.jsonl](/home/willsuh1114/minecraft/Minestudio/outputs/evaluate_rocket/p_obstacle_maze_t6_uniform_reward_vm_20260505_132138/baseline/video_full16_x1/instance_000_left_lane_z1_t6/20260505_133317/mine_coal/seed_1_ep_000/trajectory.jsonl:1)

The frame still reads visually like a raised ledge because the world places a 2-high obstacle immediately in front of the spawn:

- [mine_coal.yaml](/home/willsuh1114/minecraft/Minestudio/outputs/evaluate_rocket/p_obstacle_maze_t6_uniform_reward_vm_20260505_132138/assets/20260505_132138/eval_bank/instance_000/generated_task_groups/20260505_132138/mine_coal.yaml:34)

That obstacle reaches top `y=161`, which is just below the player eye height, so its top face fills the lower foreground and
looks like a platform under the player even though it is actually one block ahead.

Hard reset also installs a spawn support pad for command-authored scenes
([hard_reset.py](/home/willsuh1114/minecraft/Minestudio/minestudio/simulator/callbacks/hard_reset.py:150)),
but in these tunnel worlds the floor is already present at the spawn cell, so the visible foreground is not coming from the support pad.

### If You Want A Less Top-Down Goal Image

Changing only the target-center Y fix does not remove the downward view. That view is a geometric consequence of:

- feet-based teleport metadata
- eye height above feet
- ore being on the floor
- `center_z3` placing the goal camera two cells back

### Why “Just Lower Spawn Y By 1” Is Not A Direct Fix

For these procedural tunnel worlds, the same `spawn_y` currently drives both:

- the absolute teleport used before relative `~ ~ ~` build commands
- the final rollout spawn pose
- the base feet Y used to synthesize `goal_pose_hints`

So there are only two naive outcomes:

1. Lower `spawn_y` everywhere:
   - the whole scene is built 1 block lower too
   - player / goal camera / obstacle remain in the same relative geometry
   - the visual problem does **not** actually go away

2. Keep the build anchor high but lower only the final player pose:
   - the player lands inside the existing floor unless extra support geometry is added
   - hard reset's current support pad clears only the center column above the pad
     ([hard_reset.py](/home/willsuh1114/minecraft/Minestudio/minestudio/simulator/callbacks/hard_reset.py:102))
   - that creates a 1x1 lowered pocket, not a clean lowered corridor

So if a future agent wants the player/camera effectively 1 block lower **relative to the obstacle world**, it must split:

- scene build anchor Y
- final rollout spawn feet Y
- goal camera feet Y

and add matching support / corridor geometry rather than editing a single scalar.

### Current Structural State

The temporary fixed-anchor migration for procedural mine scenes was rolled back.

Current behavior is again:

- first teleport the player to the intended spawn
- build the tunnel with player-relative commands:
  - `/execute as @p at @s run ...`
- teleport once more after build

Reason for rollback:

- the fixed-anchor `/execute positioned ...` variant introduced regressions in
  maze_t6 debug validation:
  - some worlds rendered visibly wrong from the user's manual check
  - some rebakes started failing with `camera_collision_warning` across all pose candidates
  - the change did not reliably solve the original perceived support-block issue

Relevant files:

- [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:880)
- [hard_reset.py](/home/willsuh1114/minecraft/Minestudio/minestudio/simulator/callbacks/hard_reset.py:38)
- [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh:41)

For existing banks, rebake helper now normalizes task YAMLs back to the
player-relative command pattern if they were temporarily patched to
`/execute positioned ...`.

Target center note:

- the current working player-relative maze_t6 YAMLs use `target_world_center.y = 160.5`
- rebake helper also patches `goal_pose_hints[].target_world_center` to the same value

If a future agent should see a more front-on goal, the geometry has to change, for example:

- use a different goal pose family than `center_z3`
- raise the ore relative to the floor
- lower the camera by changing the world geometry, not just metadata
- remove the immediate foreground obstacle if the spawn frame should look flat

Support-pad handling:

- procedural mine tasks now request `spawn_support_pad.cleanup_after_build: true`
- `HardResetCallback` removes only still-existing support blocks by running:
  - `fill ... minecraft:air replace minecraft:barrier`
- this is safer than globally disabling the support pad, because any floor/stone
  that already replaced the barrier is left untouched

Goal-bake support note:

- rollout spawn support and goal-probe support are separate paths
- rollout uses `HardResetCallback.spawn_support_pad`
- goal baking / auto-pose probing used `goal_pose_hints[].support_block` plus
  `crossview_utils._ensure_support_block()`
- for procedural mine hints, `goal_pose_hints[].support_block.enabled` is now
  emitted as `false`, and `crossview_utils` respects that flag instead of
  silently regenerating a barrier support block

### Runtime Notes After The Anchored-Build Fix

Two separate runtime traps showed up while validating rebakes on VM:

- `scripts/_minestudio_runtime.sh` must prefer
  `/home/willsuh1114/miniconda3/envs/minestudio/bin/python`
  over a base `CONDA_PREFIX`, because some shells inherit
  `CONDA_PREFIX=/home/willsuh1114/miniconda3` and that Python is missing
  `huggingface_hub`.
- VM rebake/debug scripts should use the workspace cache
  `/home/willsuh1114/minecraft/.minestudio`, not
  `/home/willsuh1114/envgen2/.minestudio`, otherwise task-config refresh can
  hit the wrong cache and outside-workspace write paths.

Current launcher behavior:

- `minestudio/simulator/minerl/env/launchClient.sh` now uses an existing
  `DISPLAY` directly on Linux CPU-render runs, and falls back to `xvfb-run`
  only if `DISPLAY` is unset.
- `_minestudio_runtime.sh` now gives `xvfb-run` GLX-capable defaults:
  `-screen 0 1920x1200x24 -dpi 72 +extension RANDR +extension GLX +iglx +extension MIT-SHM +render -nolisten tcp -noreset`

Important distinction:

- If rebake fails with `pose_success_count=0` / `sweep_success_count=0`, first
  check stale goal geometry metadata (`target_world_center`, `goal_pose_hints`)
  and whether the bank was patched after the fixed-anchor migration.
- If rebake gets past that and then dies in `Failed to initialize GLFW`, the
  remaining issue is display/OpenGL runtime setup, not maze geometry.

## 2026-05-06 Final Working Fix For `maze_t6` Goal/Spawn Debugging

This section supersedes the earlier trial-and-error notes above for the
specific `maze_t6` procedural mine debugging flow.

### Final Symptom Split

There were two different bugs mixed together:

1. **Support-block artifact bug**
   - rollout first frame and goal bake both looked like the camera/player was
     standing on an extra block.
   - this was not the same code path in rollout vs goal bake.

2. **Goal annotation bug**
   - after support artifacts were removed, the baked goal composition looked
     good, but the red target mask/bbox still tracked the block one cell above
     the actual coal.

The final fix keeps the good composition while moving only the annotation to the
real coal.

### High-Level Fix

The final working design is:

- keep the **camera composition** from the good `center_z3` view
- keep the **world geometry** unchanged
- keep the **rollout spawn** behavior unchanged except for cleaning up leftover
  temporary support barrier
- make the **goal annotation target** point to the real coal block
- make the **camera look target** a separate concept from the annotation target

In other words:

- camera pose uses a **look target**
- bbox/mask projection uses an **annotation target**

Those two are no longer forced to be the same point.

### Low-Level Code Changes

#### 1. Rollout support cleanup

File:
- [hard_reset.py](/home/willsuh1114/minecraft/Minestudio/minestudio/simulator/callbacks/hard_reset.py)

What changed:
- `spawn_support_pad.cleanup_after_build` is now supported.
- After command-authored world build, hard reset removes only leftover temporary
  support blocks via:
  - `fill ... minecraft:air replace minecraft:barrier`

Important lines:
- support cleanup config: around lines `77-82`
- cleanup function: around lines `128-142`
- cleanup after build / settle: around lines `189-199`

Why this matters:
- rollout still gets safe temporary support during reset/build
- but the barrier does not remain visible in the final first-person spawn frame

#### 2. Goal-probe support block disabled for procedural mine hints

Files:
- [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py)
- [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py)

What changed:
- procedural mine `goal_pose_hints` now emit `support_block.enabled: false`
- `_ensure_support_block()` returns early when that flag is false

Important lines:
- hint support disable logic: [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:462)
- support block ignore path: [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py:341)

Why this matters:
- baked goal images no longer get the extra bottom support barrier

#### 3. `center_z3` pose pitch tuned upward

File:
- [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py)

What changed:
- `p_obstacle_center_z3` hint now includes `pitch_offset_deg = -6.0`
- hint generation applies that offset after computing geometric look pitch

Important lines:
- `pitch_offset_deg` on the `p_obstacle_center_z3` profile:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:546)
- pitch application:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:823)

Why this matters:
- the goal image keeps the same world but tilts the head slightly upward
- the coal is no longer pushed as low in the view

#### 4. Separate `annotation target` from `look target`

File:
- [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py)

Final procedural mine behavior:
- `target_world_center`:
  - points to the **real coal block center**
  - currently `spawn_y - 0.5`
- `goal_camera_look_target_world_center`:
  - points to the **upper visual anchor** used to preserve the good camera pose
  - currently `spawn_y + 0.5`

Important lines:
- look-target load in hint builder:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:452)
- hints store both:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:849)
- procedural mine target center set to actual coal:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:2098)
- procedural mine look target stored separately:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:2103)
- procedural layout persists look target:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:2237)

Why this matters:
- before this change, fixing the annotation target also moved the camera pitch
- now the camera can keep the good composition while the red box/mask tracks
  the real coal block

#### 5. Hint consumption now preserves serialized hint target/pose

File:
- [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py)

What changed:
- hint parsing no longer blindly overwrites `hint["target_world_center"]` with
  `default_target["world_center"]`
- hint parsing also understands `look_target_world_center`
- yaw/pitch recomputation only happens when hint yaw/pitch are absent

Important lines:
- target vs look-target parsing:
  [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py:2188)
- recompute yaw/pitch only if missing:
  [crossview_utils.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/crossview_utils.py:2214)

Why this matters:
- the serialized hint now fully controls the intended composition
- the annotation target can be lower than the visual look target

### Debug-Rebake Workflow

Files:
- [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh)
- [run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh)

What changed:
- rebake helper now defaults to **debug-copy mode**
- if `WORK_TASK_GROUP_PATH` is not provided, it auto-copies the source task
  group into:
  - `outputs/evaluate_rocket/_debug_task_groups/<timestamp>/...`
- 3-world VM debug script already rebakes and rolls out from copied task groups
  under:
  - `OUT_ROOT/task_groups/...`

Important lines:
- auto debug-copy root:
  [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh:11)
- auto copy decision:
  [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh:35)
- rebake patch now writes:
  - lower `target_world_center`
  - higher `goal_camera_look_target_world_center`
  [rebake_goal_from_task_group_local.sh](/home/willsuh1114/minecraft/Minestudio/scripts/rebake_goal_from_task_group_local.sh:84)
- 3-world debug script work root:
  [run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh:38)
- per-world copied task-group path:
  [run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_maze_t6_goal_rebake_and_rollout_3worlds_vm.sh:86)

Why this matters:
- debugging no longer mutates the source eval bank
- all rebakes/rollouts happen on disposable working copies

### Official `mine_coal` Tool Loadout

The correct starting tool for `mine_coal` in this repo's official interaction
benchmark config is:

- `wooden_pickaxe`

References:
- official task YAML:
  [mine_coal.yaml](/home/willsuh1114/minecraft/Minestudio/minestudio/benchmark/task_configs/rocket_interaction/mine_coal.yaml:7)
- procedural fallback inventory:
  [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py:993)

If a rollout visually shows `diamond_axe`, do not assume the procedural maze
worldgen is wrong first. Check whether the viewed output came from another task,
another bank, or another stale debug run.

### Final Mental Model

When debugging future goal-image issues for these P-obstacle maze worlds, keep
the concepts separate:

- **rollout support cleanup**
  - temporary reset/build support under the player
- **goal-probe support**
  - temporary support used only while baking auto-goal views
- **look target**
  - the point the camera aims at
- **annotation target**
  - the voxel center projected into bbox/mask

The final working fix was not “move everything down by one block”.
It was:

- remove temporary support artifacts
- preserve the good camera pose
- split visual aim from annotation target
- project the bbox/mask from the actual coal voxel center

### Fresh Rerun Scripts For The Fixed Maze T6 Setup

Two dedicated rerun wrappers now exist for re-running the original split
experiment from scratch with the fixed goal/spawn logic:

- local main:
  [run_p_obstacle_maze_t6_main_local_fresh.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_p_obstacle_maze_t6_main_local_fresh.sh)
- VM control:
  [run_p_obstacle_maze_t6_control_vm_fresh.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_p_obstacle_maze_t6_control_vm_fresh.sh)

Current behavior:

- both wrappers now **reuse an existing maze_t6 asset bank** if one exists
- they auto-detect the latest local / VM `assets/<timestamp>` dir unless
  `ASSET_DIR` is passed explicitly
- before running the suite, they patch the bank in-place with:
  - `time_limit = decision_steps + warmup_steps = 180`
  - `init_inventory = [{"slot": 0, "type": "diamond_pickaxe", "quantity": 1}]`
- local wrapper runs:
  - `full16 x1` video rollout probe
  - baseline eval
  - `main_waypoint_uniform` only
- VM wrapper runs:
  - baseline eval
  - `control_sparse_uniform` only

Why `time_limit=180`:

- the rollout protocol still applies `warmup_noop_steps=30`
- those warmup env steps are **already excluded** from collected PPO
  trajectories and update fragments, because fragment capture starts only after
  `session.reset(...)` returns and the episode loop begins
- but the env `JudgeReset` callback counts warmup steps against `time_limit`
- so to get **150 controllable agent steps**, the generated / reused bank must
  expose `time_limit=180`

Relevant files:

- [patch_maze_t6_asset_dir_for_rerun.py](/home/willsuh1114/minecraft/Minestudio/scripts/patch_maze_t6_asset_dir_for_rerun.py)
- [interaction_worldgen.py](/home/willsuh1114/minecraft/Minestudio/minestudio/tutorials/inference/evaluate_rocket/interaction_worldgen.py)
- [mine_coal.yaml](/home/willsuh1114/minecraft/Minestudio/minestudio/benchmark/task_configs/rocket_interaction/mine_coal.yaml)

Both keep the current defaults:

- `PATH_PROGRESS_REWARD_PER_ZONE=0.125`
- `BANK_WORKERS=4`
- `COLLECT_WORKERS=4`
- eval/final workers `=4`

### Local Engine Download Race Fix

On local reruns, the first fresh maze suite could fail before training with
repeated messages like:

- `Detecting missing simulator engine...`
- `FileExistsError: .../.minestudio/engine`

Root cause:

- multiple probe/bank worker processes were entering `check_engine()` at the
  same time
- each tried to download/extract the MineStudio engine into the same
  `.minestudio/engine` directory

Final fix:

1. `minestudio/simulator/entry.py`
   - `check_engine()` now uses an inter-process install lock file:
     `.engine_install.lock`
   - only one process downloads/extracts the engine
   - other processes wait until the engine jar exists
   - `download_engine()` also deletes an incomplete pre-existing
     `.minestudio/engine` directory before unzip if the final jar is missing,
     so aborted local installs can recover on the next run

2. `_minestudio_runtime.sh`
   - added `ensure_minestudio_engine()`
   - runs `check_engine(skip_confirmation=True)` once up front

3. fresh local/vm suite wrappers
   - now call `setup_minestudio_runtime`
   - then `ensure_minestudio_engine`
   - so engine is present before the parallel workers start

Relevant files:

- [entry.py](/home/willsuh1114/minecraft/Minestudio/minestudio/simulator/entry.py)
- [_minestudio_runtime.sh](/home/willsuh1114/minecraft/Minestudio/scripts/_minestudio_runtime.sh)
- [run_p_obstacle_maze_t6_main_local_fresh.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_p_obstacle_maze_t6_main_local_fresh.sh)
- [run_p_obstacle_maze_t6_control_vm_fresh.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_p_obstacle_maze_t6_control_vm_fresh.sh)

### VM Fresh Rebuild Control Wrapper

When old outputs or asset banks were deleted and the VM control experiment had
to be restarted from scratch, a separate rebuild wrapper was added:

- [run_p_obstacle_maze_t6_control_vm_rebuild.sh](/home/willsuh1114/minecraft/Minestudio/scripts/run_p_obstacle_maze_t6_control_vm_rebuild.sh)

Current behavior:

- does **not** reuse an existing `ASSET_DIR`
- generates a brand-new 16-world `eval_bank` under a fresh `OUT_ROOT/assets`
- bakes goals for the new bank
- runs `full16 x1` video probe before training
- runs baseline eval
- runs `control_sparse_uniform` only

Default run ordering:

1. asset generation / goal bake
2. `baseline/video_full16_x1`
3. `baseline/full16_x4`
4. `control_sparse_uniform` train/eval schedule

This wrapper is the correct restart path when a previous control run lost both
its outputs and its asset bank.
## 2026-05-07 Maze T6 follow-up

- Added `scripts/resolve_maze_t6_best_checkpoint.py` to recover the best iter checkpoint from `iter_004/008/012/016` eval split reports and map it back to the corresponding `training/iter_xxx/*/suite_iteration.json -> next_model_path`.
- Added generic probe wrapper `scripts/run_p_obstacle_maze_t6_best_checkpoint_probe.sh`.
- Added local main wrapper `scripts/run_p_obstacle_maze_t6_main_best_checkpoint_probe_local.sh`.
- Added VM control wrapper `scripts/run_p_obstacle_maze_t6_control_best_checkpoint_probe_vm.sh`.
- Added local final-model video wrapper `scripts/run_p_obstacle_maze_t6_main_final_video_full16_x4_local.sh` for `full16 x4` video rollout with the latest trained main model.
- Added `scripts/analyze_maze_t6_direction_bias.py` to compare each trained direction against its held-out mirror direction on final probe.
- Current interpretation of the split:
  - The train/held-out split is not a clean left-vs-right generalization test.
  - Each geometry family is split so one direction is trained and the mirror is held out.
  - Final main results do not show a simple global left-bias. Mirror variants are actually better in 5/8 geometry families, trained direction better in 1/8, equal in 2/8.
  - Better next experiment: include both left/right members of each geometry family in collect/train, and hold out entire geometry families instead of mirror direction.

## 2026-05-07 Maze T6 redesign runners

- Added `docs/maze_t6_redesign_plan.md` documenting the new recommended experiment structure:
  - Stage 1: all-16 seen-bank mastery
  - Stage 2: family-heldout CV
- Extended `scripts/maze_t6_uniform_reward_tools.py`:
  - `families/<family>/bank_manifest.json`
  - `cv_folds/<fold>/{train,val,test}/bank_manifest.json`
  - new fields: `family_names`, `cv_fold_names`
- Added Stage 1 generic runner:
  - `scripts/run_p_obstacle_maze_t6_mastery_suite.sh`
- Added Stage 1 machine wrappers:
  - `scripts/run_p_obstacle_maze_t6_mastery_main_local.sh`
  - `scripts/run_p_obstacle_maze_t6_mastery_control_vm.sh`
- Added Stage 2 generic runner:
  - `scripts/run_p_obstacle_maze_t6_family_cv_suite.sh`
- Added Stage 2 machine wrappers:
  - `scripts/run_p_obstacle_maze_t6_family_cv_main_local.sh`
  - `scripts/run_p_obstacle_maze_t6_family_cv_control_vm.sh`
- Stage 1 design:
  - collect from `bank_views/splits/full16`
  - `collect_world_instances=16`
  - `collect_episodes_per_task=64` => exactly 4 eps per world per iter
- Stage 2 design:
  - collect from `bank_views/cv_folds/<fold>/train`
  - `collect_world_instances=12`
  - `collect_episodes_per_task=72` => exactly 6 eps per train world per iter
  - checkpoint eval on `val`
  - best checkpoint selected by `val`
  - final probe on `test`

## 2026-05-07 PPO truncate/bootstrap fix

- Found a structural PPO bug in maze_t6 training:
  - `JudgeResetCallback` converted time-limit endings into `terminated=True` instead of `truncated=True`.
  - rollout fragments also treated `terminated || truncated` as the same `env_done`, and GAE zero-bootstrapped every fragment tail.
- Fixed in code:
  - `minestudio/simulator/callbacks/judgereset.py`
    - time-limit now sets `truncated=True` and preserves natural `terminated=True`
  - `interaction_crossview_rollout.py` and `interaction_posttrain_runner.py`
    - fragments now store per-step `terminated` and `truncated`
    - episode payload now stores `bootstrap_value`, `bootstrap_valid`, `bootstrap_reason`
    - for `truncated` or external `max_steps`, bootstrap value is estimated from the final observation
  - `interaction_ppo_update.py`
    - GAE now distinguishes `terminated` from `truncated`
    - final-step truncation / `max_steps` uses stored bootstrap value instead of unconditional zero-bootstrap
- Why this matters:
  - previous runs were underestimating continuation value at timeout/max-step boundaries
  - this could destabilize critic training and contaminate actor advantages
- Recommended reruns after this fix:
  - local: `run_p_obstacle_maze_t6_mastery_round_robin_main_local.sh`
  - VM: `run_p_obstacle_maze_t6_mastery_main_vm.sh`
  - compare against pre-fix mastery before moving to `heads_last` / lower-KL ablations
