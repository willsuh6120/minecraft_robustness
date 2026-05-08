import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import yaml

from minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_o2_generalization_assets import (
    build_bake_goal_cmd,
    build_worldgen_cmd,
    detect_generated_dir,
    list_child_dirs,
    resample_plan_rows,
    stable_layout_seed,
    write_attempt_failure,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    canonical_task_key,
    classify_factor_split,
    factor_levels_to_worldgen_suggestions,
    hard_factor_count,
    normalize_factor_levels,
)


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_OUT_DIR = "outputs/evaluate_rocket/mine_p_straight_obstacle_assets"
DEFAULT_FACTOR_LEVELS = {"R": 0, "H": 0, "O": 0, "C": 0, "P": 2}
PATH_TARGET_LOCAL = [0, 5]

PATH_OBSTACLE_VARIANTS_CENTER_Z3: List[Dict[str, object]] = [
    {
        "variant_id": "center_z2_single",
        "positions": [[0, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "center_z1_single",
        "positions": [[0, 1]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "center_z1_z2",
        "positions": [[0, 1], [0, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "bar_z2_full",
        "positions": [[-1, 2], [0, 2], [1, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "bar_z1_full",
        "positions": [[-1, 1], [0, 1], [1, 1]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "double_bar_z1_z2",
        "positions": [[-1, 1], [0, 1], [1, 1], [-1, 2], [0, 2], [1, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "left_gate",
        "positions": [[-1, 1], [-1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "right_gate",
        "positions": [[1, 1], [1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "left_zigzag",
        "positions": [[-1, 1], [0, 2], [1, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "right_zigzag",
        "positions": [[1, 1], [0, 2], [-1, 2]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "left_post_gate",
        "positions": [[0, 2], [-1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "right_post_gate",
        "positions": [[0, 2], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "left_funnel",
        "positions": [[0, 1], [1, 1], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "right_funnel",
        "positions": [[0, 1], [-1, 1], [-1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "checker_left",
        "positions": [[0, 1], [-1, 2], [1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
    },
    {
        "variant_id": "checker_right",
        "positions": [[0, 1], [-1, 2], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
    },
]

PATH_OBSTACLE_VARIANTS_HARD_V2: List[Dict[str, object]] = [
    {
        "variant_id": "left_lane_z1",
        "positions": [[0, 1], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_lane_z1",
        "positions": [[0, 1], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_lane_z2",
        "positions": [[0, 2], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_lane_z2",
        "positions": [[0, 2], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_chicane",
        "positions": [[0, 1], [1, 1], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_chicane",
        "positions": [[0, 1], [-1, 1], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_funnel_strict",
        "positions": [[0, 1], [1, 1], [0, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_funnel_strict",
        "positions": [[0, 1], [-1, 1], [0, 2], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_gate_blocked_entry",
        "positions": [[0, 1], [1, 1], [-1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_gate_blocked_entry",
        "positions": [[0, 1], [-1, 1], [1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_squeeze",
        "positions": [[0, 1], [1, 1], [1, 2], [-1, 2], [1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_squeeze",
        "positions": [[0, 1], [-1, 1], [-1, 2], [1, 2], [-1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_offset_wall",
        "positions": [[0, 2], [1, 2], [1, 1], [-1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_offset_wall",
        "positions": [[0, 2], [-1, 2], [-1, 1], [1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_double_gate",
        "positions": [[0, 1], [1, 1], [0, 2], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_double_gate",
        "positions": [[0, 1], [-1, 1], [0, 2], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
]

PATH_OBSTACLE_VARIANTS_HARD_V3: List[Dict[str, object]] = [
    {
        "variant_id": "left_lane_z1",
        "positions": [[0, 1], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_lane_z1",
        "positions": [[0, 1], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_chicane",
        "positions": [[0, 1], [1, 1], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_chicane",
        "positions": [[0, 1], [-1, 1], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_funnel_strict",
        "positions": [[0, 1], [1, 1], [0, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_funnel_strict",
        "positions": [[0, 1], [-1, 1], [0, 2], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_gate_blocked_entry",
        "positions": [[0, 1], [1, 1], [-1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_gate_blocked_entry",
        "positions": [[0, 1], [-1, 1], [1, 2], [0, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_squeeze",
        "positions": [[0, 1], [1, 1], [1, 2], [-1, 2], [1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_squeeze",
        "positions": [[0, 1], [-1, 1], [-1, 2], [1, 2], [-1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_offset_wall",
        "positions": [[0, 2], [1, 2], [1, 1], [-1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_offset_wall",
        "positions": [[0, 2], [-1, 2], [-1, 1], [1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_double_gate",
        "positions": [[0, 1], [1, 1], [0, 2], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_double_gate",
        "positions": [[0, 1], [-1, 1], [0, 2], [-1, 2], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_cross_gate",
        "positions": [[0, 1], [1, 1], [-1, 2], [0, 2], [2, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_cross_gate",
        "positions": [[0, 1], [-1, 1], [1, 2], [0, 2], [-2, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
]

PATH_OBSTACLE_VARIANTS_HARD_V4: List[Dict[str, object]] = copy.deepcopy(PATH_OBSTACLE_VARIANTS_HARD_V3)

PATH_OBSTACLE_VARIANTS_HARD_V5_OPEN_PATH: List[Dict[str, object]] = [
    {
        "variant_id": "left_lane_center_z1",
        "positions": [[0, 1], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_lane_center_z1",
        "positions": [[0, 1], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_lane_center_z2",
        "positions": [[0, 2], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_lane_center_z2",
        "positions": [[0, 2], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_chicane_open",
        "positions": [[0, 1], [1, 1], [1, 2], [-1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_chicane_open",
        "positions": [[0, 1], [-1, 1], [-1, 2], [1, 0]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_funnel_open",
        "positions": [[0, 1], [0, 2], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_funnel_open",
        "positions": [[0, 1], [0, 2], [-1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_gate_open",
        "positions": [[0, 1], [0, 2], [1, 1], [-2, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_gate_open",
        "positions": [[0, 1], [0, 2], [-1, 1], [2, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_s_curve_open",
        "positions": [[-1, 1], [0, 1], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_s_curve_open",
        "positions": [[-1, 1], [0, 1], [1, 1], [-1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_narrow_door",
        "positions": [[0, 1], [0, 2], [1, 1], [1, 2], [-2, 1]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_narrow_door",
        "positions": [[0, 1], [0, 2], [-1, 1], [-1, 2], [2, 1]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "left_outer_detour",
        "positions": [[-1, 1], [0, 1], [0, 2], [1, 1], [1, 2]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
    {
        "variant_id": "right_outer_detour",
        "positions": [[-1, 1], [-1, 2], [0, 1], [0, 2], [1, 1]],
        "material": "minecraft:cobblestone",
        "height": 2,
    },
]

PATH_OBSTACLE_VARIANTS_HARD_V6_MAZE_T6: List[Dict[str, object]] = [
    {
        "variant_id": "left_lane_z1_t6",
        "positions": [[0, 1], [1, 1], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_lane_z1_t6",
        "positions": [[0, 1], [-1, 1], [-1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_lane_z2_t6",
        "positions": [[0, 2], [1, 1], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_lane_z2_t6",
        "positions": [[0, 2], [-1, 1], [-1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_chicane_t6",
        "positions": [[0, 1], [1, 1], [-1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_chicane_t6",
        "positions": [[0, 1], [-1, 1], [1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_s_curve_t6",
        "positions": [[0, 1], [1, 1], [1, 2], [0, 3], [-1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_s_curve_t6",
        "positions": [[0, 1], [-1, 1], [-1, 2], [0, 3], [1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_funnel_t6",
        "positions": [[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_funnel_t6",
        "positions": [[0, 1], [0, 2], [-1, 1], [-1, 2], [-1, 3]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_gate_entrance_t6",
        "positions": [[0, 1], [0, 2], [1, 2], [-1, 3], [1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_gate_entrance_t6",
        "positions": [[0, 1], [0, 2], [-1, 2], [1, 3], [-1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_outer_detour_t6",
        "positions": [[-1, 1], [0, 1], [1, 1], [1, 2], [0, 3], [1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_outer_detour_t6",
        "positions": [[-1, 1], [0, 1], [1, 1], [-1, 2], [0, 3], [-1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "left_narrow_door_t6",
        "positions": [[0, 1], [1, 1], [-1, 2], [0, 2], [1, 3], [-1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
    {
        "variant_id": "right_narrow_door_t6",
        "positions": [[0, 1], [-1, 1], [1, 2], [0, 2], [-1, 3], [1, 4]],
        "material": "minecraft:cobblestone",
        "height": 2,
        "target_local": [0, 6],
    },
]

PATH_OBSTACLE_VARIANTS: List[Dict[str, object]] = list(PATH_OBSTACLE_VARIANTS_CENTER_Z3)


def resolve_path_obstacle_variants(
    variant_set: str,
    *,
    material_override: str = "",
    height_override: int = 0,
) -> List[Dict[str, object]]:
    variant_set = str(variant_set or "center_z3").strip().lower()
    if variant_set in {"hard_v4", "hard_v4_o2_target_funnel", "o2_target_funnel"}:
        raise ValueError(
            "Deprecated path obstacle variant set: "
            f"{variant_set}. Use hard_v5_o2_open_path or hard_v6_maze_t6 instead."
        )
    if variant_set == "center_z3":
        variants = PATH_OBSTACLE_VARIANTS_CENTER_Z3
    elif variant_set == "hard_v2":
        variants = PATH_OBSTACLE_VARIANTS_HARD_V2
    elif variant_set == "hard_v3":
        variants = PATH_OBSTACLE_VARIANTS_HARD_V3
    elif variant_set in {"hard_v5", "hard_v5_o2_open_path", "o2_open_path"}:
        variants = PATH_OBSTACLE_VARIANTS_HARD_V5_OPEN_PATH
    elif variant_set in {"hard_v6", "hard_v6_maze_t6", "maze_t6"}:
        variants = PATH_OBSTACLE_VARIANTS_HARD_V6_MAZE_T6
    else:
        raise ValueError(f"Unknown path obstacle variant set: {variant_set}")

    resolved: List[Dict[str, object]] = []
    for variant in variants:
        item = copy.deepcopy(variant)
        if material_override:
            item["material"] = str(material_override)
        if int(height_override or 0) > 0:
            item["height"] = max(1, min(3, int(height_override)))
        resolved.append(item)
    return resolved


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--eval-instances", type=int, default=16)
    parser.add_argument("--final-instances", type=int, default=32)
    parser.add_argument("--skip-final-bank", action="store_true")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--env-conf-dir", type=str, default=DEFAULT_ENV_CONF_DIR)
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument(
        "--mine-layout-backend",
        type=str,
        default="procedural",
        choices=["auto", "template", "procedural"],
    )
    parser.add_argument(
        "--mine-anchor-mode",
        type=str,
        default="source_or_fallback",
        choices=["source_or_fallback", "procedural_only"],
    )
    parser.add_argument("--factor-levels-json", type=str, default=json.dumps(DEFAULT_FACTOR_LEVELS))
    parser.add_argument("--bake-goals", action="store_true")
    parser.add_argument("--bake-goal-base-seed", type=int, default=1)
    parser.add_argument("--bake-goal-episode-retries", type=int, default=2)
    parser.add_argument("--bake-goal-save-debug-assets", action="store_true")
    parser.add_argument(
        "--goal-bake-world-mode",
        type=str,
        default="clean_path",
        choices=["clean_path", "same"],
        help="Use a clean companion path world for goal baking, or bake in the obstacle world itself.",
    )
    parser.add_argument(
        "--p-goal-pose-protocol",
        type=str,
        default="center_z3",
        choices=["center_z3", "front_close", "legacy_scale"],
        help="P-obstacle same-world goal camera hint protocol.",
    )
    parser.add_argument(
        "--path-obstacle-variant-set",
        type=str,
        default="center_z3",
        choices=[
            "center_z3",
            "hard_v2",
            "hard_v3",
            "hard_v5",
            "hard_v5_o2_open_path",
            "o2_open_path",
            "hard_v6",
            "hard_v6_maze_t6",
            "maze_t6",
        ],
        help="Which predefined straight-tunnel P-obstacle variant bank to generate.",
    )
    parser.add_argument(
        "--path-obstacle-material",
        type=str,
        default="",
        help="Optional material override for every P-obstacle variant, e.g. minecraft:obsidian.",
    )
    parser.add_argument(
        "--path-obstacle-height",
        type=int,
        default=0,
        help="Optional height override for every P-obstacle variant. 0 keeps each variant default.",
    )
    parser.add_argument(
        "--path-progress-reward-per-zone",
        type=float,
        default=0.25,
        help="Per-zone shaping reward written into generated maze task YAMLs.",
    )
    parser.add_argument("--max-instance-attempts", type=int, default=6)
    parser.add_argument("--bank-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_task_names(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_factor_levels(raw: str, task_name: str) -> Dict[str, int]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--factor-levels-json must decode to a JSON object.")
    return normalize_factor_levels(task_name, data)


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    command = list(cmd)
    is_goal_bake = "minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets" in command
    if is_goal_bake:
        try:
            task_group_path = Path(command[command.index("--task-group-path") + 1]).resolve()
            (task_group_path / "logs").mkdir(parents=True, exist_ok=True)
            (task_group_path / "crash-reports").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if str(env.get("DISABLE_XVFB_RUN", "0")) != "1" and not str(env.get("DISPLAY", "")).strip():
            xvfb_run = shutil.which("xvfb-run")
            if xvfb_run:
                command = [
                    xvfb_run,
                    "-a",
                    "-s",
                    str(env.get("XVFB_SCREEN_ARGS", "-screen 0 1280x1024x24")),
                    *command,
                ]
    print("[interaction-prepare-mine-p-obstacle-assets] running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def attach_path_obstacle_variant(suggestions: Dict[str, object], variant: Dict[str, object]) -> Dict[str, object]:
    merged = dict(suggestions)
    merged.update(
        {
            "mine_blueprint_id": "straight_tunnel",
            "mine_path_obstacle_variant_id": str(variant["variant_id"]),
            "mine_path_obstacle_positions": [list(item) for item in variant["positions"]],
            "mine_path_obstacle_material": str(variant.get("material") or "minecraft:cobblestone"),
            "mine_path_obstacle_height": int(variant.get("height") or 1),
        }
    )
    return merged


def attach_path_arena_controls(suggestions: Dict[str, object], variant_set: str) -> Dict[str, object]:
    merged = dict(suggestions)
    normalized_variant_set = str(variant_set or "").strip().lower()
    if normalized_variant_set == "hard_v3":
        merged.update(
            {
                "mine_disable_spawn_jitter": True,
                "mine_arena_mode": "wide_symmetric",
                "mine_arena_half_width": 5,
                "mine_arena_back_z": -3,
                "mine_arena_front_buffer": 3,
                "mine_arena_wall_height": 3,
                "mine_arena_roof": False,
            }
        )
        return merged
    if normalized_variant_set in {"hard_v4", "hard_v4_o2_target_funnel", "o2_target_funnel"}:
        raise ValueError(
            "Deprecated path obstacle variant set: "
            f"{normalized_variant_set}. Use hard_v5_o2_open_path or hard_v6_maze_t6 instead."
        )
    if normalized_variant_set not in {
        "hard_v5",
        "hard_v5_o2_open_path",
        "o2_open_path",
        "hard_v6",
        "hard_v6_maze_t6",
        "maze_t6",
    }:
        return merged
    funnel_start_z = 4 if normalized_variant_set in {"hard_v6", "hard_v6_maze_t6", "maze_t6"} else 3
    merged.update(
        {
            "mine_disable_spawn_jitter": True,
            "mine_arena_mode": "o2_target_funnel",
            "mine_arena_half_width": 4,
            "mine_arena_back_z": -3,
            "mine_arena_front_buffer": 3,
            "mine_arena_wall_height": 3,
            "mine_arena_funnel_start_z": int(funnel_start_z),
            "mine_arena_roof": False,
        }
    )
    return merged


def build_clean_goal_plan_rows(plan_rows: List[Dict]) -> List[Dict]:
    clean_rows: List[Dict] = []
    for row in plan_rows:
        clean_row = copy.deepcopy(row)
        task_name = str(clean_row.get("task_config_name") or clean_row.get("task_key") or "")
        task_key = canonical_task_key(task_name)
        clean_levels = normalize_factor_levels(task_key, clean_row.get("factor_levels") or {})
        if "P" in clean_levels:
            clean_levels["P"] = 0

        original_suggestions = dict(clean_row.get("world_generation_suggestions") or {})
        clean_suggestions = factor_levels_to_worldgen_suggestions(task_key, clean_levels)
        clean_suggestions["mine_blueprint_id"] = str(original_suggestions.get("mine_blueprint_id") or "straight_tunnel")
        if original_suggestions.get("mine_target_sign"):
            clean_suggestions["mine_target_sign"] = original_suggestions.get("mine_target_sign")
        if original_suggestions.get("mine_target_local"):
            clean_suggestions["mine_target_local"] = original_suggestions.get("mine_target_local")
        clean_suggestions["notes"] = (
            "Clean companion world for P-obstacle goal baking. "
            "Obstacle commands are intentionally omitted so the goal image remains factor-neutral."
        )

        split_label = classify_factor_split(task_key, clean_levels)
        clean_row["factor_levels"] = clean_levels
        clean_row["primary_failure_mode_majority"] = "p_obstacle_clean_goal_bake"
        clean_row["primary_factors_majority"] = [code for code, value in clean_levels.items() if int(value) > 0]
        clean_row["severity_majority"] = max([int(value) for value in clean_levels.values()] or [0])
        clean_row["requested_split_label"] = str(split_label)
        clean_row["computed_split_label"] = str(split_label)
        clean_row["hard_factor_count"] = int(hard_factor_count(task_key, clean_levels))
        clean_row["world_generation_suggestions"] = clean_suggestions
        clean_rows.append(clean_row)
    return clean_rows


def _load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: Dict) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def copy_baked_goal_metadata(
    *,
    clean_generated_dir: Path,
    obstacle_generated_dir: Path,
    task_names: List[str],
) -> Dict[str, object]:
    copied: List[Dict[str, str]] = []
    for task_name in task_names:
        clean_yaml_path = clean_generated_dir / f"{task_name}.yaml"
        obstacle_yaml_path = obstacle_generated_dir / f"{task_name}.yaml"
        if not clean_yaml_path.exists():
            raise FileNotFoundError(f"Clean goal YAML missing: {clean_yaml_path}")
        if not obstacle_yaml_path.exists():
            raise FileNotFoundError(f"Obstacle YAML missing: {obstacle_yaml_path}")

        clean_yaml = _load_yaml(clean_yaml_path)
        obstacle_yaml = _load_yaml(obstacle_yaml_path)
        goal_spec_path = str(clean_yaml.get("baked_goal_spec_path") or "").strip()
        if not goal_spec_path:
            raise RuntimeError(f"Clean goal bake did not write baked_goal_spec_path: {clean_yaml_path}")
        if not Path(goal_spec_path).exists():
            raise FileNotFoundError(f"Clean baked goal spec does not exist: {goal_spec_path}")

        for key in (
            "baked_goal_spec_path",
            "baked_goal_dir",
            "baked_goal_protocol",
            "baked_goal_seed",
        ):
            if key in clean_yaml:
                obstacle_yaml[key] = clean_yaml[key]
        obstacle_yaml["baked_goal_mode"] = "fixed_from_clean_path_goal"
        obstacle_yaml["baked_goal_source_task_group_path"] = str(clean_generated_dir.resolve())
        _dump_yaml(obstacle_yaml_path, obstacle_yaml)
        copied.append(
            {
                "task_config_name": str(task_name),
                "goal_spec_path": goal_spec_path,
                "clean_yaml_path": str(clean_yaml_path.resolve()),
                "obstacle_yaml_path": str(obstacle_yaml_path.resolve()),
            }
        )

    clean_manifest_path = clean_generated_dir / "worldgen_manifest.json"
    obstacle_manifest_path = obstacle_generated_dir / "worldgen_manifest.json"
    if clean_manifest_path.exists() and obstacle_manifest_path.exists():
        clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
        obstacle_manifest = json.loads(obstacle_manifest_path.read_text(encoding="utf-8"))
        clean_by_task = {
            str(row.get("task_config_name") or ""): row
            for row in clean_manifest.get("tasks") or []
            if isinstance(row, dict)
        }
        for row in obstacle_manifest.get("tasks") or []:
            if not isinstance(row, dict):
                continue
            task_name = str(row.get("task_config_name") or "")
            clean_row = clean_by_task.get(task_name) or {}
            if clean_row.get("baked_goal"):
                row["baked_goal"] = clean_row.get("baked_goal")
            if clean_row.get("baked_goal_spec_path"):
                row["baked_goal_spec_path"] = clean_row.get("baked_goal_spec_path")
            row["baked_goal_source_world_mode"] = "clean_path"
            row["baked_goal_source_task_group_path"] = str(clean_generated_dir.resolve())
        obstacle_manifest["goal_baked"] = True
        obstacle_manifest["goal_bake_world_mode"] = "clean_path"
        obstacle_manifest["goal_bake_source_task_group_path"] = str(clean_generated_dir.resolve())
        obstacle_manifest_path.write_text(json.dumps(obstacle_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "mode": "clean_path",
        "clean_generated_task_group_dir": str(clean_generated_dir.resolve()),
        "copied": copied,
    }


def build_plan_row(
    *,
    task_name: str,
    factor_levels: Dict[str, int],
    variant: Dict[str, object],
    bank_label: str,
    instance_idx: int,
    base_seed: int,
    p_goal_pose_protocol: str = "center_z3",
    variant_set: str = "center_z3",
) -> Dict:
    task_key = canonical_task_key(task_name)
    normalized_levels = normalize_factor_levels(task_key, factor_levels)
    split_label = classify_factor_split(task_key, normalized_levels)
    target_local = list(variant.get("target_local") or PATH_TARGET_LOCAL)
    suggestions = {
        **factor_levels_to_worldgen_suggestions(task_key, normalized_levels),
        "mine_target_local": target_local,
        "notes": (
            f"Straight P obstacle row for {bank_label}, instance={int(instance_idx)}. "
            f"Exact path obstacle variant {variant['variant_id']}; target_local={target_local}. "
            "The target-adjacent occlusion row is reserved for O-factor composition."
        ),
    }
    suggestions = attach_path_obstacle_variant(suggestions, variant)
    suggestions = attach_path_arena_controls(suggestions, variant_set)
    suggestions["mine_p_goal_pose_protocol"] = str(p_goal_pose_protocol or "front_close")
    return {
        "task_config_name": str(task_name),
        "task_key": str(task_key),
        "primary_failure_mode_majority": f"p_obstacle_{bank_label}",
        "primary_factors_majority": [code for code, value in normalized_levels.items() if int(value) > 0],
        "severity_majority": max([int(value) for value in normalized_levels.values()] or [0]),
        "factor_levels": normalized_levels,
        "layout_seed": stable_layout_seed(
            task_config_name=task_name,
            bank_label=bank_label,
            instance_idx=instance_idx,
            base_seed=base_seed,
            factor_levels=normalized_levels,
            layout_case=f"straight_{variant['variant_id']}",
        ),
        "template_index": int(instance_idx),
        "requested_split_label": str(split_label),
        "computed_split_label": str(split_label),
        "hard_factor_count": int(hard_factor_count(task_key, normalized_levels)),
        "requested_layout_case": "straight",
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": suggestions,
    }


def build_collect_plan_rows(
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    p_goal_pose_protocol: str = "center_z3",
    variant_set: str = "center_z3",
) -> List[Dict]:
    return [
        build_plan_row(
            task_name=task_name,
            factor_levels=factor_levels,
            variant=PATH_OBSTACLE_VARIANTS[0],
            bank_label="collect_plan",
            instance_idx=0,
            base_seed=base_seed,
            p_goal_pose_protocol=p_goal_pose_protocol,
            variant_set=variant_set,
        )
        for task_name in task_names
    ]


def build_collect_block_plan_rows(
    *,
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    block_index: int,
    p_goal_pose_protocol: str = "center_z3",
    variant_set: str = "center_z3",
) -> List[Dict]:
    variant = PATH_OBSTACLE_VARIANTS[int(block_index) - 1]
    return [
        build_plan_row(
            task_name=task_name,
            factor_levels=factor_levels,
            variant=variant,
            bank_label=f"collect_block_{int(block_index):02d}",
            instance_idx=0,
            base_seed=base_seed,
            p_goal_pose_protocol=p_goal_pose_protocol,
            variant_set=variant_set,
        )
        for task_name in task_names
    ]


def generate_bank(
    *,
    bank_dir: Path,
    bank_label: str,
    instances: int,
    task_names: List[str],
    factor_levels: Dict[str, int],
    args,
    env: Dict[str, str],
) -> None:
    bank_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = max(1, int(args.max_instance_attempts))
    worker_count = max(1, int(args.bank_workers))
    instance_worlds: List[Dict] = []
    env_conf_dir = Path(args.env_conf_dir)

    def build_instance(instance_idx: int) -> Dict:
        instance_dir = bank_dir / f"instance_{int(instance_idx):03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        variant = PATH_OBSTACLE_VARIANTS[int(instance_idx) % len(PATH_OBSTACLE_VARIANTS)]
        base_plan_rows = [
            build_plan_row(
                task_name=task_name,
                factor_levels=factor_levels,
                variant=variant,
                bank_label=bank_label,
                instance_idx=instance_idx,
                base_seed=int(args.base_seed),
                p_goal_pose_protocol=str(args.p_goal_pose_protocol),
                variant_set=str(args.path_obstacle_variant_set),
            )
            for task_name in task_names
        ]
        plan_json_path = instance_dir / "worldgen_plan.json"
        generated_groups_root = instance_dir / "generated_task_groups"
        generated_groups_root.mkdir(parents=True, exist_ok=True)
        attempt_records: List[Dict] = []
        selected_entry = None
        last_exc = None
        for attempt_idx in range(max_attempts):
            plan_rows = copy.deepcopy(base_plan_rows) if attempt_idx == 0 else resample_plan_rows(base_plan_rows, attempt_idx)
            plan_json_path.write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            generated_dir = None
            goal_bake_record: Dict[str, object] = {}
            try:
                before_dirs = list_child_dirs(generated_groups_root)
                run_command(
                    build_worldgen_cmd(
                        plan_json_path=plan_json_path,
                        env_conf_dir=env_conf_dir,
                        out_dir=generated_groups_root,
                        mine_layout_backend=args.mine_layout_backend,
                        mine_anchor_mode=args.mine_anchor_mode,
                        path_progress_reward_per_zone=float(args.path_progress_reward_per_zone),
                    ),
                    env,
                )
                generated_dir = detect_generated_dir(generated_groups_root, before_dirs)
                if args.bake_goals:
                    if str(args.goal_bake_world_mode) == "clean_path":
                        clean_attempt_dir = instance_dir / "goal_bake_clean_path" / f"attempt_{int(attempt_idx):02d}"
                        clean_plan_json_path = clean_attempt_dir / "worldgen_plan.json"
                        clean_generated_groups_root = clean_attempt_dir / "generated_task_groups"
                        clean_generated_groups_root.mkdir(parents=True, exist_ok=True)
                        clean_plan_rows = build_clean_goal_plan_rows(plan_rows)
                        clean_plan_json_path.write_text(
                            json.dumps(clean_plan_rows, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        clean_before_dirs = list_child_dirs(clean_generated_groups_root)
                        run_command(
                            build_worldgen_cmd(
                                plan_json_path=clean_plan_json_path,
                                env_conf_dir=env_conf_dir,
                                out_dir=clean_generated_groups_root,
                                mine_layout_backend=args.mine_layout_backend,
                                mine_anchor_mode=args.mine_anchor_mode,
                                path_progress_reward_per_zone=float(args.path_progress_reward_per_zone),
                            ),
                            env,
                        )
                        clean_generated_dir = detect_generated_dir(clean_generated_groups_root, clean_before_dirs)
                        run_command(
                            build_bake_goal_cmd(
                                task_group=f"{bank_label}_instance_{instance_idx:03d}_clean_goal",
                                task_group_path=clean_generated_dir,
                                tasks=task_names,
                                protocol_name=args.protocol,
                                base_seed=int(args.bake_goal_base_seed),
                                episode_retries=int(args.bake_goal_episode_retries),
                                save_debug_assets=bool(args.bake_goal_save_debug_assets),
                            ),
                            env,
                        )
                        goal_bake_record = copy_baked_goal_metadata(
                            clean_generated_dir=clean_generated_dir,
                            obstacle_generated_dir=generated_dir,
                            task_names=task_names,
                        )
                        goal_bake_record["clean_plan_json"] = str(clean_plan_json_path.resolve())
                    else:
                        run_command(
                            build_bake_goal_cmd(
                                task_group=f"{bank_label}_instance_{instance_idx:03d}",
                                task_group_path=generated_dir,
                                tasks=task_names,
                                protocol_name=args.protocol,
                                base_seed=int(args.bake_goal_base_seed),
                                episode_retries=int(args.bake_goal_episode_retries),
                                save_debug_assets=bool(args.bake_goal_save_debug_assets),
                            ),
                            env,
                        )
                        goal_bake_record = {
                            "mode": "same",
                            "generated_task_group_dir": str(generated_dir.resolve()),
                        }
                manifest_path = generated_dir / "worldgen_manifest.json"
                selected_entry = {
                    "instance_idx": int(instance_idx),
                    "split_label": str(bank_label),
                    "layout_case": "straight",
                    "path_obstacle_variant_id": str(variant["variant_id"]),
                    "plan_json": str(plan_json_path.resolve()),
                    "generated_task_group_dir": str(generated_dir.resolve()),
                    "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
                    "plan_rows": plan_rows,
                    "goal_bake": goal_bake_record,
                    "attempt_idx": int(attempt_idx),
                    "attempt_number": int(attempt_idx + 1),
                    "attempt_count": int(attempt_idx + 1),
                }
                attempt_records.append(
                    {
                        "attempt_idx": int(attempt_idx),
                        "attempt_number": int(attempt_idx + 1),
                        "status": "success",
                        "plan_json": str(plan_json_path.resolve()),
                        "generated_task_group_dir": str(generated_dir.resolve()),
                        "goal_bake": goal_bake_record,
                        "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                    }
                )
                break
            except Exception as exc:
                last_exc = exc
                failure_payload = {
                    "instance_idx": int(instance_idx),
                    "split_label": str(bank_label),
                    "layout_case": "straight",
                    "attempt_idx": int(attempt_idx),
                    "attempt_number": int(attempt_idx + 1),
                    "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                    "plan_json": str(plan_json_path.resolve()),
                    "generated_task_group_dir": str(generated_dir.resolve()) if generated_dir is not None else "",
                    "error_message": str(exc),
                }
                attempt_records.append({**failure_payload, "status": "failed"})
                write_attempt_failure(instance_dir, attempt_idx, failure_payload)
        if selected_entry is None:
            raise RuntimeError(
                f"Failed to build P obstacle bank instance after {max_attempts} attempts: "
                f"bank={bank_label} instance={instance_idx}"
            ) from last_exc
        selected_entry["attempt_records"] = attempt_records
        return selected_entry

    total_instances = max(0, int(instances))
    if worker_count == 1:
        for instance_idx in range(total_instances):
            instance_worlds.append(build_instance(instance_idx))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(build_instance, instance_idx): int(instance_idx) for instance_idx in range(total_instances)}
            results: Dict[int, Dict] = {}
            for future in as_completed(future_map):
                instance_idx = future_map[future]
                results[instance_idx] = future.result()
            for instance_idx in range(total_instances):
                instance_worlds.append(results[instance_idx])

    bank_manifest = {
        "split_label": str(bank_label),
        "tasks": task_names,
        "instances_per_split": int(instances),
        "bank_dir": str(bank_dir.resolve()),
        "bank_workers": int(worker_count),
        "protocol": str(args.protocol),
        "bake_goals": bool(args.bake_goals),
        "goal_bake_world_mode": str(args.goal_bake_world_mode),
        "p_goal_pose_protocol": str(args.p_goal_pose_protocol),
        "bake_goal_base_seed": int(args.bake_goal_base_seed),
        "factor_levels": factor_levels,
        "layout_cases": ["straight"],
        "path_obstacle_variants": PATH_OBSTACLE_VARIANTS,
        "path_obstacle_variant_set": str(args.path_obstacle_variant_set),
        "path_obstacle_material_override": str(args.path_obstacle_material or ""),
        "path_obstacle_height_override": int(args.path_obstacle_height or 0),
        "path_progress_reward_per_zone": float(args.path_progress_reward_per_zone),
        "instance_worlds": instance_worlds,
    }
    (bank_dir / "bank_manifest.json").write_text(
        json.dumps(bank_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    global PATH_OBSTACLE_VARIANTS
    PATH_OBSTACLE_VARIANTS = resolve_path_obstacle_variants(
        str(args.path_obstacle_variant_set),
        material_override=str(args.path_obstacle_material or ""),
        height_override=int(args.path_obstacle_height or 0),
    )
    task_names = parse_task_names(args.tasks)
    if not task_names:
        raise ValueError("No tasks provided.")
    factor_levels = parse_factor_levels(args.factor_levels_json, task_names[0])
    out_root = (Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    collect_plan_path = out_root / "collect_plan.json"
    collect_plan_path.write_text(
        json.dumps(
            build_collect_plan_rows(
                task_names,
                factor_levels,
                int(args.base_seed),
                p_goal_pose_protocol=str(args.p_goal_pose_protocol),
                variant_set=str(args.path_obstacle_variant_set),
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    collect_block_plan_dir = out_root / "collect_block_plans"
    collect_block_plan_dir.mkdir(parents=True, exist_ok=True)
    collect_block_plan_paths: List[str] = []
    for block_index in range(1, len(PATH_OBSTACLE_VARIANTS) + 1):
        block_rows = build_collect_block_plan_rows(
            task_names=task_names,
            factor_levels=factor_levels,
            base_seed=int(args.base_seed),
            block_index=block_index,
            p_goal_pose_protocol=str(args.p_goal_pose_protocol),
            variant_set=str(args.path_obstacle_variant_set),
        )
        block_plan_path = collect_block_plan_dir / f"block_{int(block_index):03d}_collect_plan.json"
        block_plan_path.write_text(json.dumps(block_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        collect_block_plan_paths.append(str(block_plan_path.resolve()))

    eval_bank_dir = out_root / "eval_bank"
    final_eval_bank_dir = out_root / "final_eval_bank"
    generate_bank(
        bank_dir=eval_bank_dir,
        bank_label="eval_bank",
        instances=int(args.eval_instances),
        task_names=task_names,
        factor_levels=factor_levels,
        args=args,
        env=env,
    )
    if not bool(args.skip_final_bank) and int(args.final_instances) > 0:
        generate_bank(
            bank_dir=final_eval_bank_dir,
            bank_label="final_eval_bank",
            instances=int(args.final_instances),
            task_names=task_names,
            factor_levels=factor_levels,
            args=args,
            env=env,
        )
        final_eval_bank_dir_value = str(final_eval_bank_dir.resolve())
    else:
        final_eval_bank_dir_value = ""

    manifest = {
        "asset_type": "mine_p_straight_obstacle_generalization_assets",
        "tasks": task_names,
        "factor_levels": factor_levels,
        "layout_cases": ["straight"],
        "path_obstacle_variants": PATH_OBSTACLE_VARIANTS,
        "path_obstacle_variant_set": str(args.path_obstacle_variant_set),
        "path_obstacle_material_override": str(args.path_obstacle_material or ""),
        "path_obstacle_height_override": int(args.path_obstacle_height or 0),
        "path_progress_reward_per_zone": float(args.path_progress_reward_per_zone),
        "collect_plan_path": str(collect_plan_path.resolve()),
        "collect_block_plan_dir": str(collect_block_plan_dir.resolve()),
        "collect_block_plan_paths": collect_block_plan_paths,
        "eval_bank_dir": str(eval_bank_dir.resolve()),
        "final_eval_bank_dir": final_eval_bank_dir_value,
        "skip_final_bank": bool(args.skip_final_bank),
        "bake_goals": bool(args.bake_goals),
        "goal_bake_world_mode": str(args.goal_bake_world_mode),
        "p_goal_pose_protocol": str(args.p_goal_pose_protocol),
    }
    (out_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
