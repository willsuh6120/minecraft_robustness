import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from minestudio.tutorials.inference.evaluate_rocket.path_progress_reward import (
    build_path_progress_reward_config,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    classify_factor_split,
    canonical_task_key,
    factor_levels_to_worldgen_suggestions,
    hard_factor_count,
    normalize_factor_levels,
    normalize_worldgen_suggestions,
)


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_BENCHMARK_TASK_CONFIG_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "task_configs" / "rocket_interaction"
PROCEDURAL_MINE_TASKS = {"mine_coal", "mine_emerald"}
PROCEDURAL_MINE_SAFE_ANCHOR_Y = 160.0
PROCEDURAL_MINE_SAFE_ANCHOR_X_CHOICES = [-2048, -1536, -1024, 1024, 1536, 2048]
PROCEDURAL_MINE_SAFE_ANCHOR_Z_CHOICES = [1024, 1536, 2048, 2560, 3072]
PROCEDURAL_MINE_ROLLOUT_WARMUP_STEPS = 30


TARGET_HINTS: Dict[str, Tuple[int, int]] = {
    "hunt_sheep_right_fence": (4, 7),
    "hunt_cow_do_not_touch_sheep": (3, 4),
    "mine_emerald": (1, 5),
    "mine_coal": (-1, 5),
    "interact_left_chest": (-2, 5),
    "open_door_then_open_chest_in_house": (0, 5),
    "approach_nearest_village": (0, 8),
    "approach_ocean": (0, 8),
    "set_fire_on_tree": (0, 6),
    "use_bucket_get_lava": (1, 4),
    "place_minecart_on_rail": (0, 4),
    "place_oak_door_on_diamond_block": (0, 4),
}

COMMAND_LAYOUT_TASKS = {
    "hunt_sheep_right_fence",
    "hunt_cow_do_not_touch_sheep",
    "interact_left_chest",
    "use_bucket_get_lava",
    "place_minecart_on_rail",
    "place_oak_door_on_diamond_block",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=str, required=True)
    parser.add_argument("--env-conf-dir", type=str, default=DEFAULT_ENV_CONF_DIR)
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/generated_task_group")
    parser.add_argument("--path-progress-reward-per-zone", type=float, default=0.25)
    parser.add_argument(
        "--mine-layout-backend",
        type=str,
        default="auto",
        choices=["auto", "template", "procedural"],
    )
    parser.add_argument(
        "--mine-anchor-mode",
        type=str,
        default="source_or_fallback",
        choices=["source_or_fallback", "procedural_only"],
        help="For procedural mine worlds, either reuse source spawn anchors when available or force a source-free safe anchor.",
    )
    parser.add_argument("--only-trainable", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def format_relative_token(prefix: str, value: int) -> str:
    if value == 0:
        return prefix
    return f"{prefix}{int(value)}"


def shift_relative_token(token: str, delta: int) -> str:
    if delta == 0:
        return token
    token = str(token)
    if not token or token[0] not in {"~", "^"}:
        return token
    prefix = token[0]
    suffix = token[1:]
    if suffix == "":
        base = 0
    else:
        try:
            base = int(float(suffix))
        except ValueError:
            return token
    return format_relative_token(prefix, base + int(delta))


def shift_command_layout(command: str, dx: int = 0, dz: int = 0) -> str:
    tokens = str(command).split()
    if not tokens:
        return command
    op = tokens[0].lstrip("/").lower()
    if op == "setblock" and len(tokens) >= 5:
        tokens[1] = shift_relative_token(tokens[1], dx)
        tokens[3] = shift_relative_token(tokens[3], dz)
    elif op == "fill" and len(tokens) >= 8:
        tokens[1] = shift_relative_token(tokens[1], dx)
        tokens[3] = shift_relative_token(tokens[3], dz)
        tokens[4] = shift_relative_token(tokens[4], dx)
        tokens[6] = shift_relative_token(tokens[6], dz)
    return " ".join(tokens)


def shift_custom_commands(commands: List[str], dx: int = 0, dz: int = 0) -> List[str]:
    return [shift_command_layout(command, dx=dx, dz=dz) for command in commands or []]


def shift_summon_mobs(summon_mobs: List[Dict], dx: int = 0, dz: int = 0) -> List[Dict]:
    shifted: List[Dict] = []
    for item in summon_mobs or []:
        new_item = copy.deepcopy(item)
        if "range_x" in new_item and isinstance(new_item["range_x"], list) and len(new_item["range_x"]) == 2:
            new_item["range_x"] = [int(new_item["range_x"][0]) + dx, int(new_item["range_x"][1]) + dx]
        if "range_z" in new_item and isinstance(new_item["range_z"], list) and len(new_item["range_z"]) == 2:
            new_item["range_z"] = [int(new_item["range_z"][0]) + dz, int(new_item["range_z"][1]) + dz]
        shifted.append(new_item)
    return shifted


def shift_spawn_positions(spawn_positions: List[Dict], dx: int = 0, dz: int = 0) -> List[Dict]:
    shifted: List[Dict] = []
    for item in spawn_positions or []:
        new_item = copy.deepcopy(item)
        position = list(new_item.get("position", []))
        if len(position) >= 3:
            position[0] = float(position[0]) + float(dx)
            position[2] = float(position[2]) + float(dz)
            new_item["position"] = position
        shifted.append(new_item)
    return shifted


def orient_spawn_positions(spawn_positions: List[Dict], yaw: float | None = None, pitch: float | None = None) -> List[Dict]:
    oriented: List[Dict] = []
    for item in spawn_positions or []:
        new_item = copy.deepcopy(item)
        if yaw is not None:
            new_item["yaw"] = float(yaw)
        if pitch is not None:
            new_item["pitch"] = float(pitch)
        oriented.append(new_item)
    return oriented


def add_spawn_safety_commands(data: Dict, commands: List[str]):
    spawn_positions = list(data.get("spawn_positions") or [])
    if not spawn_positions:
        return
    spawn_position = list((spawn_positions[0] or {}).get("position") or [])
    if len(spawn_position) < 3:
        return
    try:
        x = int(math.floor(float(spawn_position[0])))
        y = int(round(float(spawn_position[1])))
        z = int(math.floor(float(spawn_position[2])))
    except Exception:
        return
    # Carve a small walkable spawn pad rather than a 1x2x1 shaft.
    # The old shaft removed suffocation but could still trap the agent in a pit
    # that required jumping to move. A 3x2x3 cavity with a matching floor pad
    # preserves the spawn shift while keeping the initial state walkable.
    commands.append(f"/fill {x - 1} {y} {z - 1} {x + 1} {y + 1} {z + 1} minecraft:air")
    commands.append(f"/fill {x - 1} {y - 1} {z - 1} {x + 1} {y - 1} {z + 1} minecraft:stone")


def build_layout_rng(plan_row: Dict) -> random.Random:
    raw_seed = plan_row.get("layout_seed")
    if raw_seed in (None, ""):
        stable_payload = {
            "task_config_name": str(plan_row.get("task_config_name", "")),
            "factor_levels": normalize_factor_levels(
                canonical_task_key(plan_row.get("task_config_name", "")),
                plan_row.get("factor_levels") or {},
            ),
            "requested_split_label": str(plan_row.get("requested_split_label", "")),
            "template_index": int(plan_row.get("template_index", 0) or 0),
        }
        raw_seed = abs(hash(json.dumps(stable_payload, sort_keys=True, ensure_ascii=False))) % (2**31 - 1)
    try:
        seed = int(raw_seed)
    except Exception:
        seed = 0
    return random.Random(seed)


def spawn_shift_from_factor_levels(task_key: str, factor_levels: Dict[str, int], rng: Optional[random.Random] = None) -> Tuple[int, int]:
    levels = normalize_factor_levels(task_key, factor_levels)
    distance_level = int(levels.get("R", 0))
    dz_choices = {
        0: [0],
        1: [-2, -3, -4],
        2: [-5, -6, -7],
        3: [-8, -10, -12],
    }
    choices = dz_choices.get(distance_level, [-10])
    dz = int((rng or random).choice(choices))
    return 0, dz


def spawn_heading_offset_from_factor_levels(task_key: str, factor_levels: Dict[str, int], rng: Optional[random.Random] = None) -> float:
    levels = normalize_factor_levels(task_key, factor_levels)
    heading_level = int(levels.get("H", 0))
    choices = {
        0: [0.0],
        1: [12.0, 16.0, 20.0],
        2: [28.0, 35.0, 42.0],
        3: [50.0, 60.0, 70.0],
    }.get(heading_level, [60.0])
    return float((rng or random).choice(choices))


def normalize_yaw_deg(yaw: float) -> float:
    yaw = float(yaw)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def yaw_delta_deg(base_yaw: float, spawn_yaw: float) -> float:
    return normalize_yaw_deg(float(spawn_yaw) - float(base_yaw))


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_target_blocks(task_key: str, data: Dict) -> List[Dict[str, object]]:
    target_types = {
        "mine_coal": ("coal",),
        "mine_emerald": ("emerald",),
    }.get(task_key, ())
    entries: List[Dict[str, object]] = []

    single_center = data.get("target_world_center")
    if isinstance(single_center, (list, tuple)) and len(single_center) == 3:
        entries.append(
            {
                "type": str(data.get("target_type") or ""),
                "world_center": [float(single_center[0]), float(single_center[1]), float(single_center[2])],
            }
        )

    raw_targets = data.get("target_blocks") or data.get("targets") or []
    if isinstance(raw_targets, dict):
        raw_targets = [raw_targets]
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            center = item.get("world_center") or item.get("target_world_center") or item.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) == 3):
                continue
            entries.append(
                {
                    "type": str(item.get("type") or item.get("target_type") or ""),
                    "world_center": [float(center[0]), float(center[1]), float(center[2])],
                }
            )

    normalized: List[Dict[str, object]] = []
    seen = set()
    for entry in entries:
        target_type = str(entry.get("type") or "").strip().lower()
        if target_types and target_type and not any(name in target_type for name in target_types):
            continue
        center = entry.get("world_center") or []
        if not (isinstance(center, list) and len(center) == 3):
            continue
        key = (
            target_type,
            round(float(center[0]), 4),
            round(float(center[1]), 4),
            round(float(center[2]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "type": target_type,
                "world_center": [float(center[0]), float(center[1]), float(center[2])],
            }
        )
    return normalized


def _shift_target_blocks(target_blocks: List[Dict[str, object]], dx: int = 0, dz: int = 0) -> List[Dict[str, object]]:
    shifted: List[Dict[str, object]] = []
    for entry in target_blocks or []:
        center = entry.get("world_center") or []
        if not (isinstance(center, list) and len(center) == 3):
            continue
        shifted.append(
            {
                "type": str(entry.get("type") or ""),
                "world_center": [
                    float(center[0]) + float(dx),
                    float(center[1]),
                    float(center[2]) + float(dz),
                ],
            }
        )
    return shifted


def annotate_target_metadata(task_key: str, source_data: Dict, generated_data: Dict, applied_changes: Dict) -> Tuple[Dict, Dict]:
    generated_data = copy.deepcopy(generated_data)
    configured_targets = _normalize_target_blocks(task_key, generated_data)
    source_kind = "generated_yaml"
    if not configured_targets:
        configured_targets = _normalize_target_blocks(task_key, source_data)
        source_kind = "source_yaml"
    if not configured_targets:
        return generated_data, {"configured_target": False, "target_source": "unavailable"}

    scene_shift = (applied_changes or {}).get("scene_shift") or {}
    dx = int(scene_shift.get("dx", 0) or 0)
    dz = int(scene_shift.get("dz", 0) or 0)
    if dx or dz:
        configured_targets = _shift_target_blocks(configured_targets, dx=dx, dz=dz)
        source_kind = f"{source_kind}+scene_shift"

    generated_data["target_blocks"] = copy.deepcopy(configured_targets)
    if len(configured_targets) == 1:
        generated_data["target_world_center"] = list(configured_targets[0]["world_center"])
        if str(configured_targets[0].get("type") or "").strip():
            generated_data["target_type"] = str(configured_targets[0]["type"])
    generated_data["target_metadata"] = {
        "source": "worldgen_annotation",
        "target_source": source_kind,
        "num_target_blocks": int(len(configured_targets)),
        "scene_shift_applied": {"dx": int(dx), "dz": int(dz)},
    }
    return generated_data, {
        "configured_target": True,
        "target_source": source_kind,
        "num_target_blocks": int(len(configured_targets)),
        "target_world_center": list(configured_targets[0]["world_center"]) if len(configured_targets) == 1 else None,
    }


def _look_angles_for_target(
    camera_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    *,
    eye_height: float = 1.62,
) -> tuple[float, float]:
    origin_x, origin_y, origin_z = camera_position
    target_x, target_y, target_z = target_position
    dx = float(target_x) - float(origin_x)
    dy = float(target_y) - (float(origin_y) + float(eye_height))
    dz = float(target_z) - float(origin_z)
    horizontal = max(1e-6, math.sqrt(dx * dx + dz * dz))
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = math.degrees(math.atan2(-dy, horizontal))
    return yaw, pitch


def _face_axes_from_label(face_label: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    normalized = str(face_label or "").strip().lower()
    mapping = {
        "-z": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
        "+z": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    }
    return mapping.get(normalized)


def _preferred_face_and_axes(data: Dict) -> tuple[str, tuple[float, float, float], tuple[float, float, float]] | None:
    explicit_face = str(
        data.get("goal_camera_preferred_face")
        or data.get("goal_preferred_face")
        or ""
    ).strip()
    explicit_axes = _face_axes_from_label(explicit_face)
    if explicit_axes is not None:
        outward, lateral = explicit_axes
        return explicit_face, outward, lateral
    spawn_positions = list(data.get("spawn_positions") or [])
    target_center = data.get("target_world_center")
    if not spawn_positions or not (isinstance(target_center, (list, tuple)) and len(target_center) >= 3):
        return None
    spawn_position = list((spawn_positions[0] or {}).get("position") or [])
    if len(spawn_position) < 3:
        return None
    try:
        dx = float(target_center[0]) - float(spawn_position[0])
        dz = float(target_center[2]) - float(spawn_position[2])
    except Exception:
        return None
    if abs(dz) >= abs(dx):
        if dz >= 0.0:
            return ("-z", (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
        return ("+z", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    if dx >= 0.0:
        return ("-x", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return ("+x", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def build_goal_pose_hints(task_key: str, data: Dict) -> List[Dict]:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return []
    target_center = data.get("target_world_center")
    if not (isinstance(target_center, (list, tuple)) and len(target_center) >= 3):
        return []
    face_axes = _preferred_face_and_axes(data)
    if face_axes is None:
        return []
    preferred_face, outward, lateral = face_axes
    target = (float(target_center[0]), float(target_center[1]), float(target_center[2]))
    target_type = str(data.get("target_type") or "")
    hints: List[Dict] = []
    procedural_layout = data.get("procedural_layout") if isinstance(data.get("procedural_layout"), dict) else {}
    look_target_center = procedural_layout.get("goal_camera_look_target_world_center")
    if isinstance(look_target_center, (list, tuple)) and len(look_target_center) >= 3:
        look_target = (
            float(look_target_center[0]),
            float(look_target_center[1]),
            float(look_target_center[2]),
        )
    else:
        look_target = target
    blueprint_id = str(procedural_layout.get("blueprint_id") or "")
    disable_hint_support_block = bool(
        task_key in PROCEDURAL_MINE_TASKS
        or str(procedural_layout.get("backend") or "").strip() == "procedural_mine_v1"
    )
    target_local = procedural_layout.get("target_local") if isinstance(procedural_layout.get("target_local"), (list, tuple)) else None
    spawn_positions = list(data.get("spawn_positions") or [])
    spawn_position = list((spawn_positions[0] or {}).get("position") or []) if spawn_positions else []
    if len(spawn_position) >= 2:
        base_feet_y = float(spawn_position[1])
    else:
        # Hint cameras should use standing feet Y, not block-center Y.
        # The previous +0.5 convention caused a systematic planned-vs-actual
        # teleport mismatch on full-block floors.
        base_feet_y = math.floor(float(target[1]))

    profile_specs: List[Dict[str, Any]] = []
    if (
        task_key in PROCEDURAL_MINE_TASKS
        and blueprint_id
        and isinstance(target_local, (list, tuple))
        and len(target_local) >= 2
        and len(spawn_position) >= 3
    ):
        tx = float(target_local[0])
        tz = float(target_local[1])
        if blueprint_id == "offset_chamber":
            base_anchor = (
                float(spawn_position[0]) + max(-1.25, min(1.25, tx * 0.35)),
                float(base_feet_y),
                float(spawn_position[2]) + max(2.0, tz - 2.0),
            )
            inner_anchor = (
                float(spawn_position[0]) + max(-2.0, min(2.0, tx * 0.65)),
                float(base_feet_y),
                float(spawn_position[2]) + max(2.5, tz - 1.5),
            )
            profile_specs = [
                {"name": "chamber_entry_center", "position": base_anchor, "height_offset": 0.0, "pose_family": "chamber_entry", "selection_bias": 2600.0},
                {"name": "chamber_entry_inner", "position": inner_anchor, "height_offset": 0.0, "pose_family": "chamber_entry", "selection_bias": 2000.0},
                {
                    "name": "chamber_left_overlook",
                    "position": (float(inner_anchor[0]) - float(lateral[0]) * 0.5, float(inner_anchor[1]), float(inner_anchor[2]) - float(lateral[2]) * 0.5),
                    "height_offset": 0.0,
                    "pose_family": "oblique_left",
                    "selection_bias": 900.0,
                },
                {
                    "name": "chamber_right_overlook",
                    "position": (float(inner_anchor[0]) + float(lateral[0]) * 0.5, float(inner_anchor[1]), float(inner_anchor[2]) + float(lateral[2]) * 0.5),
                    "height_offset": 0.0,
                    "pose_family": "oblique_right",
                    "selection_bias": 900.0,
                },
            ]
        elif blueprint_id == "straight_tunnel":
            path_obstacle_variant = (
                procedural_layout.get("path_obstacle_variant")
                if isinstance(procedural_layout.get("path_obstacle_variant"), dict)
                else {}
            )
            obstacle_positions = {
                (int(item[0]), int(item[1]))
                for item in (path_obstacle_variant.get("positions") or [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            }
            if obstacle_positions:
                p_goal_pose_protocol = str(
                    data.get("p_goal_pose_protocol")
                    or procedural_layout.get("p_goal_pose_protocol")
                    or "center_z3"
                ).strip().lower()
                if p_goal_pose_protocol == "center_z3":
                    # P-obstacle banks keep target_local=(0,5) and reserve
                    # local (0,3)/(0,4) as clear cells. This gives a single
                    # centered same-world goal pose with the old auto-goal scale
                    # while avoiding oblique fallback views whose projected
                    # target overlays are less reliable.
                    center_anchor = (
                        float(spawn_position[0]) + tx,
                        float(base_feet_y),
                        float(spawn_position[2]) + max(2.0, tz - 2.0),
                    )
                    profile_specs = [
                        {
                            "name": "p_obstacle_center_z3",
                            "position": center_anchor,
                            "height_offset": 0.0,
                            "pitch_offset_deg": -6.0,
                            "pose_family": "p_obstacle_center",
                            "selection_bias": 4200.0,
                        },
                    ]
                elif p_goal_pose_protocol == "legacy_scale":
                    base_anchor = (
                        float(spawn_position[0]) + max(-0.25, min(0.25, tx)),
                        float(base_feet_y),
                        float(spawn_position[2]) + max(2.0, tz - 2.25),
                    )
                    mid_anchor = (
                        float(base_anchor[0]),
                        float(base_anchor[1]),
                        float(base_anchor[2]) + float(outward[2]) * 0.5 + float(outward[0]) * 0.5,
                    )
                    profile_specs = [
                        {
                            "name": "p_obstacle_legacy_center_close",
                            "position": base_anchor,
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_legacy",
                            "selection_bias": 3200.0,
                        },
                        {
                            "name": "p_obstacle_legacy_center_mid",
                            "position": mid_anchor,
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_legacy",
                            "selection_bias": 2500.0,
                        },
                        {
                            "name": "p_obstacle_legacy_left_shallow",
                            "position": (
                                float(base_anchor[0]) - float(lateral[0]) * 0.45,
                                float(base_anchor[1]),
                                float(base_anchor[2]) - float(lateral[2]) * 0.45,
                            ),
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_legacy_oblique",
                            "selection_bias": 900.0,
                        },
                        {
                            "name": "p_obstacle_legacy_right_shallow",
                            "position": (
                                float(base_anchor[0]) + float(lateral[0]) * 0.45,
                                float(base_anchor[1]),
                                float(base_anchor[2]) + float(lateral[2]) * 0.45,
                            ),
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_legacy_oblique",
                            "selection_bias": 900.0,
                        },
                    ]
                else:
                    # P-obstacle variants often occupy the old close-goal camera
                    # cell around local z=2. The generator keeps local (0, tz-1)
                    # open as the target approach cell, so the front-close
                    # protocol bakes same-world goals from there.
                    front_center = (
                        float(spawn_position[0]) + tx,
                        float(base_feet_y),
                        float(spawn_position[2]) + max(2.0, tz - 0.75),
                    )
                    front_close = (
                        float(spawn_position[0]) + tx,
                        float(base_feet_y),
                        float(spawn_position[2]) + max(2.0, tz - 0.55),
                    )
                    front_safe = (
                        float(spawn_position[0]) + tx,
                        float(base_feet_y),
                        float(spawn_position[2]) + max(2.0, tz - 0.95),
                    )
                    profile_specs = [
                        {
                            "name": "p_obstacle_front_center",
                            "position": front_center,
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_front",
                            "selection_bias": 3200.0,
                        },
                        {
                            "name": "p_obstacle_front_close",
                            "position": front_close,
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_front",
                            "selection_bias": 3000.0,
                        },
                        {
                            "name": "p_obstacle_front_safe",
                            "position": front_safe,
                            "height_offset": 0.0,
                            "pose_family": "p_obstacle_front",
                            "selection_bias": 2600.0,
                        },
                    ]
            else:
                base_anchor = (
                    float(spawn_position[0]) + max(-0.25, min(0.25, tx)),
                    float(base_feet_y),
                    float(spawn_position[2]) + max(2.0, tz - 2.25),
                )
                mid_anchor = (
                    float(base_anchor[0]),
                    float(base_anchor[1]),
                    float(base_anchor[2]) + float(outward[2]) * 0.5 + float(outward[0]) * 0.5,
                )
                profile_specs = [
                    {"name": "corridor_center_close", "position": base_anchor, "height_offset": 0.0, "pose_family": "corridor_center", "selection_bias": 2200.0},
                    {"name": "corridor_center_mid", "position": mid_anchor, "height_offset": 0.0, "pose_family": "corridor_center", "selection_bias": 1500.0},
                    {
                        "name": "corridor_left_shallow",
                        "position": (float(base_anchor[0]) - float(lateral[0]) * 0.45, float(base_anchor[1]), float(base_anchor[2]) - float(lateral[2]) * 0.45),
                        "height_offset": 0.0,
                        "pose_family": "oblique_left",
                        "selection_bias": 700.0,
                    },
                    {
                        "name": "corridor_right_shallow",
                        "position": (float(base_anchor[0]) + float(lateral[0]) * 0.45, float(base_anchor[1]), float(base_anchor[2]) + float(lateral[2]) * 0.45),
                        "height_offset": 0.0,
                        "pose_family": "oblique_right",
                        "selection_bias": 700.0,
                    },
                ]
        elif blueprint_id in {"turn_left", "turn_right"}:
            corner_anchor = (
                float(spawn_position[0]) + tx + float(outward[0]) * 1.6,
                float(base_feet_y),
                float(spawn_position[2]) + tz + float(outward[2]) * 1.6,
            )
            entry_anchor = (
                float(spawn_position[0]) + tx * 0.5,
                float(base_feet_y),
                float(spawn_position[2]) + max(2.5, tz - 1.5),
            )
            profile_specs = [
                {"name": "corner_peek_center", "position": corner_anchor, "height_offset": 0.0, "pose_family": "corner_center", "selection_bias": 2500.0},
                {"name": "corner_entry_view", "position": entry_anchor, "height_offset": 0.0, "pose_family": "corner_entry", "selection_bias": 1700.0},
                {
                    "name": "corner_outer_oblique",
                    "position": (float(corner_anchor[0]) - float(lateral[0]) * 0.55, float(corner_anchor[1]), float(corner_anchor[2]) - float(lateral[2]) * 0.55),
                    "height_offset": 0.0,
                    "pose_family": "oblique_left",
                    "selection_bias": 800.0,
                },
                {
                    "name": "corner_inner_oblique",
                    "position": (float(corner_anchor[0]) + float(lateral[0]) * 0.35, float(corner_anchor[1]), float(corner_anchor[2]) + float(lateral[2]) * 0.35),
                    "height_offset": 0.0,
                    "pose_family": "oblique_right",
                    "selection_bias": 700.0,
                },
            ]
        elif blueprint_id in {"side_alcove_left", "side_alcove_right"}:
            # For side alcoves, the canonical behavioral view is the +z face seen
            # from the open center lane directly in front of the target. The O2
            # leaves are intentionally placed on the lateral flanks of that lane,
            # so shifting the camera sideways actually moves it into the occluder.
            frontal_lateral_offset = 0.0
            frontal_center_distances = (1.35, 1.6, 1.85)
            fallback_lateral_offset = 1.25
            fallback_center_distance = 1.55

            def _alcove_position(center_distance: float, lateral_offset: float) -> tuple[float, float, float]:
                return (
                    float(spawn_position[0]) + tx + float(outward[0]) * float(center_distance) + float(lateral[0]) * float(lateral_offset),
                    float(base_feet_y),
                    float(spawn_position[2]) + tz + float(outward[2]) * float(center_distance) + float(lateral[2]) * float(lateral_offset),
                )

            profile_specs = [
                {
                    "name": "side_alcove_plus_z_frontal_near",
                    "position": _alcove_position(frontal_center_distances[0], frontal_lateral_offset),
                    "height_offset": 0.0,
                    "pose_family": "side_alcove_frontal",
                    "selection_bias": 2600.0,
                },
                {
                    "name": "side_alcove_plus_z_frontal_mid",
                    "position": _alcove_position(frontal_center_distances[1], frontal_lateral_offset),
                    "height_offset": 0.0,
                    "pose_family": "side_alcove_frontal",
                    "selection_bias": 2350.0,
                },
                {
                    "name": "side_alcove_plus_z_frontal_far",
                    "position": _alcove_position(frontal_center_distances[2], frontal_lateral_offset),
                    "height_offset": 0.0,
                    "pose_family": "side_alcove_frontal",
                    "selection_bias": 2100.0,
                },
                {
                    "name": "side_alcove_plus_z_fallback_oblique",
                    "position": _alcove_position(fallback_center_distance, fallback_lateral_offset),
                    "height_offset": 0.0,
                    "pose_family": "side_alcove_oblique",
                    "selection_bias": 650.0,
                },
            ]
        else:
            anchor_local = (tx, max(2.0, tz - 2.0))
            base_anchor = (
                float(spawn_position[0]) + float(anchor_local[0]),
                float(base_feet_y),
                float(spawn_position[2]) + float(anchor_local[1]),
            )
            profile_specs = [
                {"name": "front_center_near", "position": base_anchor, "height_offset": 0.0, "pose_family": "frontal_center", "selection_bias": 1800.0},
                {"name": "front_center_far", "position": (float(base_anchor[0]) + float(outward[0]), float(base_anchor[1]), float(base_anchor[2]) + float(outward[2])), "height_offset": 0.0, "pose_family": "frontal_center", "selection_bias": 1200.0},
                {
                    "name": "front_left",
                    "position": (float(base_anchor[0]) - float(lateral[0]) * 0.6, float(base_anchor[1]), float(base_anchor[2]) - float(lateral[2]) * 0.6),
                    "height_offset": 0.0,
                    "pose_family": "oblique_left",
                    "selection_bias": 600.0,
                },
                {
                    "name": "front_right",
                    "position": (float(base_anchor[0]) + float(lateral[0]) * 0.6, float(base_anchor[1]), float(base_anchor[2]) + float(lateral[2]) * 0.6),
                    "height_offset": 0.0,
                    "pose_family": "oblique_right",
                    "selection_bias": 600.0,
                },
            ]
    else:
        profile_specs = [
            {
                "name": "front_center_near",
                "position": (float(target[0]) + float(outward[0]) * 2.0, float(base_feet_y), float(target[2]) + float(outward[2]) * 2.0),
                "height_offset": 0.0,
                "pose_family": "frontal_center",
                "selection_bias": 1800.0,
            },
            {
                "name": "front_center_far",
                "position": (float(target[0]) + float(outward[0]) * 3.0, float(base_feet_y), float(target[2]) + float(outward[2]) * 3.0),
                "height_offset": 0.0,
                "pose_family": "frontal_center",
                "selection_bias": 1200.0,
            },
            {
                "name": "front_left",
                "position": (
                    float(target[0]) + float(outward[0]) * 2.5 - float(lateral[0]) * 0.6,
                    float(base_feet_y),
                    float(target[2]) + float(outward[2]) * 2.5 - float(lateral[2]) * 0.6,
                ),
                "height_offset": 0.0,
                "pose_family": "oblique_left",
                "selection_bias": 600.0,
            },
            {
                "name": "front_right",
                "position": (
                    float(target[0]) + float(outward[0]) * 2.5 + float(lateral[0]) * 0.6,
                    float(base_feet_y),
                    float(target[2]) + float(outward[2]) * 2.5 + float(lateral[2]) * 0.6,
                ),
                "height_offset": 0.0,
                "pose_family": "oblique_right",
                "selection_bias": 600.0,
            },
        ]

    for spec in profile_specs:
        name = str(spec["name"])
        camera_position = tuple(float(v) for v in spec["position"])
        height_offset = float(spec.get("height_offset", 0.0))
        pitch_offset_deg = float(spec.get("pitch_offset_deg", 0.0))
        pose_family = str(spec.get("pose_family") or "")
        selection_bias = float(spec.get("selection_bias", 0.0))
        yaw, pitch = _look_angles_for_target(camera_position, look_target)
        pitch = max(-89.0, min(89.0, float(pitch) + float(pitch_offset_deg)))
        support_block = {
            "enabled": False,
            "x": int(math.floor(float(camera_position[0]))),
            "y": int(math.floor(float(camera_position[1]))) - 1,
            "z": int(math.floor(float(camera_position[2]))),
            "feet_y": round(float(camera_position[1]), 4),
            "block_type": "minecraft:barrier",
        } if disable_hint_support_block else {
            "x": int(math.floor(float(camera_position[0]))),
            "y": int(math.floor(float(camera_position[1]))) - 1,
            "z": int(math.floor(float(camera_position[2]))),
            "feet_y": round(float(camera_position[1]), 4),
            "block_type": "minecraft:barrier",
        }
        hints.append(
            {
                "name": str(name),
                "position": [
                    round(float(camera_position[0]), 4),
                    round(float(camera_position[1]), 4),
                    round(float(camera_position[2]), 4),
                ],
                "yaw": round(float(yaw), 4),
                "pitch": round(float(pitch), 4),
                "target_world_center": [round(float(v), 4) for v in target],
                "look_target_world_center": [round(float(v), 4) for v in look_target],
                "target_type": target_type,
                "preferred_face": str(preferred_face),
                "pose_family": pose_family,
                "selection_bias": round(float(selection_bias), 4),
                "distance": round(
                    math.sqrt((float(camera_position[0]) - float(target[0])) ** 2 + (float(camera_position[2]) - float(target[2])) ** 2),
                    4,
                ),
                "lateral_offset": round(
                    (float(camera_position[0]) - float(target[0])) * float(lateral[0])
                    + (float(camera_position[2]) - float(target[2])) * float(lateral[2]),
                    4,
                ),
                "height_offset": round(float(height_offset), 4),
                "pitch_offset_deg": round(float(pitch_offset_deg), 4),
                "support_block": support_block,
            }
        )
    return hints


def should_use_procedural_mine_backend(task_key: str, backend: str) -> bool:
    if task_key not in PROCEDURAL_MINE_TASKS:
        return False
    backend = str(backend or "auto").strip().lower()
    if backend == "template":
        return False
    return True


def _relative_exec(command: str) -> str:
    return f"/execute as @p at @s run {command}"


def _relative_fill(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block: str) -> str:
    return _relative_exec(
        f"fill ~{x1} ~{y1} ~{z1} ~{x2} ~{y2} ~{z2} {block}"
    )


def _relative_setblock(x: int, y: int, z: int, block: str) -> str:
    return _relative_exec(f"setblock ~{x} ~{y} ~{z} {block}")


def _parse_bool_suggestion(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _absolute_tp_command(x: float, y: float, z: float, yaw: float, pitch: float) -> str:
    return (
        f"/tp @a "
        f"{float(x):.3f} {float(y):.3f} {float(z):.3f} "
        f"{float(yaw):.3f} {float(pitch):.3f}"
    )


_RELATIVE_EXEC_PREFIX = "/execute as @p at @s run "


def _anchor_relative_command(command: str, *, anchor_x: float, anchor_y: float, anchor_z: float) -> str:
    stripped = str(command or "").strip()
    if not stripped.startswith(_RELATIVE_EXEC_PREFIX):
        return str(command)
    suffix = stripped[len(_RELATIVE_EXEC_PREFIX):]
    return (
        f"/execute positioned "
        f"{float(anchor_x):.3f} {float(anchor_y):.3f} {float(anchor_z):.3f} "
        f"run {suffix}"
    )


def anchor_relative_commands(commands: List[str], *, anchor_x: float, anchor_y: float, anchor_z: float) -> List[str]:
    return [
        _anchor_relative_command(
            command,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            anchor_z=anchor_z,
        )
        for command in list(commands or [])
    ]


def _procedural_anchor_position(rng: random.Random) -> tuple[float, float, float]:
    grid_x = int(rng.choice(PROCEDURAL_MINE_SAFE_ANCHOR_X_CHOICES))
    grid_z = int(rng.choice(PROCEDURAL_MINE_SAFE_ANCHOR_Z_CHOICES))
    return float(grid_x) + 0.5, float(PROCEDURAL_MINE_SAFE_ANCHOR_Y), float(grid_z) + 0.5


def _base_spawn_position(
    source_data: Dict,
    rng: random.Random,
    *,
    anchor_mode: str = "source_or_fallback",
    disable_jitter: bool = False,
    fixed_position: Optional[List[float]] = None,
) -> tuple[float, float, float, int, Dict[str, Any]]:
    spawn_positions = list(source_data.get("spawn_positions") or [])
    first = spawn_positions[0] if spawn_positions else {}
    position = list((first or {}).get("position") or [])
    seed = int((first or {}).get("seed", 19961103) or 19961103)
    if isinstance(fixed_position, (list, tuple)) and len(fixed_position) >= 3:
        base_x = float(fixed_position[0])
        base_y = float(fixed_position[1])
        base_z = float(fixed_position[2])
        jitter_choices = [0]
        anchor_source = "fixed_spawn_position"
    elif anchor_mode == "fixed_source" and len(position) >= 3:
        base_x = float(position[0])
        base_y = float(position[1])
        base_z = float(position[2])
        jitter_choices = [0]
        anchor_source = "fixed_source_spawn"
    elif anchor_mode != "procedural_only" and len(position) >= 3:
        base_x = float(position[0])
        base_y = float(position[1])
        base_z = float(position[2])
        jitter_choices = [-6, -3, 0, 3, 6]
        anchor_source = "source_spawn"
    else:
        base_x, base_y, base_z = _procedural_anchor_position(rng)
        jitter_choices = [-16, -8, 0, 8, 16]
        anchor_source = "procedural_safe_anchor"
    if disable_jitter:
        jitter_choices = [0]
    jitter_x = float(rng.choice(jitter_choices))
    jitter_z = float(rng.choice(jitter_choices))
    spawn_x = base_x + jitter_x
    spawn_z = base_z + jitter_z
    anchor_metadata = {
        "anchor_mode": str(anchor_mode),
        "anchor_source": str(anchor_source),
        "source_spawn_available": bool(len(position) >= 3),
        "base_position": [round(float(base_x), 4), round(float(base_y), 4), round(float(base_z), 4)],
        "jitter": {"dx": round(float(jitter_x), 4), "dz": round(float(jitter_z), 4)},
        "disable_jitter": bool(disable_jitter),
        "fixed_position": [round(float(v), 4) for v in fixed_position[:3]] if isinstance(fixed_position, (list, tuple)) and len(fixed_position) >= 3 else None,
        "spawn_seed": int(seed),
    }
    return spawn_x, base_y, spawn_z, seed, anchor_metadata


def _mine_target_type(task_key: str) -> str:
    return "coal_ore" if task_key == "mine_coal" else "emerald_ore"


def _mine_required_tool_type(task_key: str) -> str:
    if task_key == "mine_emerald":
        return "iron_pickaxe"
    if task_key == "mine_coal":
        return "diamond_pickaxe"
    return "wooden_pickaxe"


def _mine_fallback_inventory(task_key: str) -> List[Dict]:
    return [{"slot": 0, "type": _mine_required_tool_type(task_key), "quantity": 1}]


def _ensure_required_mine_tool(task_key: str, inventory: List[Dict]) -> List[Dict]:
    required_tool_type = _mine_required_tool_type(task_key)
    normalized: List[Dict] = []
    slot_zero_idx: Optional[int] = None
    for item in inventory:
        if not isinstance(item, dict):
            continue
        copied = copy.deepcopy(item)
        try:
            slot_value = int(copied.get("slot", 0))
        except Exception:
            slot_value = 0
        if slot_value == 0 and slot_zero_idx is None:
            slot_zero_idx = len(normalized)
        normalized.append(copied)
    if slot_zero_idx is None:
        normalized.insert(0, {"slot": 0, "type": required_tool_type, "quantity": 1})
        return normalized
    normalized[slot_zero_idx]["slot"] = 0
    normalized[slot_zero_idx]["type"] = required_tool_type
    normalized[slot_zero_idx]["quantity"] = 1
    return normalized


def _mine_inherited_inventory(source_data: Dict, task_key: str) -> List[Dict]:
    raw_inventory = source_data.get("init_inventory")
    if isinstance(raw_inventory, list) and raw_inventory:
        inherited: List[Dict] = []
        for item in raw_inventory:
            if isinstance(item, dict):
                inherited.append(copy.deepcopy(item))
        if inherited:
            return _ensure_required_mine_tool(task_key, inherited)
    return list(_mine_fallback_inventory(task_key))


def _mine_effective_time_limit(source_data: Dict) -> int:
    base_time_limit = int(source_data.get("time_limit") or 150)
    return max(1, base_time_limit) + int(PROCEDURAL_MINE_ROLLOUT_WARMUP_STEPS)


def _mine_distance_options(distance_level: int) -> List[int]:
    options = {
        0: [4],
        1: [5],
        2: [6],
        3: [7],
    }
    return list(options.get(int(distance_level), [10]))


def _mine_turn_targets(distance_level: int, *, right_turn: bool) -> List[tuple[int, int]]:
    sign = 1 if right_turn else -1
    options = {
        0: [(3 * sign, 4), (4 * sign, 4)],
        1: [(4 * sign, 5), (5 * sign, 5)],
        2: [(5 * sign, 6), (6 * sign, 6)],
        3: [(6 * sign, 7), (7 * sign, 7)],
    }
    return list(options.get(int(distance_level), [(4 * sign, 5)]))


def _mine_alcove_targets(distance_level: int, *, right_side: bool) -> List[tuple[int, int]]:
    sign = 1 if right_side else -1
    options = {
        0: [(3 * sign, 4), (4 * sign, 4)],
        1: [(4 * sign, 5), (5 * sign, 5)],
        2: [(5 * sign, 6), (6 * sign, 6)],
        3: [(6 * sign, 7), (7 * sign, 7)],
    }
    return list(options.get(int(distance_level), [(4 * sign, 6)]))


def _choose_mine_blueprint(path_level: int, rng: random.Random) -> str:
    # Layout family is no longer encoded by P. P should control obstacle/path
    # obstruction within a layout, while geometry family is treated as a
    # separate context axis for audit and benchmark stratification.
    del path_level
    choices = [
        "straight_tunnel",
        "offset_chamber",
        "turn_left",
        "turn_right",
        "side_alcove_left",
        "side_alcove_right",
    ]
    return str(rng.choice(choices))


def _normalize_mine_blueprint_override(value: Any) -> str | None:
    candidate = str(value or "").strip()
    if candidate in {
        "straight_tunnel",
        "offset_chamber",
        "turn_left",
        "turn_right",
        "side_alcove_left",
        "side_alcove_right",
    }:
        return candidate
    return None


def _mine_target_local(
    blueprint_id: str,
    factor_levels: Dict[str, int],
    rng: random.Random,
    target_sign_override: str | None = None,
) -> tuple[int, int]:
    distance_level = int(factor_levels.get("R", 0))
    if blueprint_id == "straight_tunnel":
        return 0, int(rng.choice(_mine_distance_options(distance_level)))
    if blueprint_id == "offset_chamber":
        z = int(rng.choice(_mine_distance_options(distance_level)))
        x_choices = {
            0: [-2, 2],
            1: [-2, 2, -3, 3],
            2: [-3, 3, -4, 4],
            3: [-4, 4, -5, 5],
        }.get(distance_level, [-3, 3])
        sign_token = str(target_sign_override or "").strip().lower()
        if sign_token == "neg":
            x_choices = [value for value in x_choices if int(value) < 0] or x_choices
        elif sign_token == "pos":
            x_choices = [value for value in x_choices if int(value) > 0] or x_choices
        return int(rng.choice(x_choices)), int(z)
    if blueprint_id == "turn_left":
        return tuple(int(v) for v in rng.choice(_mine_turn_targets(distance_level, right_turn=False)))
    if blueprint_id == "turn_right":
        return tuple(int(v) for v in rng.choice(_mine_turn_targets(distance_level, right_turn=True)))
    if blueprint_id == "side_alcove_left":
        return tuple(int(v) for v in rng.choice(_mine_alcove_targets(distance_level, right_side=False)))
    if blueprint_id == "side_alcove_right":
        return tuple(int(v) for v in rng.choice(_mine_alcove_targets(distance_level, right_side=True)))
    return 0, int(rng.choice(_mine_distance_options(distance_level)))


def _mine_target_local_override(suggestions: Dict | None) -> tuple[int, int] | None:
    suggestions = suggestions or {}
    raw_local = suggestions.get("mine_target_local")
    if isinstance(raw_local, (list, tuple)) and len(raw_local) >= 2:
        try:
            return int(raw_local[0]), int(raw_local[1])
        except Exception:
            return None
    raw_z = suggestions.get("mine_target_z")
    if raw_z is not None:
        try:
            return 0, int(raw_z)
        except Exception:
            return None
    return None


def _mine_preferred_face(blueprint_id: str) -> str:
    if blueprint_id == "turn_left":
        return "+x"
    if blueprint_id == "turn_right":
        return "-x"
    if blueprint_id == "side_alcove_left":
        return "+x"
    if blueprint_id == "side_alcove_right":
        return "-x"
    return "-z"


def _mine_goal_camera_preferred_face(
    blueprint_id: str,
    target_local: tuple[int, int],
    layout_preferred_face: str,
) -> str:
    tx = int(target_local[0])
    if blueprint_id in {"side_alcove_left", "side_alcove_right"}:
        return "+z"
    if blueprint_id == "offset_chamber":
        if tx < 0:
            return "+x"
        if tx > 0:
            return "-x"
    return str(layout_preferred_face)


def _mine_action_preferred_face(
    blueprint_id: str,
    target_local: tuple[int, int],
    layout_preferred_face: str,
    goal_camera_preferred_face: str,
) -> str:
    if blueprint_id in {"side_alcove_left", "side_alcove_right", "offset_chamber"}:
        if str(goal_camera_preferred_face or "").strip():
            return str(goal_camera_preferred_face)
    return str(layout_preferred_face)


def _mine_visibility_preferred_face(
    blueprint_id: str,
    target_local: tuple[int, int],
    layout_preferred_face: str,
    goal_camera_preferred_face: str,
) -> str:
    # Visibility/O-factor should track the face we want the goal camera and
    # rollout to reason about, not the structural carve face used for layout
    # metrics. Using the structural face for side alcoves was rewriting the
    # corridor wall and collapsing the layout into a turn-like shortcut.
    if str(goal_camera_preferred_face or "").strip():
        return str(goal_camera_preferred_face)
    return str(layout_preferred_face)


def _mine_lateral_offsets(face_label: str) -> List[tuple[int, int]]:
    if face_label in {"+x", "-x"}:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _face_label_from_offset(dx: int, dz: int) -> str:
    if int(dx) > 0:
        return "+x"
    if int(dx) < 0:
        return "-x"
    if int(dz) > 0:
        return "+z"
    if int(dz) < 0:
        return "-z"
    return ""


def _mine_core_open_faces(face_label: str) -> List[str]:
    label = str(face_label or "").strip()
    return [label] if label else []


def _mine_target_neighborhood_commands(
    blueprint_id: str,
    target_local: tuple[int, int],
    preferred_face: str,
    action_level: int,
    rng: random.Random,
) -> tuple[List[str], List[str]]:
    tx, tz = int(target_local[0]), int(target_local[1])
    commands: List[str] = []
    opened_faces: List[str] = []
    lateral_offsets = list(_mine_lateral_offsets(preferred_face))
    rng.shuffle(lateral_offsets)
    keep_top_open = int(action_level) <= 2
    lateral_open_count = {
        0: 2,
        1: 1,
        2: 0,
        3: 0,
    }.get(int(action_level), 0)
    open_laterals = lateral_offsets[:lateral_open_count]
    closed_laterals = lateral_offsets[lateral_open_count:]

    # Side-alcove layouts should present the corridor-side +z/-z face as the
    # canonical approach face, not a fake backside opening into the wall behind
    # the ore. Leaving the opposite z-face open was creating an accidental
    # branch-like shortcut that belongs to path confusion, not action margin.
    if blueprint_id in {"side_alcove_left", "side_alcove_right"} and preferred_face in {"+z", "-z"}:
        preferred_dz = 1 if preferred_face == "+z" else -1
        opposite_dz = -preferred_dz
        commands.append(_relative_setblock(tx, 0, tz + preferred_dz, "minecraft:air"))
        commands.append(_relative_setblock(tx, 0, tz + opposite_dz, "minecraft:stone"))

    for dx, dz in open_laterals:
        commands.append(_relative_setblock(tx + dx, 0, tz + dz, "minecraft:air"))
        face_label = _face_label_from_offset(dx, dz)
        if face_label:
            opened_faces.append(face_label)
    # Do not rely on the surrounding carve to implicitly define action margin.
    # Some layouts, notably offset_chamber, already leave the target laterals open
    # as part of the chamber floorplan, which made A2/A3 effectively no-op.
    # Explicitly sealing the non-open laterals keeps the realized geometry aligned
    # with the recorded action-margin metadata.
    for dx, dz in closed_laterals:
        commands.append(_relative_setblock(tx + dx, 0, tz + dz, "minecraft:stone"))
    if keep_top_open:
        commands.append(_relative_setblock(tx, 1, tz, "minecraft:air"))
        opened_faces.append("+y")
    else:
        commands.append(_relative_setblock(tx, 1, tz, "minecraft:stone"))
    unique_opened_faces = sorted({label for label in opened_faces if label})
    return commands, unique_opened_faces


def _mine_target_back_wall_commands(
    blueprint_id: str,
    target_local: tuple[int, int],
    *,
    height: int = 3,
    half_width: int = 5,
) -> List[str]:
    tx, tz = int(target_local[0]), int(target_local[1])
    height = max(1, min(6, int(height)))
    half_width = max(1, min(12, int(half_width)))
    if blueprint_id != "straight_tunnel":
        return []
    # In wide arena mode the usual straight-tunnel stone shell is intentionally
    # disabled to preserve side-route space. Restore a full-width target backing
    # wall so the agent cannot solve the task by walking around the ore from
    # behind, while keeping the front side open for the intended mine interaction.
    return [_relative_fill(-half_width, 0, tz + 1, half_width, height, tz + 1, "minecraft:stone")]


def _mine_path_obstacle_positions(
    blueprint_id: str,
    target_local: tuple[int, int],
    rng: random.Random,
) -> List[tuple[int, int]]:
    tx, tz = int(target_local[0]), int(target_local[1])
    points: List[tuple[int, int]] = []

    if blueprint_id == "straight_tunnel":
        side = int(rng.choice([-1, 1]))
        z1 = max(3, min(tz - 3, max(4, tz // 2)))
        z2 = max(z1 + 1, min(tz - 2, max(5, tz - 2)))
        z3 = max(3, min(z1 + 2, tz - 1))
        points = [(side, z1), (-side, z2), (side, z3)]
    elif blueprint_id == "offset_chamber":
        chamber_start = max(4, tz - 2)
        corridor_side = -1 if tx > 0 else (1 if tx < 0 else int(rng.choice([-1, 1])))
        z1 = max(3, chamber_start - 1)
        z2 = chamber_start
        z3 = max(chamber_start + 1, min(tz - 1, chamber_start + 2))
        points = [(corridor_side, z1), (0, z2), (corridor_side, z3)]
    elif blueprint_id == "turn_left":
        turn_z = max(4, tz)
        points = [(0, max(3, turn_z - 2)), (-1, max(4, turn_z - 1)), (-2, turn_z)]
    elif blueprint_id == "turn_right":
        turn_z = max(4, tz)
        points = [(0, max(3, turn_z - 2)), (1, max(4, turn_z - 1)), (2, turn_z)]
    elif blueprint_id == "side_alcove_left":
        alcove_z = max(5, tz)
        points = [(0, max(3, alcove_z - 2)), (-1, max(4, alcove_z - 1)), (-1, alcove_z)]
    elif blueprint_id == "side_alcove_right":
        alcove_z = max(5, tz)
        points = [(0, max(3, alcove_z - 2)), (1, max(4, alcove_z - 1)), (1, alcove_z)]

    deduped: List[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, z in points:
        point = (int(x), int(z))
        if point == (tx, tz):
            continue
        if point in seen:
            continue
        seen.add(point)
        deduped.append(point)
    return deduped


def _normalize_minecraft_block_id(value: Any, default: str = "minecraft:cobblestone") -> str:
    block_id = str(value or "").strip()
    if not block_id:
        return str(default)
    if ":" not in block_id:
        block_id = f"minecraft:{block_id}"
    return block_id


def _mine_path_obstacle_variant_override(suggestions: Dict | None) -> Dict[str, Any]:
    suggestions = suggestions or {}
    raw_positions = suggestions.get("mine_path_obstacle_positions") or []
    positions: List[List[int]] = []
    if isinstance(raw_positions, list):
        for item in raw_positions:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            try:
                x = int(item[0])
                z = int(item[1])
            except Exception:
                continue
            positions.append([x, z])
    if not positions:
        return {}
    try:
        height = int(suggestions.get("mine_path_obstacle_height", 1) or 1)
    except Exception:
        height = 1
    height = max(1, min(3, height))
    return {
        "variant_id": str(suggestions.get("mine_path_obstacle_variant_id") or "exact_path_obstacle"),
        "positions": positions,
        "height": int(height),
        "material": _normalize_minecraft_block_id(
            suggestions.get("mine_path_obstacle_material"),
            default="minecraft:cobblestone",
        ),
    }


def _mine_fixed_spawn_position_override(suggestions: Dict | None) -> Optional[List[float]]:
    suggestions = suggestions or {}
    raw = suggestions.get("mine_fixed_spawn_position")
    if not (isinstance(raw, (list, tuple)) and len(raw) >= 3):
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    except Exception:
        return None


def _mine_arena_settings_override(suggestions: Dict | None) -> Dict[str, Any]:
    suggestions = suggestions or {}
    mode = str(suggestions.get("mine_arena_mode") or "").strip().lower()
    if mode not in {"wide_symmetric", "o2_target_funnel"}:
        return {}
    settings: Dict[str, Any] = {"mode": mode}
    for key, default in (
        ("half_width", 5),
        ("back_z", -3),
        ("front_buffer", 3),
        ("wall_height", 3),
    ):
        raw_value = suggestions.get(f"mine_arena_{key}")
        try:
            settings[key] = int(raw_value if raw_value is not None else default)
        except Exception:
            settings[key] = int(default)
    settings["half_width"] = max(2, min(12, int(settings["half_width"])))
    settings["back_z"] = max(-12, min(-1, int(settings["back_z"])))
    settings["front_buffer"] = max(1, min(12, int(settings["front_buffer"])))
    settings["wall_height"] = max(2, min(8, int(settings["wall_height"])))
    settings["roof"] = _parse_bool_suggestion(suggestions.get("mine_arena_roof"), False)
    try:
        settings["funnel_start_z"] = int(suggestions.get("mine_arena_funnel_start_z", 3))
    except Exception:
        settings["funnel_start_z"] = 3
    settings["funnel_start_z"] = max(1, min(12, int(settings["funnel_start_z"])))
    return settings


def _mine_occluder_commands(
    target_local: tuple[int, int],
    preferred_face: str,
    visibility_level: int,
    rng: random.Random,
    blueprint_id: str = "",
    variant_override: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], Dict[str, Any]]:
    if int(visibility_level) <= 0:
        return [], {
            "variant_name": "none",
            "front_depth": 0,
            "side_span": 0,
            "side_pattern": "none",
            "center_material": "none",
            "left_material": "none",
            "right_material": "none",
        }
    tx, tz = int(target_local[0]), int(target_local[1])
    face_axes = _face_axes_from_label(preferred_face)
    if face_axes is None:
        return [], {
            "variant_name": "invalid_face",
            "front_depth": 0,
            "side_span": 0,
            "side_pattern": "none",
            "center_material": "none",
            "left_material": "none",
            "right_material": "none",
        }
    outward, lateral = face_axes
    fx = tx + int(outward[0])
    fz = tz + int(outward[2])
    lx = int(lateral[0])
    lz = int(lateral[2])
    material = "minecraft:glass" if int(visibility_level) <= 1 else "minecraft:oak_leaves[persistent=true]"
    commands: List[str] = []
    if int(visibility_level) == 1:
        # O1 should partially occlude the target from the canonical approach face,
        # not sit on a lateral neighboring cell where it fails to interfere at all.
        commands.append(_relative_fill(fx, 0, fz, fx, 1, fz, material))
        return commands, {
            "variant_name": "o1_center_column",
            "front_depth": 1,
            "side_span": 0,
            "side_pattern": "center_only",
            "center_material": material,
            "left_material": "none",
            "right_material": "none",
        }
    elif int(visibility_level) == 2:
        # O2 should be visibly harder than O1, but it must not collapse into a
        # fully hidden target. Keep a narrow center sightline for every layout;
        # otherwise short straight-tunnel cases at R0/R1 can become unbakeable
        # because the frontal wall fully covers the canonical approach face.
        #
        # Side-alcove layouts are structurally tighter than straight tunnels:
        # the target has only a single-cell stand-off along its carved layout
        # face. Placing a center O2 block directly on that structural face can
        # consume the only valid stand position and collapse both baking and
        # rollout visibility. Keep the center lane open and only place the
        # lateral side columns there.
        override = dict(variant_override or {})
        if blueprint_id == "straight_tunnel":
            front_depth = max(1, min(2, int(override.get("front_depth", rng.choice([1, 2])))))
            side_span = max(1, min(2, int(override.get("side_span", rng.choice([1, 2])))))
            exact_top_row = override.get("top_row")
            exact_bottom_row = override.get("bottom_row")
            if exact_top_row is not None or exact_bottom_row is not None:
                def _parse_row_tokens(raw_value, *, fallback: List[str]) -> List[str]:
                    if isinstance(raw_value, str):
                        tokens = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
                    elif isinstance(raw_value, (list, tuple)):
                        tokens = [str(item).strip().lower() for item in raw_value]
                    else:
                        tokens = []
                    if len(tokens) != 3:
                        tokens = list(fallback)
                    normalized_tokens: List[str] = []
                    for token in tokens:
                        if token in {"glass", "g"}:
                            normalized_tokens.append("glass")
                        elif token in {"leaves", "leaf", "l"}:
                            normalized_tokens.append("leaves")
                        else:
                            normalized_tokens.append("glass")
                    return normalized_tokens

                top_tokens = _parse_row_tokens(exact_top_row, fallback=["glass", "glass", "glass"])
                bottom_tokens = _parse_row_tokens(exact_bottom_row, fallback=["glass", "glass", "glass"])
                leaf_material = str(
                    override.get("leaf_material", "minecraft:oak_leaves[persistent=true]")
                ).strip() or "minecraft:oak_leaves[persistent=true]"
                glass_material = str(override.get("glass_material", "minecraft:glass")).strip() or "minecraft:glass"
                variant_id = str(override.get("variant_id", "")).strip()
                fx = tx + int(outward[0]) * front_depth
                fz = tz + int(outward[2]) * front_depth
                positions = [
                    ((fx - lx * side_span, 1, fz - lz * side_span), top_tokens[0]),
                    ((fx, 1, fz), top_tokens[1]),
                    ((fx + lx * side_span, 1, fz + lz * side_span), top_tokens[2]),
                    ((fx - lx * side_span, 0, fz - lz * side_span), bottom_tokens[0]),
                    ((fx, 0, fz), bottom_tokens[1]),
                    ((fx + lx * side_span, 0, fz + lz * side_span), bottom_tokens[2]),
                ]
                for (cx, cy, cz), token in positions:
                    block = glass_material if token == "glass" else leaf_material
                    commands.append(_relative_setblock(cx, cy, cz, block))
                left_material = {"top": top_tokens[0], "bottom": bottom_tokens[0]}
                center_material = {"top": top_tokens[1], "bottom": bottom_tokens[1]}
                right_material = {"top": top_tokens[2], "bottom": bottom_tokens[2]}
                pattern_slug = "".join("g" if token == "glass" else "l" for token in top_tokens + bottom_tokens)
                resolved_variant_name = variant_id or f"o2_straight_grid_d{front_depth}_s{side_span}_{pattern_slug}"
                return commands, {
                    "variant_name": str(resolved_variant_name),
                    "front_depth": int(front_depth),
                    "side_span": int(side_span),
                    "side_pattern": "grid_exact",
                    "top_row": list(top_tokens),
                    "bottom_row": list(bottom_tokens),
                    "center_material": center_material,
                    "left_material": left_material,
                    "right_material": right_material,
                    "leaf_material": str(leaf_material),
                    "glass_material": str(glass_material),
                }
            side_pattern = str(
                override.get(
                    "side_pattern",
                    rng.choice(["leaves_leaves", "glass_leaves", "leaves_glass", "glass_glass"]),
                )
            ).strip().lower()
            if side_pattern not in {"leaves_leaves", "glass_leaves", "leaves_glass", "glass_glass"}:
                side_pattern = "leaves_leaves"
            fx = tx + int(outward[0]) * front_depth
            fz = tz + int(outward[2]) * front_depth
            left_material, right_material = {
                "leaves_leaves": ("minecraft:oak_leaves[persistent=true]", "minecraft:oak_leaves[persistent=true]"),
                "glass_leaves": ("minecraft:glass", "minecraft:oak_leaves[persistent=true]"),
                "leaves_glass": ("minecraft:oak_leaves[persistent=true]", "minecraft:glass"),
                "glass_glass": ("minecraft:glass", "minecraft:glass"),
            }[side_pattern]
            commands.append(
                _relative_fill(
                    fx - lx * side_span,
                    0,
                    fz - lz * side_span,
                    fx - lx * side_span,
                    1,
                    fz - lz * side_span,
                    left_material,
                )
            )
            commands.append(
                _relative_fill(
                    fx + lx * side_span,
                    0,
                    fz + lz * side_span,
                    fx + lx * side_span,
                    1,
                    fz + lz * side_span,
                    right_material,
                )
            )
            commands.append(_relative_fill(fx, 0, fz, fx, 1, fz, "minecraft:glass"))
            return commands, {
                "variant_name": f"o2_straight_d{front_depth}_s{side_span}_{side_pattern}",
                "front_depth": int(front_depth),
                "side_span": int(side_span),
                "side_pattern": str(side_pattern),
                "center_material": "minecraft:glass",
                "left_material": str(left_material),
                "right_material": str(right_material),
            }
        commands.append(_relative_fill(fx - lx, 0, fz - lz, fx - lx, 1, fz - lz, material))
        commands.append(_relative_fill(fx + lx, 0, fz + lz, fx + lx, 1, fz + lz, material))
        if blueprint_id not in {"side_alcove_left", "side_alcove_right"}:
            commands.append(_relative_fill(fx, 0, fz, fx, 1, fz, "minecraft:glass"))
        return commands, {
            "variant_name": "o2_default",
            "front_depth": 1,
            "side_span": 1,
            "side_pattern": "leaves_leaves",
            "center_material": "minecraft:glass" if blueprint_id not in {"side_alcove_left", "side_alcove_right"} else "none",
            "left_material": material,
            "right_material": material,
        }
    else:
        # O3 should still be hard, but it must leave a thin valid sightline so
        # goal baking and evaluation do not depend on a fully hidden target.
        second_x = fx + int(outward[0])
        second_z = fz + int(outward[2])
        commands.append(_relative_fill(fx - lx, 0, fz - lz, fx - lx, 1, fz - lz, material))
        commands.append(_relative_fill(fx + lx, 0, fz + lz, fx + lx, 1, fz + lz, material))
        commands.append(_relative_fill(fx, 0, fz, fx, 1, fz, "minecraft:glass"))
        commands.append(_relative_fill(second_x - lx, 0, second_z - lz, second_x - lx, 1, second_z - lz, material))
        commands.append(_relative_fill(second_x + lx, 0, second_z + lz, second_x + lx, 1, second_z + lz, material))
        return commands, {
            "variant_name": "o3_default",
            "front_depth": 1,
            "side_span": 1,
            "side_pattern": "leaves_leaves",
            "center_material": "minecraft:glass",
            "left_material": material,
            "right_material": material,
        }


def _mine_distractor_commands(
    task_key: str,
    blueprint_id: str,
    target_local: tuple[int, int],
    clutter_level: int,
    rng: random.Random,
) -> List[str]:
    clutter_level = int(clutter_level)
    if clutter_level <= 0:
        return []
    tx, tz = int(target_local[0]), int(target_local[1])
    different_ore = "minecraft:emerald_ore" if task_key == "mine_coal" else "minecraft:coal_ore"
    same_ore = f"minecraft:{_mine_target_type(task_key)}"
    if blueprint_id == "offset_chamber":
        chamber_sign = -1 if tx < 0 else 1
        # Offset-chamber clutter should compete in the chamber view itself.
        # The old far-side placements often landed outside the canonical +x/-x
        # goal view, so C looked visually absent despite being present in YAML.
        candidates = [
            (tx, tz - 1),
            (tx, tz + 1),
            (tx - chamber_sign, tz - 1),
            (tx - chamber_sign, tz + 1),
            (tx, tz + 2),
            (tx - chamber_sign, tz + 2),
        ]
    elif blueprint_id in {"turn_left", "turn_right", "side_alcove_left", "side_alcove_right"}:
        wall_step = -1 if tx < 0 else 1
        candidates = [(tx, tz - 2), (tx, tz + 2), (tx + wall_step, tz - 1), (tx + wall_step, tz + 1)]
    else:
        candidates = [(-3, max(4, tz - 1)), (3, max(4, tz - 2)), (-4, tz), (4, tz), (-2, max(4, tz - 3)), (2, max(4, tz - 4))]
    deduped_candidates: List[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        point = (int(candidate[0]), int(candidate[1]))
        if point == (tx, tz):
            continue
        if point in seen:
            continue
        seen.add(point)
        deduped_candidates.append(point)
    candidates = deduped_candidates
    rng.shuffle(candidates)
    placements: List[tuple[tuple[int, int], str]] = []
    if clutter_level >= 1 and candidates:
        placements.append((candidates.pop(0), different_ore))
    if clutter_level >= 2 and candidates:
        placements.append((candidates.pop(0), same_ore))
    if clutter_level >= 3:
        if candidates:
            placements.append((candidates.pop(0), same_ore))
        if candidates:
            placements.append((candidates.pop(0), different_ore))
    commands: List[str] = []
    for (dx, dz), block in placements:
        commands.append(_relative_setblock(int(dx), 0, int(dz), block))
    return commands


def _mine_visibility_metrics(visibility_level: int) -> Dict:
    level = int(visibility_level)
    if level <= 0:
        return {
            "level": 0,
            "occluder_material": "none",
            "occluder_shape": "none",
            "front_span_blocks": 0,
            "depth_layers": 0,
        }
    if level == 1:
        return {
            "level": 1,
            "occluder_material": "minecraft:glass",
            "occluder_shape": "front_column",
            "front_span_blocks": 1,
            "depth_layers": 1,
        }
    if level == 2:
        return {
            "level": 2,
            "occluder_material": "minecraft:oak_leaves[persistent=true]+minecraft:glass",
            "occluder_shape": "front_windowed_wall",
            "front_span_blocks": 3,
            "depth_layers": 1,
            "visible_window_blocks": 1,
        }
    return {
        "level": level,
        "occluder_material": "minecraft:oak_leaves[persistent=true]+minecraft:glass",
        "occluder_shape": "front_wall_windowed_double_layer",
        "front_span_blocks": 3,
        "depth_layers": 2,
        "visible_window_blocks": 2,
    }


def _mine_occluder_variant_override(suggestions: Dict | None) -> Dict[str, Any]:
    if not isinstance(suggestions, dict):
        return {}
    variant: Dict[str, Any] = {}
    for source_key, target_key in (
        ("mine_occluder_front_depth", "front_depth"),
        ("mine_occluder_side_span", "side_span"),
        ("mine_occluder_side_pattern", "side_pattern"),
        ("mine_occluder_variant_id", "variant_id"),
        ("mine_occluder_top_row", "top_row"),
        ("mine_occluder_bottom_row", "bottom_row"),
        ("mine_occluder_leaf_material", "leaf_material"),
        ("mine_occluder_glass_material", "glass_material"),
    ):
        if source_key in suggestions:
            variant[target_key] = suggestions.get(source_key)
    return variant


def _mine_clutter_metrics(task_key: str, clutter_level: int) -> Dict:
    level = int(clutter_level)
    same_target_type_count = 0
    different_target_type_count = 0
    if level >= 1:
        different_target_type_count += 1
    if level >= 2:
        same_target_type_count += 1
    if level >= 3:
        same_target_type_count += 1
        different_target_type_count += 1
    return {
        "level": level,
        "target_type": _mine_target_type(task_key),
        "same_target_type_count": same_target_type_count,
        "different_target_type_count": different_target_type_count,
        "total_distractor_count": same_target_type_count + different_target_type_count,
    }


def _mine_path_metrics(
    blueprint_id: str,
    path_level: int,
    target_local: tuple[int, int],
    path_obstacle_variant: Optional[Dict[str, Any]] = None,
) -> Dict:
    level = int(path_level)
    tx, tz = int(target_local[0]), int(target_local[1])
    turn_layouts = {"turn_left", "turn_right"}
    side_alcove_layouts = {"side_alcove_left", "side_alcove_right"}
    variant_positions = list((path_obstacle_variant or {}).get("positions") or [])
    return {
        "level": level,
        "blueprint_id": str(blueprint_id),
        "target_local": [tx, tz],
        "corridor_turn_count": 1 if blueprint_id in turn_layouts else 0,
        "has_side_alcove": bool(blueprint_id in side_alcove_layouts),
        "has_offset_chamber": bool(blueprint_id == "offset_chamber"),
        "obstacle_count": len(variant_positions) if variant_positions else min(3, max(0, level)),
        "obstacle_variant_id": str((path_obstacle_variant or {}).get("variant_id") or ""),
        "obstacle_positions": variant_positions,
        "obstacle_height": int((path_obstacle_variant or {}).get("height") or 1) if variant_positions else 1,
        "obstacle_material": str((path_obstacle_variant or {}).get("material") or ""),
    }


def _mine_action_metrics(
    blueprint_id: str,
    action_level: int,
    *,
    layout_preferred_face: str,
    action_preferred_face: str,
    goal_preferred_face: str,
    explicit_opened_faces: List[str],
) -> Dict:
    level = int(action_level)
    core_open_faces = {label for label in _mine_core_open_faces(str(action_preferred_face)) if label}
    explicit_open_faces = {str(label).strip() for label in explicit_opened_faces if str(label).strip()}
    open_faces = core_open_faces | explicit_open_faces
    effective_layout_face = str(layout_preferred_face or "").strip()
    effective_action_face = str(action_preferred_face or effective_layout_face or "").strip()
    effective_goal_face = str(goal_preferred_face or effective_layout_face or "").strip()
    lateral_face_labels = {
        _face_label_from_offset(dx, dz)
        for dx, dz in _mine_lateral_offsets(effective_action_face)
    } if effective_action_face else set()
    lateral_face_labels.discard("")
    top_open = "+y" in open_faces
    return {
        "level": level,
        "layout_face_label": str(effective_layout_face),
        "action_face_label": str(effective_action_face),
        "goal_face_label": str(effective_goal_face),
        "core_open_face_labels": sorted(core_open_faces),
        "explicit_action_open_face_labels": sorted(explicit_open_faces),
        "open_face_labels": sorted(open_faces),
        "approach_face_label": str(effective_action_face),
        "approach_face_open": bool(effective_action_face and effective_action_face in open_faces),
        "lateral_open_count": int(sum(1 for label in lateral_face_labels if label in open_faces)),
        "top_open": bool(top_open),
    }


def build_procedural_mine_realized_metrics(
    task_key: str,
    factor_levels: Dict[str, int],
    blueprint_id: str,
    target_local: tuple[int, int],
    preferred_face: str,
    action_preferred_face: str,
    goal_preferred_face: str,
    spawn_position: list[float],
    target_world_center: list[float],
    base_yaw: float,
    spawn_yaw: float,
    spawn_pitch: float,
    heading_offset: float,
    explicit_action_open_faces: List[str],
    requested_heading_offset_deg: Optional[float] = None,
    requested_heading_offset_abs_deg: Optional[float] = None,
    requested_spawn_yaw_deg: Optional[float] = None,
    requested_spawn_pitch_deg: Optional[float] = None,
    exact_heading_override_used: bool = False,
    heading_override_mode: str = "factor_levels",
    occluder_variant: Optional[Dict[str, Any]] = None,
    path_obstacle_variant: Optional[Dict[str, Any]] = None,
) -> Dict:
    tx, tz = int(target_local[0]), int(target_local[1])
    normalized_levels = normalize_factor_levels(task_key, factor_levels)
    visibility_metrics = _mine_visibility_metrics(int(normalized_levels.get("O", 0)))
    if occluder_variant:
        visibility_metrics.update({str(key): value for key, value in occluder_variant.items()})
    return {
        "task_family": "mine",
        "task_key": str(task_key),
        "layout_backend": "procedural_mine_v1",
        "blueprint_id": str(blueprint_id),
        "preferred_face": str(preferred_face),
        "layout_preferred_face": str(preferred_face),
        "goal_preferred_face": str(goal_preferred_face),
        "goal_camera_preferred_face": str(goal_preferred_face),
        "spawn_position": [round(float(v), 4) for v in spawn_position],
        "target_world_center": [round(float(v), 4) for v in target_world_center],
        "target_local": [tx, tz],
        "spawn_target_horizontal_distance": round(math.sqrt(float(tx * tx + tz * tz)), 4),
        "base_yaw": round(float(base_yaw), 4),
        "spawn_yaw": round(float(spawn_yaw), 4),
        "spawn_pitch": round(float(spawn_pitch), 4),
        "heading_offset_deg": round(float(heading_offset), 4),
        "requested_heading_offset_deg": None if requested_heading_offset_deg is None else round(float(requested_heading_offset_deg), 4),
        "requested_heading_offset_abs_deg": None if requested_heading_offset_abs_deg is None else round(float(requested_heading_offset_abs_deg), 4),
        "requested_spawn_yaw_deg": None if requested_spawn_yaw_deg is None else round(float(requested_spawn_yaw_deg), 4),
        "requested_spawn_pitch_deg": None if requested_spawn_pitch_deg is None else round(float(requested_spawn_pitch_deg), 4),
        "exact_heading_override_used": bool(exact_heading_override_used),
        "heading_override_mode": str(heading_override_mode),
        "requested_factor_levels": normalized_levels,
        "visibility": visibility_metrics,
        "clutter": _mine_clutter_metrics(task_key, int(normalized_levels.get("C", 0))),
        "path": _mine_path_metrics(
            str(blueprint_id),
            int(normalized_levels.get("P", 0)),
            target_local,
            path_obstacle_variant=path_obstacle_variant,
        ),
        "action_margin": _mine_action_metrics(
            str(blueprint_id),
            int(normalized_levels.get("A", 0)),
            layout_preferred_face=str(preferred_face),
            action_preferred_face=str(action_preferred_face),
            goal_preferred_face=str(goal_preferred_face),
            explicit_opened_faces=list(explicit_action_open_faces),
        ),
    }


def _mine_blueprint_commands(
    blueprint_id: str,
    target_local: tuple[int, int],
    factor_levels: Dict[str, int],
    rng: random.Random,
    path_obstacle_variant: Optional[Dict[str, Any]] = None,
    arena_settings: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], str]:
    tx, tz = int(target_local[0]), int(target_local[1])
    max_extent_z = max(22, tz + 6)
    arena_settings = arena_settings or {}
    arena_half_width = int(arena_settings.get("half_width") or 5)
    arena_back_z = int(arena_settings.get("back_z") or -3)
    arena_front_buffer = int(arena_settings.get("front_buffer") or 3)
    arena_wall_height = int(arena_settings.get("wall_height") or 3)
    arena_funnel_start_z = int(arena_settings.get("funnel_start_z") or 3)
    arena_front_z = max(int(tz) + arena_front_buffer, 8)
    arena_mode = str(arena_settings.get("mode") or "")
    controlled_arena = arena_mode in {"wide_symmetric", "o2_target_funnel"}
    o2_target_funnel_arena = arena_mode == "o2_target_funnel"
    commands: List[str] = [
        "/time set day",
        "/weather clear",
        "/gamerule doDaylightCycle false",
        # Build the mine scene on a large artificial pad that clears both the
        # nearby terrain and any overhead blocks. This keeps source-free
        # procedural anchors away from ocean spawns, cave ceilings, and spawn
        # suffocation while preserving the same local mine task semantics.
        _relative_fill(-28, -4, -20, 28, 20, max_extent_z + 8, "minecraft:air"),
        _relative_fill(-28, -10, -20, 28, -2, max_extent_z + 8, "minecraft:stone"),
        _relative_fill(-28, -1, -20, 28, -1, max_extent_z + 8, "minecraft:grass_block"),
        # Do not rely only on the broad outer pad for walkability. The mine shell
        # itself needs an explicit local floor, otherwise the agent can end up on
        # a tiny support pad with void directly in front of it.
    ]
    if controlled_arena:
        wall_x = arena_half_width + 1
        # Re-author the local arena after the broad pad. This makes spawn-side
        # visual context and left/right navigable space symmetric while keeping a
        # wide path field for obstacle avoidance.
        commands.extend(
            [
                _relative_fill(-wall_x, -1, arena_back_z, wall_x, -1, arena_front_z, "minecraft:stone"),
                _relative_fill(-wall_x, 0, arena_back_z, wall_x, arena_wall_height, arena_front_z, "minecraft:air"),
                _relative_fill(-wall_x, 0, arena_back_z, -wall_x, arena_wall_height, arena_front_z, "minecraft:stone"),
                _relative_fill(wall_x, 0, arena_back_z, wall_x, arena_wall_height, arena_front_z, "minecraft:stone"),
                _relative_fill(-wall_x, 0, arena_back_z, wall_x, arena_wall_height, arena_back_z, "minecraft:stone"),
            ]
        )
        if bool(arena_settings.get("roof")):
            commands.append(_relative_fill(-wall_x, arena_wall_height + 1, arena_back_z, wall_x, arena_wall_height + 1, arena_front_z, "minecraft:stone"))
        if o2_target_funnel_arena:
            funnel_start_z = max(1, min(int(tz), int(arena_funnel_start_z)))
            # Keep the spawn/obstacle chamber symmetric, then restore the O2
            # straight-tunnel target geometry from z>=funnel_start_z. This keeps
            # path complexity isolated to the obstacle field while matching the
            # successful O2 target enclosure and action-margin openings.
            commands.extend(
                [
                    _relative_fill(-wall_x, 0, funnel_start_z, wall_x, arena_wall_height, arena_front_z, "minecraft:stone"),
                    _relative_fill(-1, 0, funnel_start_z, 1, 2, max(funnel_start_z, tz - 1), "minecraft:air"),
                ]
            )
    else:
        commands.extend(
            [
                _relative_fill(-6, -1, -2, 6, -1, 2, "minecraft:stone"),
                _relative_fill(-12, -1, 2, 12, -1, max_extent_z + 2, "minecraft:stone"),
                _relative_fill(-12, 0, 2, 12, 4, max_extent_z + 2, "minecraft:stone"),
                _relative_fill(-5, 0, -2, 5, 3, 2, "minecraft:air"),
            ]
        )
    preferred_face = _mine_preferred_face(blueprint_id)
    if blueprint_id == "straight_tunnel" and not controlled_arena:
        commands.append(_relative_fill(-1, 0, 2, 1, 2, max(2, tz - 1), "minecraft:air"))
    elif blueprint_id == "offset_chamber":
        chamber_start = max(4, tz - 2)
        commands.append(_relative_fill(-1, 0, 2, 1, 2, max(2, chamber_start - 1), "minecraft:air"))
        # Offset-chamber goal cameras reason about the chamber-side +/-x face.
        # The chamber carve therefore has to reach the target row itself; if it
        # stops at tz-1, then R1/R2/R3 targets keep the +/-x side sealed by
        # stone and every visible_face_plus/minus_x probe fails with
        # preferred_face_not_visible.
        commands.append(_relative_fill(-4, 0, chamber_start, 4, 2, max(chamber_start, tz), "minecraft:air"))
    elif blueprint_id == "turn_left":
        turn_z = max(4, tz)
        commands.append(_relative_fill(-1, 0, 2, 1, 2, turn_z, "minecraft:air"))
        commands.append(_relative_fill(tx + 1, 0, turn_z - 1, -1, 2, turn_z + 1, "minecraft:air"))
    elif blueprint_id == "turn_right":
        turn_z = max(4, tz)
        commands.append(_relative_fill(-1, 0, 2, 1, 2, turn_z, "minecraft:air"))
        commands.append(_relative_fill(1, 0, turn_z - 1, tx - 1, 2, turn_z + 1, "minecraft:air"))
    elif blueprint_id == "side_alcove_left":
        # Side-alcove targets are embedded in the corridor-side wall. The
        # corridor should continue past the target mouth, with the carved alcove
        # starting one block beyond the target so the +z face remains the frontal
        # visible face without opening a shortcut from spawn.
        alcove_z0 = max(4, tz + 1)
        alcove_z1 = max(alcove_z0, tz + 2)
        commands.append(_relative_fill(-1, 0, 2, 1, 2, alcove_z1, "minecraft:air"))
        commands.append(_relative_fill(tx, 0, alcove_z0, -1, 2, alcove_z1, "minecraft:air"))
    elif blueprint_id == "side_alcove_right":
        alcove_z0 = max(4, tz + 1)
        alcove_z1 = max(alcove_z0, tz + 2)
        commands.append(_relative_fill(-1, 0, 2, 1, 2, alcove_z1, "minecraft:air"))
        commands.append(_relative_fill(1, 0, alcove_z0, tx, 2, alcove_z1, "minecraft:air"))
    else:
        commands.append(_relative_fill(-1, 0, 2, 1, 2, max(2, tz - 1), "minecraft:air"))

    variant_positions = list((path_obstacle_variant or {}).get("positions") or [])
    if variant_positions:
        material = _normalize_minecraft_block_id(
            (path_obstacle_variant or {}).get("material"),
            default="minecraft:cobblestone",
        )
        try:
            obstacle_height = int((path_obstacle_variant or {}).get("height") or 1)
        except Exception:
            obstacle_height = 1
        obstacle_height = max(1, min(3, obstacle_height))
        for obstacle_x, obstacle_z in variant_positions:
            for obstacle_y in range(obstacle_height):
                commands.append(_relative_setblock(int(obstacle_x), int(obstacle_y), int(obstacle_z), material))
    else:
        path_level = int(factor_levels.get("P", 0))
        if path_level < 1:
            return commands, preferred_face
        for obstacle_x, obstacle_z in _mine_path_obstacle_positions(blueprint_id, target_local, rng)[: min(3, path_level)]:
            commands.append(_relative_setblock(int(obstacle_x), 0, int(obstacle_z), "minecraft:cobblestone"))
    return commands, preferred_face


def _mine_lighting_commands(blueprint_id: str, target_local: tuple[int, int]) -> List[str]:
    tx, tz = int(target_local[0]), int(target_local[1])
    positions: list[tuple[int, int]] = [(0, 3), (0, max(5, tz - 2))]
    if blueprint_id == "offset_chamber":
        chamber_start = max(5, tz - 4)
        positions.extend([(0, chamber_start), (0, max(chamber_start + 1, tz - 1))])
        if tx != 0:
            positions.append((int(max(-2, min(2, tx // 2 or (1 if tx > 0 else -1)))), max(chamber_start, tz - 1)))
    elif blueprint_id in {"turn_left", "turn_right"}:
        turn_z = max(5, tz)
        positions.extend([(0, turn_z), (int(tx // 2), turn_z)])
    elif blueprint_id in {"side_alcove_left", "side_alcove_right"}:
        alcove_z = max(5, tz)
        positions.extend([(0, alcove_z), (int(tx), alcove_z)])
    else:
        positions.append((0, max(6, tz - 1)))

    unique_positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, z in positions:
        point = (int(x), int(z))
        if point not in seen:
            seen.add(point)
            unique_positions.append(point)
    return [_relative_setblock(x, 3, z, "minecraft:sea_lantern") for x, z in unique_positions]


def build_procedural_mine_layout(
    task_key: str,
    source_data: Dict,
    factor_levels: Dict[str, int],
    rng: random.Random,
    suggestions: Optional[Dict[str, Any]] = None,
    blueprint_id_override: str | None = None,
    target_sign_override: str | None = None,
    target_local_override: tuple[int, int] | None = None,
    occluder_variant_override: Optional[Dict[str, Any]] = None,
    path_obstacle_variant_override: Optional[Dict[str, Any]] = None,
    anchor_mode: str = "source_or_fallback",
    disable_spawn_jitter: bool = False,
    fixed_spawn_position: Optional[List[float]] = None,
    arena_settings: Optional[Dict[str, Any]] = None,
    path_progress_reward_per_zone: float = 0.25,
) -> tuple[Dict, Dict]:
    normalized_levels = normalize_factor_levels(task_key, factor_levels)
    suggestions = suggestions or {}
    generated_data = copy.deepcopy(source_data)
    generated_data["time_limit"] = _mine_effective_time_limit(source_data)
    generated_data.pop("custom_init_commands", None)
    generated_data.pop("commands", None)
    generated_data.pop("target_blocks", None)
    generated_data.pop("targets", None)
    generated_data.pop("target_world_center", None)
    generated_data.pop("target_type", None)
    generated_data.pop("target_metadata", None)

    spawn_x, spawn_y, spawn_z, spawn_seed, spawn_anchor = _base_spawn_position(
        source_data,
        rng,
        anchor_mode=anchor_mode,
        disable_jitter=bool(disable_spawn_jitter),
        fixed_position=fixed_spawn_position,
    )
    blueprint_id = _normalize_mine_blueprint_override(blueprint_id_override) or _choose_mine_blueprint(int(normalized_levels.get("P", 0)), rng)
    target_local = target_local_override or _mine_target_local(
        blueprint_id,
        normalized_levels,
        rng,
        target_sign_override=target_sign_override,
    )
    target_type = _mine_target_type(task_key)
    layout_preferred_face = _mine_preferred_face(blueprint_id)
    target_world_center = [
        round(float(spawn_x) + float(target_local[0]), 4),
        round(float(spawn_y) - 0.5, 4),
        round(float(spawn_z) + float(target_local[1]), 4),
    ]
    goal_camera_look_target_world_center = [
        round(float(spawn_x) + float(target_local[0]), 4),
        round(float(spawn_y) + 0.5, 4),
        round(float(spawn_z) + float(target_local[1]), 4),
    ]
    spawn_position = [round(float(spawn_x), 4), round(float(spawn_y), 4), round(float(spawn_z), 4)]

    commands, layout_preferred_face = _mine_blueprint_commands(
        blueprint_id,
        target_local,
        normalized_levels,
        rng,
        path_obstacle_variant=path_obstacle_variant_override,
        arena_settings=arena_settings,
    )
    goal_camera_preferred_face = _mine_goal_camera_preferred_face(
        blueprint_id,
        target_local,
        layout_preferred_face,
    )
    action_preferred_face = _mine_action_preferred_face(
        blueprint_id,
        target_local,
        layout_preferred_face,
        goal_camera_preferred_face,
    )
    visibility_preferred_face = _mine_visibility_preferred_face(
        blueprint_id,
        target_local,
        layout_preferred_face,
        goal_camera_preferred_face,
    )
    heading_info = resolve_spawn_heading(
        task_key,
        data=generated_data,
        suggestions=suggestions,
        factor_levels=normalized_levels,
        rng=rng,
        default_base_yaw=0.0,
    ) or {
        "base_yaw": 0.0,
        "spawn_yaw": 0.0,
        "spawn_pitch": 0.0,
        "heading_offset_deg": 0.0,
        "requested_heading_offset_deg": None,
        "requested_heading_offset_abs_deg": None,
        "requested_spawn_yaw_deg": None,
        "requested_spawn_pitch_deg": None,
        "exact_heading_override_used": False,
        "heading_override_mode": "factor_levels",
    }
    base_yaw = float(heading_info["base_yaw"])
    spawn_yaw = float(heading_info["spawn_yaw"])
    spawn_pitch = float(heading_info["spawn_pitch"])
    heading_offset = float(heading_info.get("heading_offset_deg") or 0.0)
    commands.append(_relative_setblock(int(target_local[0]), 0, int(target_local[1]), f"minecraft:{target_type}"))
    action_neighborhood_commands, action_open_faces = _mine_target_neighborhood_commands(
        str(blueprint_id),
        target_local,
        action_preferred_face,
        int(normalized_levels.get("A", 0)),
        rng,
    )
    commands.extend(action_neighborhood_commands)
    if str((arena_settings or {}).get("mode") or "") in {"wide_symmetric", "o2_target_funnel"}:
        commands.extend(
            _mine_target_back_wall_commands(
                str(blueprint_id),
                target_local,
                height=int((arena_settings or {}).get("wall_height") or 3),
                half_width=int((arena_settings or {}).get("half_width") or 5),
            )
        )
    occluder_commands, occluder_variant = _mine_occluder_commands(
        target_local,
        visibility_preferred_face,
        int(normalized_levels.get("O", 0)),
        rng,
        blueprint_id=blueprint_id,
        variant_override=occluder_variant_override,
    )
    commands.extend(occluder_commands)
    commands.extend(_mine_distractor_commands(task_key, blueprint_id, target_local, int(normalized_levels.get("C", 0)), rng))
    commands.extend(_mine_lighting_commands(blueprint_id, target_local))
    commands.insert(0, _absolute_tp_command(spawn_x, spawn_y, spawn_z, spawn_yaw, spawn_pitch))
    commands.append(_absolute_tp_command(spawn_x, spawn_y, spawn_z, spawn_yaw, spawn_pitch))
    realized_metrics = build_procedural_mine_realized_metrics(
        task_key=task_key,
        factor_levels=normalized_levels,
        blueprint_id=str(blueprint_id),
        target_local=target_local,
        preferred_face=str(layout_preferred_face),
        action_preferred_face=str(action_preferred_face),
        goal_preferred_face=str(goal_camera_preferred_face),
        spawn_position=spawn_position,
        target_world_center=target_world_center,
        base_yaw=base_yaw,
        spawn_yaw=spawn_yaw,
        spawn_pitch=spawn_pitch,
        heading_offset=heading_offset,
        explicit_action_open_faces=action_open_faces,
        requested_heading_offset_deg=heading_info.get("requested_heading_offset_deg"),
        requested_heading_offset_abs_deg=heading_info.get("requested_heading_offset_abs_deg"),
        requested_spawn_yaw_deg=heading_info.get("requested_spawn_yaw_deg"),
        requested_spawn_pitch_deg=heading_info.get("requested_spawn_pitch_deg"),
        exact_heading_override_used=bool(heading_info.get("exact_heading_override_used")),
        heading_override_mode=str(heading_info.get("heading_override_mode") or "factor_levels"),
        occluder_variant=occluder_variant,
        path_obstacle_variant=path_obstacle_variant_override,
    )
    generated_data["spawn_positions"] = [
        {
            "seed": int(spawn_seed),
            "position": list(spawn_position),
            "yaw": round(float(spawn_yaw), 4),
            "pitch": round(float(spawn_pitch), 4),
        }
    ]
    generated_data["biomes"] = list(source_data.get("biomes") or ["plains"])
    generated_data["random_tp_range"] = int(source_data.get("random_tp_range") or 1000)
    generated_data["init_inventory"] = _mine_inherited_inventory(source_data, task_key)
    generated_data["custom_init_commands"] = commands
    generated_data["spawn_support_pad"] = {"cleanup_after_build": True}
    generated_data["target_blocks"] = [{"type": target_type, "world_center": list(target_world_center)}]
    generated_data["target_world_center"] = list(target_world_center)
    generated_data["target_type"] = str(target_type)
    generated_data["preferred_face"] = str(layout_preferred_face)
    generated_data["layout_preferred_face"] = str(layout_preferred_face)
    generated_data["goal_camera_preferred_face"] = str(goal_camera_preferred_face)
    generated_data["goal_preferred_face"] = str(goal_camera_preferred_face)
    generated_data["procedural_layout"] = {
        "backend": "procedural_mine_v1",
        "blueprint_id": str(blueprint_id),
        "target_local": [int(target_local[0]), int(target_local[1])],
        "goal_camera_look_target_world_center": list(goal_camera_look_target_world_center),
        "arena_settings": arena_settings or {},
        "preferred_face": str(layout_preferred_face),
        "layout_preferred_face": str(layout_preferred_face),
        "goal_camera_preferred_face": str(goal_camera_preferred_face),
        "goal_preferred_face": str(goal_camera_preferred_face),
        "realized_metrics": realized_metrics,
        "path_obstacle_variant": path_obstacle_variant_override or {},
        "spawn_heading": {
            "base_yaw": round(float(base_yaw), 4),
            "offset_deg": round(float(heading_offset), 4),
            "heading_offset_deg": round(float(heading_offset), 4),
            "spawn_yaw": round(float(spawn_yaw), 4),
            "spawn_pitch": round(float(spawn_pitch), 4),
            "requested_heading_offset_deg": None if heading_info.get("requested_heading_offset_deg") is None else round(float(heading_info["requested_heading_offset_deg"]), 4),
            "requested_heading_offset_abs_deg": None if heading_info.get("requested_heading_offset_abs_deg") is None else round(float(heading_info["requested_heading_offset_abs_deg"]), 4),
            "requested_spawn_yaw_deg": None if heading_info.get("requested_spawn_yaw_deg") is None else round(float(heading_info["requested_spawn_yaw_deg"]), 4),
            "requested_spawn_pitch_deg": None if heading_info.get("requested_spawn_pitch_deg") is None else round(float(heading_info["requested_spawn_pitch_deg"]), 4),
            "exact_heading_override_used": bool(heading_info.get("exact_heading_override_used")),
            "heading_override_mode": str(heading_info.get("heading_override_mode") or "factor_levels"),
        },
    }
    reward_config = None
    if (
        str(blueprint_id) == "straight_tunnel"
        and isinstance(arena_settings, dict)
        and int(arena_settings.get("funnel_start_z", 0) or 0) >= 4
        and isinstance(path_obstacle_variant_override, dict)
        and path_obstacle_variant_override.get("positions")
    ):
        reward_config = build_path_progress_reward_config(
            target_local=target_local,
            obstacle_positions=path_obstacle_variant_override.get("positions") or [],
            arena_settings=arena_settings,
            reward_per_zone=float(path_progress_reward_per_zone),
        )
    if reward_config:
        generated_data["path_progress_reward"] = reward_config
    applied = {
        "layout_backend": "procedural_mine_v1",
        "layout_tags": [str(blueprint_id)],
        "scene_shift": {"dx": 0, "dz": 0},
        "spawn_shift": {"dx": 0, "dz": 0},
        "added_commands": list(commands),
        "factor_levels": normalized_levels,
        "spawn_heading": {
            "base_yaw": round(float(base_yaw), 4),
            "offset_deg": round(float(heading_offset), 4),
            "heading_offset_deg": round(float(heading_offset), 4),
            "spawn_yaw": round(float(spawn_yaw), 4),
            "spawn_pitch": round(float(spawn_pitch), 4),
            "requested_heading_offset_deg": None if heading_info.get("requested_heading_offset_deg") is None else round(float(heading_info["requested_heading_offset_deg"]), 4),
            "requested_heading_offset_abs_deg": None if heading_info.get("requested_heading_offset_abs_deg") is None else round(float(heading_info["requested_heading_offset_abs_deg"]), 4),
            "requested_spawn_yaw_deg": None if heading_info.get("requested_spawn_yaw_deg") is None else round(float(heading_info["requested_spawn_yaw_deg"]), 4),
            "requested_spawn_pitch_deg": None if heading_info.get("requested_spawn_pitch_deg") is None else round(float(heading_info["requested_spawn_pitch_deg"]), 4),
            "exact_heading_override_used": bool(heading_info.get("exact_heading_override_used")),
            "heading_override_mode": str(heading_info.get("heading_override_mode") or "factor_levels"),
        },
        "spawn_anchor": spawn_anchor,
        "spawn_position": list(spawn_position),
        "blueprint_id": str(blueprint_id),
        "target_local": [int(target_local[0]), int(target_local[1])],
        "preferred_face": str(layout_preferred_face),
        "layout_preferred_face": str(layout_preferred_face),
        "goal_camera_preferred_face": str(goal_camera_preferred_face),
        "goal_preferred_face": str(goal_camera_preferred_face),
        "target_world_center": list(target_world_center),
        "realized_factor_levels": normalized_levels,
        "realized_factor_metrics": realized_metrics,
        "occluder_variant": occluder_variant,
        "path_obstacle_variant": path_obstacle_variant_override or {},
    }
    return generated_data, applied


def sign_for_task(task_key: str) -> int:
    x_hint = TARGET_HINTS.get(task_key, (1, 0))[0]
    return 1 if x_hint >= 0 else -1


def derive_layout_tags(suggestions: Dict) -> List[str]:
    layout_tags = set((suggestions or {}).get("layout_changes", []) or [])
    if suggestions.get("visibility") == "increase":
        layout_tags.add("partial_occluder")
    elif suggestions.get("visibility") == "decrease":
        layout_tags.add("make_target_visible_at_step0")

    if suggestions.get("distractors") == "increase":
        layout_tags.add("add_similar_distractor")

    if suggestions.get("path_difficulty") == "increase":
        layout_tags.add("add_medium_obstacle")
    elif suggestions.get("path_difficulty") == "decrease":
        layout_tags.add("keep_layout")

    if suggestions.get("view_difficulty") == "increase":
        layout_tags.add("shift_target_off_center")
        layout_tags.add("move_spawn_back")
    elif suggestions.get("view_difficulty") == "decrease":
        layout_tags.add("make_target_visible_at_step0")
    return sorted(layout_tags or {"keep_layout"})


def compute_scene_shift(task_key: str, layout_tags: List[str]) -> Tuple[int, int]:
    x_hint, _ = TARGET_HINTS.get(task_key, (0, 5))
    direction = sign_for_task(task_key)
    dx = 0
    dz = 0
    if "make_target_visible_at_step0" in layout_tags:
        if x_hint <= -2:
            dx += 1
        elif x_hint >= 2:
            dx -= 1
        dz -= 1
    if "shift_target_off_center" in layout_tags:
        dx += 2 * direction
    if "shift_target_farther" in layout_tags or "move_spawn_back" in layout_tags:
        dz += 2
    return dx, dz


def ensure_command_list(data: Dict) -> List[str]:
    commands = list(data.get("custom_init_commands") or [])
    data["custom_init_commands"] = commands
    return commands


def add_visibility_commands(
    task_key: str,
    data: Dict,
    commands: List[str],
    level: int = 1,
    harder: bool = True,
    rng: Optional[random.Random] = None,
):
    x_hint, z_hint = infer_target_local_offset(task_key, data)
    rng = rng or random.Random(0)
    z_occ = max(2, z_hint - rng.choice([1, 2]))
    if harder:
        if int(level) <= 1:
            material = rng.choice(["minecraft:glass", "minecraft:glass", "minecraft:oak_leaves[persistent=true]"])
            width = 0
            lateral_shift = rng.choice([-1, 0, 1])
            commands.append(
                f"/fill ~{x_hint + lateral_shift} ~ ~{z_occ} ~{x_hint + lateral_shift} ~1 ~{z_occ} {material}"
            )
            return
        material = rng.choice(["minecraft:glass", "minecraft:oak_leaves[persistent=true]"]) if int(level) == 2 else "minecraft:oak_leaves[persistent=true]"
        width = 1 if int(level) <= 2 else 2
        center_shift = rng.choice([-1, 0, 1]) if int(level) == 2 else 0
        commands.append(
            f"/fill ~{x_hint + center_shift - width} ~ ~{z_occ} ~{x_hint + center_shift + width} ~1 ~{z_occ} {material}"
        )
        if int(level) >= 3:
            z_occ_2 = max(2, z_occ - 1)
            commands.append(f"/fill ~{x_hint} ~ ~{z_occ} ~{x_hint} ~1 ~{z_occ} minecraft:glass")
            commands.append(f"/fill ~{x_hint - 1} ~ ~{z_occ_2} ~{x_hint - 1} ~1 ~{z_occ_2} minecraft:oak_leaves[persistent=true]")
            commands.append(f"/fill ~{x_hint + 1} ~ ~{z_occ_2} ~{x_hint + 1} ~1 ~{z_occ_2} minecraft:oak_leaves[persistent=true]")
    else:
        commands.append("/fill ~-1 ~ ~3 ~1 ~1 ~3 minecraft:air")


def add_obstacle_commands(task_key: str, data: Dict, commands: List[str], strength: str, rng: Optional[random.Random] = None):
    x_hint, z_hint = infer_target_local_offset(task_key, data)
    rng = rng or random.Random(0)
    z_obs = max(2, z_hint - rng.choice([1, 2]))
    if strength == "low":
        x_obs = x_hint + rng.choice([-1, 0, 1])
        commands.append(f"/setblock ~{x_obs} ~ ~{z_obs} minecraft:cobblestone")
    elif strength == "medium":
        lateral_span = rng.choice([1, 2])
        commands.append(f"/setblock ~{x_hint - lateral_span} ~ ~{z_obs} minecraft:cobblestone")
        commands.append(f"/setblock ~{x_hint + lateral_span} ~ ~{z_obs} minecraft:cobblestone")
        if rng.random() < 0.5:
            commands.append(f"/setblock ~{x_hint} ~ ~{max(2, z_obs - 1)} minecraft:cobblestone")


def infer_target_local_offset(task_key: str, data: Dict) -> Tuple[int, int]:
    x_hint, z_hint = TARGET_HINTS.get(task_key, (0, 5))
    spawn_positions = list(data.get("spawn_positions") or [])
    if not spawn_positions:
        return int(x_hint), int(z_hint)
    spawn_position = list((spawn_positions[0] or {}).get("position") or [])
    target_center = data.get("target_world_center")
    if not (
        isinstance(target_center, (list, tuple))
        and len(target_center) >= 3
        and len(spawn_position) >= 3
    ):
        return int(x_hint), int(z_hint)
    try:
        rel_x = int(round(float(target_center[0]) - float(spawn_position[0])))
        rel_z = int(round(float(target_center[2]) - float(spawn_position[2])))
        return rel_x, rel_z
    except Exception:
        return int(x_hint), int(z_hint)


def canonical_approach_axis(task_key: str, data: Dict) -> str:
    rel_x, rel_z = infer_target_local_offset(task_key, data)
    return "z" if abs(rel_z) >= abs(rel_x) else "x"


def spawn_target_yaw(task_key: str, data: Dict) -> float | None:
    spawn_positions = list(data.get("spawn_positions") or [])
    if not spawn_positions:
        return None
    spawn_position = list((spawn_positions[0] or {}).get("position") or [])
    target_center = data.get("target_world_center")
    if not (
        isinstance(target_center, (list, tuple))
        and len(target_center) >= 3
        and len(spawn_position) >= 3
    ):
        return None
    try:
        dx = float(target_center[0]) - float(spawn_position[0])
        dz = float(target_center[2]) - float(spawn_position[2])
    except Exception:
        return None
    return float(math.degrees(math.atan2(-dx, dz)))


def apply_heading_offset(base_yaw: float, offset_deg: float, task_key: str) -> float:
    direction = sign_for_task(task_key)
    signed = float(offset_deg) if int(direction) >= 0 else -float(offset_deg)
    return normalize_yaw_deg(float(base_yaw) + signed)


def resolve_spawn_heading(
    task_key: str,
    *,
    data: Dict,
    suggestions: Dict,
    factor_levels: Dict[str, int],
    rng: Optional[random.Random] = None,
    default_base_yaw: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    rng = rng or random.Random(0)
    requested_spawn_yaw_deg = _optional_float(suggestions.get("mine_spawn_yaw_deg"))
    requested_spawn_pitch_deg = _optional_float(suggestions.get("mine_spawn_pitch_deg"))
    requested_heading_offset_deg = _optional_float(suggestions.get("mine_heading_offset_deg"))
    requested_heading_offset_abs_deg = _optional_float(suggestions.get("mine_heading_offset_abs_deg"))
    spawn_pitch = 0.0 if requested_spawn_pitch_deg is None else float(requested_spawn_pitch_deg)

    base_yaw = _optional_float(default_base_yaw)
    if base_yaw is None:
        base_yaw = spawn_target_yaw(task_key, data)

    if requested_spawn_yaw_deg is not None:
        spawn_yaw = normalize_yaw_deg(float(requested_spawn_yaw_deg))
        heading_offset_deg = None if base_yaw is None else yaw_delta_deg(base_yaw, spawn_yaw)
        exact_heading_override_used = True
        heading_override_mode = "spawn_yaw_deg"
    elif requested_heading_offset_deg is not None:
        if base_yaw is None:
            return None
        spawn_yaw = normalize_yaw_deg(float(base_yaw) + float(requested_heading_offset_deg))
        heading_offset_deg = yaw_delta_deg(base_yaw, spawn_yaw)
        exact_heading_override_used = True
        heading_override_mode = "heading_offset_deg"
    elif requested_heading_offset_abs_deg is not None:
        if base_yaw is None:
            return None
        spawn_yaw = apply_heading_offset(base_yaw, abs(float(requested_heading_offset_abs_deg)), task_key)
        heading_offset_deg = yaw_delta_deg(base_yaw, spawn_yaw)
        exact_heading_override_used = True
        heading_override_mode = "heading_offset_abs_deg"
    else:
        if base_yaw is None:
            return None
        sampled_offset = spawn_heading_offset_from_factor_levels(task_key, factor_levels, rng=rng)
        spawn_yaw = apply_heading_offset(base_yaw, sampled_offset, task_key)
        heading_offset_deg = yaw_delta_deg(base_yaw, spawn_yaw)
        exact_heading_override_used = False
        heading_override_mode = "factor_levels"

    return {
        "base_yaw": float(0.0 if base_yaw is None else base_yaw),
        "spawn_yaw": float(spawn_yaw),
        "spawn_pitch": float(spawn_pitch),
        "heading_offset_deg": None if heading_offset_deg is None else float(heading_offset_deg),
        "requested_heading_offset_deg": requested_heading_offset_deg,
        "requested_heading_offset_abs_deg": requested_heading_offset_abs_deg,
        "requested_spawn_yaw_deg": requested_spawn_yaw_deg,
        "requested_spawn_pitch_deg": requested_spawn_pitch_deg,
        "exact_heading_override_used": bool(exact_heading_override_used),
        "heading_override_mode": str(heading_override_mode),
    }


def lateral_neighbor_offsets(axis: str) -> List[Tuple[int, int]]:
    if axis == "x":
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def add_action_margin_commands(task_key: str, data: Dict, commands: List[str], level: int, rng: Optional[random.Random] = None):
    if task_key not in {"mine_coal", "mine_emerald"}:
        return
    x_hint, z_hint = infer_target_local_offset(task_key, data)
    level = int(level)
    axis = canonical_approach_axis(task_key, data)
    laterals = lateral_neighbor_offsets(axis)
    rng = rng or random.Random(0)
    rng.shuffle(laterals)
    # Mine-family A controls the number of approach-relevant exposed faces.
    # We keep the canonical approach face open and progressively seal:
    # A1: one lateral face
    # A2: both lateral faces
    # A3: both lateral faces + top face
    if level >= 1:
        dx, dz = laterals[0]
        commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:stone")
    if level >= 2:
        dx, dz = laterals[1]
        commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:stone")
    if level >= 3:
        commands.append(f"/setblock ~{x_hint} ~1 ~{z_hint} minecraft:stone")


def add_distractor_layout(task_key: str, data: Dict, level: int = 1, rng: Optional[random.Random] = None):
    commands = ensure_command_list(data)
    summon_mobs = list(data.get("summon_mobs") or [])
    x_hint, z_hint = infer_target_local_offset(task_key, data)
    rng = rng or random.Random(0)

    if task_key == "hunt_sheep_right_fence":
        if summon_mobs:
            summon_mobs[0]["number"] = int(summon_mobs[0].get("number", 3)) + 2
            data["summon_mobs"] = summon_mobs
            return
    elif task_key == "hunt_cow_do_not_touch_sheep":
        if summon_mobs:
            summon_mobs[0]["number"] = int(summon_mobs[0].get("number", 2)) + 2
            data["summon_mobs"] = summon_mobs
            return
    elif task_key == "interact_left_chest":
        commands.append("/setblock ~4 ~ ~5 minecraft:chest[facing=north]")
        return
    elif task_key == "mine_emerald":
        candidate_offsets = [(-1, 0), (2, 1), (-3, 1), (1, -1), (-2, -1), (3, 0)]
        rng.shuffle(candidate_offsets)
        if int(level) >= 1:
            dx, dz = candidate_offsets[0]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:coal_ore")
        if int(level) >= 2:
            dx, dz = candidate_offsets[1]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:emerald_ore")
        if int(level) >= 3:
            dx, dz = candidate_offsets[2]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:emerald_ore")
        return
    elif task_key == "mine_coal":
        candidate_offsets = [(1, 0), (-2, 1), (2, 1), (-1, -1), (3, 0), (-3, 0)]
        rng.shuffle(candidate_offsets)
        if int(level) >= 1:
            dx, dz = candidate_offsets[0]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:emerald_ore")
        if int(level) >= 2:
            dx, dz = candidate_offsets[1]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:coal_ore")
        if int(level) >= 3:
            dx, dz = candidate_offsets[2]
            commands.append(f"/setblock ~{x_hint + dx} ~ ~{z_hint + dz} minecraft:coal_ore")
        return
    elif task_key == "set_fire_on_tree":
        commands.append("/setblock ~2 ~ ~6 minecraft:oak_log")
        commands.append("/setblock ~2 ~1 ~6 minecraft:oak_leaves[persistent=true]")
        return
    elif task_key == "use_bucket_get_lava":
        commands.append("/setblock ~-4 ~-1 ~4 minecraft:water")
        commands.append("/setblock ~-4 ~-1 ~3 minecraft:water")
        return
    elif task_key == "place_minecart_on_rail":
        commands.extend(
            [
                "/setblock ~-3 ~0 ~6 minecraft:rail",
                "/setblock ~-2 ~0 ~6 minecraft:rail",
                "/setblock ~-1 ~0 ~6 minecraft:rail",
                "/setblock ~0 ~0 ~6 minecraft:rail",
                "/setblock ~1 ~0 ~6 minecraft:rail",
                "/setblock ~2 ~0 ~6 minecraft:rail",
                "/setblock ~3 ~0 ~6 minecraft:rail",
            ]
        )
        return
    elif task_key == "place_oak_door_on_diamond_block":
        commands.append("/setblock ~-4 ~0 ~4 minecraft:emerald_block")
        return


def apply_layout_changes(
    task_key: str,
    data: Dict,
    suggestions: Dict,
    factor_levels: Dict[str, int] | None = None,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict, Dict]:
    data = copy.deepcopy(data)
    normalized_factor_levels = normalize_factor_levels(task_key, factor_levels or {})
    has_explicit_factor_levels = factor_levels is not None
    layout_tags = derive_layout_tags(suggestions)
    rng = rng or random.Random(0)
    applied: Dict[str, object] = {
        "layout_tags": layout_tags,
        "scene_shift": {"dx": 0, "dz": 0},
        "spawn_shift": {"dx": 0, "dz": 0},
        "added_commands": [],
        "factor_levels": normalized_factor_levels,
    }

    if task_key in COMMAND_LAYOUT_TASKS:
        dx, dz = compute_scene_shift(task_key, layout_tags)
        if "shift_target_off_center" in layout_tags:
            dx += rng.choice([-1, 0, 1]) * sign_for_task(task_key)
        if "shift_target_farther" in layout_tags:
            dz += rng.choice([1, 2])
        applied["scene_shift"] = {"dx": int(dx), "dz": int(dz)}
        if dx or dz:
            data["custom_init_commands"] = shift_custom_commands(list(data.get("custom_init_commands") or []), dx=dx, dz=dz)
            data["summon_mobs"] = shift_summon_mobs(list(data.get("summon_mobs") or []), dx=dx, dz=dz)
    else:
        spawn_dx, spawn_dz = spawn_shift_from_factor_levels(task_key, normalized_factor_levels, rng=rng)
        if not has_explicit_factor_levels:
            if spawn_dx == 0 and ("move_spawn_lateral" in layout_tags or suggestions.get("view_difficulty") == "increase"):
                spawn_dx += rng.choice([1, 2, 3]) * sign_for_task(task_key)
            if spawn_dz == 0 and ("move_spawn_back" in layout_tags or suggestions.get("view_difficulty") == "increase"):
                spawn_dz -= rng.choice([2, 3, 4])
        if not has_explicit_factor_levels and "make_target_visible_at_step0" in layout_tags:
            spawn_dz += 1
        applied["spawn_shift"] = {"dx": int(spawn_dx), "dz": int(spawn_dz)}
        if spawn_dx or spawn_dz:
            data["spawn_positions"] = shift_spawn_positions(list(data.get("spawn_positions") or []), dx=spawn_dx, dz=spawn_dz)
            commands = ensure_command_list(data)
            before = len(commands)
            add_spawn_safety_commands(data, commands)
            applied["added_commands"].extend(commands[before:])
        heading_info = resolve_spawn_heading(
            task_key,
            data=data,
            suggestions=suggestions,
            factor_levels=normalized_factor_levels,
            rng=rng,
        )
        if heading_info is not None:
            base_yaw = float(heading_info["base_yaw"])
            spawn_yaw = float(heading_info["spawn_yaw"])
            spawn_pitch = float(heading_info["spawn_pitch"])
            heading_offset = float(heading_info.get("heading_offset_deg") or 0.0)
            data["spawn_positions"] = orient_spawn_positions(list(data.get("spawn_positions") or []), yaw=spawn_yaw, pitch=spawn_pitch)
            applied["spawn_heading"] = {
                "base_yaw": round(float(base_yaw), 4),
                "offset_deg": round(float(heading_offset), 4),
                "heading_offset_deg": round(float(heading_offset), 4),
                "spawn_yaw": round(float(spawn_yaw), 4),
                "spawn_pitch": round(float(spawn_pitch), 4),
                "requested_heading_offset_deg": None if heading_info.get("requested_heading_offset_deg") is None else round(float(heading_info["requested_heading_offset_deg"]), 4),
                "requested_heading_offset_abs_deg": None if heading_info.get("requested_heading_offset_abs_deg") is None else round(float(heading_info["requested_heading_offset_abs_deg"]), 4),
                "requested_spawn_yaw_deg": None if heading_info.get("requested_spawn_yaw_deg") is None else round(float(heading_info["requested_spawn_yaw_deg"]), 4),
                "requested_spawn_pitch_deg": None if heading_info.get("requested_spawn_pitch_deg") is None else round(float(heading_info["requested_spawn_pitch_deg"]), 4),
                "exact_heading_override_used": bool(heading_info.get("exact_heading_override_used")),
                "heading_override_mode": str(heading_info.get("heading_override_mode") or "factor_levels"),
            }

    if "partial_occluder" in layout_tags:
        commands = ensure_command_list(data)
        before = len(commands)
        add_visibility_commands(task_key, data, commands, level=max(1, int(normalized_factor_levels.get("O", 1) or 1)), harder=True, rng=rng)
        applied["added_commands"].extend(commands[before:])
    elif suggestions.get("visibility") == "decrease":
        commands = ensure_command_list(data)
        before = len(commands)
        add_visibility_commands(task_key, data, commands, harder=False, rng=rng)
        applied["added_commands"].extend(commands[before:])

    if "add_low_obstacle" in layout_tags or suggestions.get("path_difficulty") == "increase":
        commands = ensure_command_list(data)
        before = len(commands)
        strength = "medium" if ("add_medium_obstacle" in layout_tags or int(normalized_factor_levels.get("P", 0)) >= 2) else "low"
        add_obstacle_commands(task_key, data, commands, strength=strength, rng=rng)
        applied["added_commands"].extend(commands[before:])

    if "add_similar_distractor" in layout_tags or suggestions.get("distractors") == "increase":
        commands = ensure_command_list(data)
        before = len(commands)
        add_distractor_layout(task_key, data, level=max(1, int(normalized_factor_levels.get("C", 1) or 1)), rng=rng)
        commands_after = list(data.get("custom_init_commands") or [])
        applied["added_commands"].extend(commands_after[before:])

    if "narrow_action_margin" in layout_tags or int(normalized_factor_levels.get("A", 0)) > 0:
        commands = ensure_command_list(data)
        before = len(commands)
        add_action_margin_commands(task_key, data, commands, level=int(normalized_factor_levels.get("A", 1) or 1), rng=rng)
        applied["added_commands"].extend(commands[before:])

    return data, applied


def collect_source_yaml_candidates(env_conf_dir: Path, task_config_name: str) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path: Path):
        if not path.exists() or not path.is_file():
            return
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    add_candidate(env_conf_dir / f"{task_config_name}.yaml")
    if env_conf_dir.exists():
        for path in sorted(env_conf_dir.rglob(f"{task_config_name}.yaml")):
            add_candidate(path)
    return candidates


def resolve_source_yaml(env_conf_dir: Path, task_config_name: str, template_index: int = 0) -> tuple[Path, int, int]:
    candidates = collect_source_yaml_candidates(env_conf_dir, task_config_name)
    if not candidates:
        raise FileNotFoundError(
            f"Task config '{task_config_name}.yaml' not found under source directory: {env_conf_dir}"
        )
    count = len(candidates)
    normalized_index = int(template_index) % count
    return candidates[normalized_index], normalized_index, count


def resolve_builtin_benchmark_yaml(task_config_name: str) -> Path:
    candidate = DEFAULT_BENCHMARK_TASK_CONFIG_DIR / f"{task_config_name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Benchmark task config '{task_config_name}.yaml' not found under: {DEFAULT_BENCHMARK_TASK_CONFIG_DIR}"
        )
    return candidate


def build_manifest_row(plan_row: Dict, output_path: Path, applied_changes: Dict) -> Dict:
    task_key = canonical_task_key(plan_row["task_config_name"])
    factor_levels = normalize_factor_levels(task_key, plan_row.get("factor_levels", {}))
    realized_factor_levels = normalize_factor_levels(
        task_key,
        applied_changes.get("realized_factor_levels") or factor_levels,
    )
    return {
        "task_config_name": plan_row["task_config_name"],
        "task_key": task_key,
        "primary_failure_mode_majority": plan_row.get("primary_failure_mode_majority", "unknown"),
        "primary_factors_majority": plan_row.get("primary_factors_majority", []),
        "severity_majority": plan_row.get("severity_majority"),
        "factor_levels": factor_levels,
        "layout_seed": int(plan_row.get("layout_seed", 0) or 0),
        "template_index": int(plan_row.get("template_index", applied_changes.get("template_index", 0)) or 0),
        "template_count": int(plan_row.get("template_count", applied_changes.get("template_count", 1)) or 1),
        "requested_split_label": str(plan_row.get("requested_split_label", "")).strip(),
        "computed_split_label": str(plan_row.get("computed_split_label", "")).strip() or classify_factor_split(task_key, factor_levels),
        "hard_factor_count": int(plan_row.get("hard_factor_count", hard_factor_count(task_key, factor_levels))),
        "trainable_with_rl_majority": bool(plan_row.get("trainable_with_rl_majority", True)),
        "world_generation_suggestions": plan_row.get("world_generation_suggestions", {}),
        "output_yaml": str(output_path.resolve()),
        "layout_backend": applied_changes.get("layout_backend", ""),
        "blueprint_id": applied_changes.get("blueprint_id"),
        "preferred_face": applied_changes.get("preferred_face"),
        "layout_preferred_face": applied_changes.get("layout_preferred_face"),
        "goal_camera_preferred_face": applied_changes.get("goal_camera_preferred_face"),
        "spawn_position": applied_changes.get("spawn_position"),
        "realized_factor_levels": realized_factor_levels,
        "realized_split_label": classify_factor_split(task_key, realized_factor_levels),
        "realized_factor_metrics": applied_changes.get("realized_factor_metrics", {}),
        "applied_changes": applied_changes,
        "configured_target": applied_changes.get("configured_target"),
        "target_source": applied_changes.get("target_source"),
        "target_world_center": applied_changes.get("target_world_center"),
        "source_template_path": applied_changes.get("source_template_path"),
    }


def main():
    args = parse_args()
    plan_json = Path(args.plan_json)
    env_conf_dir = Path(args.env_conf_dir)
    base_out_dir = Path(args.out_dir)
    run_dir = base_out_dir / time.strftime("%Y%m%d_%H%M%S")

    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = load_json(plan_json)
    manifest_rows: List[Dict] = []

    for plan_row in plan_rows:
        if args.only_trainable and not bool(plan_row.get("trainable_with_rl_majority", True)):
            continue

        task_config_name = str(plan_row["task_config_name"]).strip()
        task_key = canonical_task_key(task_config_name)
        factor_levels = normalize_factor_levels(task_key, plan_row.get("factor_levels") or {})
        layout_rng = build_layout_rng(plan_row)
        if any(int(v) > 0 for v in factor_levels.values()):
            suggestions = factor_levels_to_worldgen_suggestions(task_key, factor_levels)
        else:
            suggestions = normalize_worldgen_suggestions(task_key, plan_row.get("world_generation_suggestions") or {})

        raw_template_index = plan_row.get("template_index", plan_row.get("world_instance_index", 0))
        try:
            template_index = int(raw_template_index or 0)
        except Exception:
            template_index = 0
        if should_use_procedural_mine_backend(task_key, args.mine_layout_backend):
            try:
                source_yaml, resolved_template_index, template_count = resolve_source_yaml(
                    env_conf_dir,
                    task_config_name,
                    template_index=template_index,
                )
            except FileNotFoundError:
                source_yaml = resolve_builtin_benchmark_yaml(task_config_name)
                resolved_template_index = 0
                template_count = 1
        else:
            source_yaml, resolved_template_index, template_count = resolve_source_yaml(
                env_conf_dir,
                task_config_name,
                template_index=template_index,
            )
        source_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        if should_use_procedural_mine_backend(task_key, args.mine_layout_backend):
            blueprint_id_override = None
            target_sign_override = None
            target_local_override = None
            occluder_variant_override = {}
            path_obstacle_variant_override = {}
            fixed_spawn_position_override = None
            disable_spawn_jitter = False
            arena_settings_override: Dict[str, Any] = {}
            if isinstance(plan_row.get("world_generation_suggestions"), dict):
                suggestions = plan_row["world_generation_suggestions"]
                blueprint_id_override = suggestions.get("mine_blueprint_id")
                target_sign_override = suggestions.get("mine_target_sign")
                target_local_override = _mine_target_local_override(suggestions)
                occluder_variant_override = _mine_occluder_variant_override(suggestions)
                path_obstacle_variant_override = _mine_path_obstacle_variant_override(suggestions)
                fixed_spawn_position_override = _mine_fixed_spawn_position_override(suggestions)
                disable_spawn_jitter = _parse_bool_suggestion(suggestions.get("mine_disable_spawn_jitter"), False)
                arena_settings_override = _mine_arena_settings_override(suggestions)
            generated_data, applied_changes = build_procedural_mine_layout(
                task_key,
                source_data,
                factor_levels,
                layout_rng,
                suggestions=suggestions if isinstance(plan_row.get("world_generation_suggestions"), dict) else {},
                blueprint_id_override=blueprint_id_override,
                target_sign_override=target_sign_override,
                target_local_override=target_local_override,
                occluder_variant_override=occluder_variant_override,
                path_obstacle_variant_override=path_obstacle_variant_override,
                anchor_mode=args.mine_anchor_mode,
                disable_spawn_jitter=disable_spawn_jitter,
                fixed_spawn_position=fixed_spawn_position_override,
                arena_settings=arena_settings_override,
                path_progress_reward_per_zone=float(args.path_progress_reward_per_zone),
            )
        else:
            generated_data, applied_changes = apply_layout_changes(
                task_key,
                source_data,
                suggestions,
                factor_levels=factor_levels,
                rng=layout_rng,
            )
        generated_data, target_annotation = annotate_target_metadata(task_key, source_data, generated_data, applied_changes)
        plan_suggestions = (
            plan_row.get("world_generation_suggestions")
            if isinstance(plan_row.get("world_generation_suggestions"), dict)
            else {}
        )
        p_goal_pose_protocol = str(plan_suggestions.get("mine_p_goal_pose_protocol") or "").strip()
        if p_goal_pose_protocol:
            generated_data["p_goal_pose_protocol"] = p_goal_pose_protocol
            procedural_layout_for_protocol = (
                generated_data.get("procedural_layout")
                if isinstance(generated_data.get("procedural_layout"), dict)
                else {}
            )
            procedural_layout_for_protocol["p_goal_pose_protocol"] = p_goal_pose_protocol
            generated_data["procedural_layout"] = procedural_layout_for_protocol
        generated_data["goal_pose_hints"] = build_goal_pose_hints(task_key, generated_data)
        procedural_layout = (
            generated_data.get("procedural_layout")
            if isinstance(generated_data.get("procedural_layout"), dict)
            else {}
        )
        path_obstacle_variant = (
            procedural_layout.get("path_obstacle_variant")
            if isinstance(procedural_layout.get("path_obstacle_variant"), dict)
            else {}
        )
        if (
            task_key in PROCEDURAL_MINE_TASKS
            and int(factor_levels.get("P", 0) or 0) > 0
            and int(factor_levels.get("O", 0) or 0) == 0
            and path_obstacle_variant.get("positions")
        ):
            generated_data["goal_pose_candidate_mode"] = "hints_only"
            generated_data["auto_goal_pose_candidate_mode"] = "hints_only"
        applied_changes = dict(applied_changes)
        applied_changes.update(target_annotation)
        if p_goal_pose_protocol:
            applied_changes["p_goal_pose_protocol"] = p_goal_pose_protocol
        applied_changes["goal_pose_hints_count"] = int(len(generated_data.get("goal_pose_hints") or []))
        applied_changes["layout_seed"] = int(plan_row.get("layout_seed", 0) or 0)
        applied_changes["template_index"] = int(resolved_template_index)
        applied_changes["template_count"] = int(template_count)
        applied_changes["source_template_path"] = str(source_yaml.resolve())

        output_path = run_dir / f"{task_config_name}.yaml"
        dump_yaml(output_path, generated_data)
        manifest_rows.append(build_manifest_row(plan_row, output_path, applied_changes))
        print(json.dumps({"task_config_name": task_config_name, "output_yaml": str(output_path), "applied_changes": applied_changes}, ensure_ascii=False))

    manifest = {
        "source_plan_json": str(plan_json.resolve()),
        "source_env_conf_dir": str(env_conf_dir.resolve()),
        "generated_task_group_dir": str(run_dir.resolve()),
        "tasks": manifest_rows,
    }
    (run_dir / "worldgen_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved generated task group to {run_dir}")


if __name__ == "__main__":
    main()
