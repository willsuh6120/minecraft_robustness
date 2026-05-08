import json
import time
import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from minestudio.models import CrossViewRocket, load_cross_view_rocket
from minestudio.models.rocket_two import CFGWrapper
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import PrevActionCallback, VoxelsCallback, load_callbacks_from_config
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    resolve_interaction_task_specs,
)


SEGMENT_MAPPING = {
    "Hunt": 0,
    "Use": 3,
    "Mine": 2,
    "Interact": 3,
    "Craft": 4,
    "Switch": 5,
    "Approach": 6,
    "None": -1,
}

AUTO_GOAL_VOXEL_BOUNDS = [-20, 20, -10, 10, -20, 20]
AUTO_GOAL_MOBS_BOUNDS = [-8, 8, -3, 4, -8, 8]
AUTO_GOAL_BLOCK_TYPES = {
    "mine_emerald": ("emerald",),
    "mine_coal": ("coal",),
}
AUTO_GOAL_MOB_TYPES = {
    "hunt_cow_do_not_touch_sheep": ("cow",),
}
AUTO_GOAL_SUPPORTED_TASKS = frozenset(set(AUTO_GOAL_BLOCK_TYPES) | set(AUTO_GOAL_MOB_TYPES))
DEFAULT_FOV_DEG = 70.0
DEFAULT_EYE_HEIGHT = 1.62


def _minecraft_camera_basis(player_pose: Dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Malmo-rendered Minecraft camera: yaw 0 faces +Z, positive pitch looks down."""
    yaw = math.radians(float(player_pose["yaw"]))
    pitch = math.radians(float(player_pose["pitch"]))
    forward = np.array(
        [
            -math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float32,
    )
    forward /= max(1e-6, float(np.linalg.norm(forward)))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= float(np.linalg.norm(right))
    up = np.cross(right, forward)
    up /= max(1e-6, float(np.linalg.norm(up)))
    return forward, right, up


AUTO_GOAL_DISTANCE_SAMPLES = (2.0, 3.0, 4.0)
AUTO_GOAL_AZIMUTH_DEG_SAMPLES = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
AUTO_GOAL_HEIGHT_SAMPLES = (0.0, 1.0, 2.0)
AUTO_GOAL_CANONICAL_FACE_DISTANCE_SAMPLES = (1.0, 1.5, 2.0)
AUTO_GOAL_CANONICAL_FACE_DISTANCE_SAMPLES_TIGHT_ALCOVE = (0.4, 0.6, 0.8)
AUTO_GOAL_PITCH_OFFSET_SAMPLES = (0.0, -8.0, 8.0)
AUTO_GOAL_MAX_ANCHORS = 3
AUTO_GOAL_MAX_POSE_CANDIDATES = 12
AUTO_GOAL_MAX_POSE_SUCCESSES = 4
AUTO_GOAL_PROBE_RESET_RETRIES = 3
ROLLOUT_RESET_RETRIES = 3
ROLLOUT_RESET_MAX_HORIZONTAL_DRIFT = 6.0
ROLLOUT_RESET_MAX_VERTICAL_DRIFT = 4.0
AUTO_GOAL_POST_TP_SETTLE_STEPS = 3
AUTO_GOAL_MAX_DEBUG_CANDIDATES = 24
AUTO_GOAL_CAPTURE_VISUAL_STREAK_REQUIRED = 2
AUTO_GOAL_CAPTURE_VISUAL_DIFF_MAX = 0.5
AUTO_GOAL_CAPTURE_VISUAL_EXTRA_BUFFER_STEPS = 8
AUTO_GOAL_MIN_CAMERA_DISPLACEMENT = 2.5
AUTO_GOAL_MIN_YAW_DELTA_DEG = 25.0
AUTO_GOAL_MIN_PITCH_DELTA_DEG = 12.0
AUTO_GOAL_MIN_MEAN_ABS_DIFF = 6.0
AUTO_GOAL_MIN_CHANGED_PIXEL_FRAC = 0.01
AUTO_GOAL_MIN_IMAGE_MEAN = 35.0
AUTO_GOAL_MIN_BBOX_MEAN = 35.0
AUTO_GOAL_MIN_BBOX_HEIGHT_PX = 18
AUTO_GOAL_MIN_BBOX_WIDTH_PX = 20
AUTO_GOAL_MIN_BBOX_AREA_FRAC = 0.01
AUTO_GOAL_MIN_MINE_BBOX_AREA_FRAC = 0.05
AUTO_GOAL_MIN_COAL_MASK_STD = 8.0
AUTO_GOAL_MAX_COAL_GREEN_DOMINANT_FRAC = 0.60
AUTO_GOAL_MAX_COAL_GREEN_EXCESS_MEAN = 10.0
AUTO_GOAL_MAX_COAL_DARK80_OCCLUDED = 0.92
AUTO_GOAL_ENABLE_PROJECTION_CONSISTENCY = False
AUTO_GOAL_PROJECTION_CONSISTENCY_MAX_ABS_DY = 2
AUTO_GOAL_PROJECTION_CONSISTENCY_MAX_GAIN = 0.02
AUTO_GOAL_PROJECTION_CONSISTENCY_DY_VALUES = tuple(range(-16, 17))
AUTO_GOAL_PROJECTION_CONSISTENCY_CROP_MARGIN = 24
AUTO_GOAL_REFINE_SHIFT_VALUES = tuple(range(-18, 19, 2))
AUTO_GOAL_REFINE_SCALE_VALUES = (0.88, 0.94, 1.0, 1.06, 1.12, 1.18, 1.24)
AUTO_GOAL_WORLD_REFINE_XZ = (-0.5, -0.25, 0.0, 0.25, 0.5)
AUTO_GOAL_WORLD_REFINE_Y = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5)
AUTO_GOAL_SWEEP_SCRIPT = (
    {"name": "yaw_right_1", "camera": (0.0, 18.0)},
    {"name": "yaw_right_2", "camera": (0.0, 18.0)},
    {"name": "yaw_right_3", "camera": (0.0, 18.0)},
    {"name": "yaw_left_1", "camera": (0.0, -36.0)},
    {"name": "yaw_left_2", "camera": (0.0, -36.0)},
    {"name": "yaw_left_3", "camera": (0.0, -36.0)},
    {"name": "yaw_center_1", "camera": (0.0, 18.0)},
    {"name": "yaw_center_2", "camera": (0.0, 18.0)},
    {"name": "pitch_up", "camera": (-10.0, 0.0)},
    {"name": "pitch_up_right", "camera": (0.0, 18.0)},
    {"name": "pitch_up_left", "camera": (0.0, -36.0)},
    {"name": "pitch_down_reset", "camera": (10.0, 18.0)},
)


class NearbyMobsCallback(MinecraftCallback):
    def __init__(self, mobs_ins = AUTO_GOAL_MOBS_BOUNDS):
        super().__init__()
        self.mobs_ins = list(mobs_ins)

    def before_step(self, sim, action):
        action["mobs"] = list(self.mobs_ins)
        return action


def load_goal_assets(
    goal_image_path: str | Path,
    goal_mask_path: str | Path,
    *,
    obs_size: Tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(obs_size[0]), int(obs_size[1])
    goal_image = Image.open(goal_image_path).convert("RGB")
    goal_image = goal_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    goal_image_np = np.asarray(goal_image, dtype=np.uint8)

    raw_mask = Image.open(goal_mask_path)
    alpha = np.asarray(raw_mask.getchannel("A")) if "A" in raw_mask.getbands() else None
    if alpha is not None and np.any(alpha > 0):
        mask_gray = Image.fromarray(alpha, mode="L")
    else:
        mask_gray = raw_mask.convert("L")
    mask_gray = mask_gray.resize((width, height), resample=Image.Resampling.NEAREST)
    goal_mask_np = (np.asarray(mask_gray) > 0).astype(np.uint8)
    return goal_image_np, goal_mask_np


class GoalConditionCallback(MinecraftCallback):
    def __init__(self, goal_image: np.ndarray, goal_mask: np.ndarray, segment_type: str):
        super().__init__()
        self.update_goal(goal_image=goal_image, goal_mask=goal_mask, segment_type=segment_type)

    def update_goal(self, *, goal_image: np.ndarray, goal_mask: np.ndarray, segment_type: str):
        if segment_type not in SEGMENT_MAPPING:
            valid = ", ".join(sorted(SEGMENT_MAPPING))
            raise ValueError(f"Unknown segment type '{segment_type}'. Expected one of: {valid}.")
        self.goal_image = np.asarray(goal_image, dtype=np.uint8)
        self.goal_mask = (np.asarray(goal_mask) > 0).astype(np.uint8)
        self.segment_type = segment_type
        self.obj_id = SEGMENT_MAPPING[segment_type]

    def _inject(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(obs)
        obs["cross_view"] = {
            "cross_view_image": self.goal_image.copy(),
            "cross_view_obj_id": torch.tensor(self.obj_id, dtype=torch.long),
            "cross_view_obj_mask": torch.tensor(self.goal_mask.copy(), dtype=torch.uint8),
        }
        return obs

    def after_reset(self, sim, obs: Dict[str, Any], info: Dict[str, Any]):
        return self._inject(obs), info

    def after_step(self, sim, obs: Dict[str, Any], reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]):
        return self._inject(obs), reward, terminated, truncated, info


def load_goal_spec(goal_spec_path: str | Path) -> Dict[str, Any]:
    goal_spec_path = Path(goal_spec_path)
    if goal_spec_path.is_dir():
        goal_spec_json = goal_spec_path / "goal_spec.json"
        if goal_spec_json.exists():
            goal_spec_path = goal_spec_json
        else:
            return _build_goal_spec_from_directory(goal_spec_path)
    goal_spec = json.loads(goal_spec_path.read_text(encoding="utf-8"))
    required = ["goal_image_path", "goal_mask_path", "segment_type"]
    missing = [key for key in required if key not in goal_spec]
    if missing:
        raise KeyError(f"Goal spec missing required keys: {missing}")
    spec_dir = goal_spec_path.parent.resolve()
    for key in ("goal_image_path", "goal_mask_path", "goal_mask_overlay_path", "goal_bbox_overlay_path"):
        raw_path = str(goal_spec.get(key) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            resolved = (spec_dir / candidate).resolve()
            if resolved.exists():
                goal_spec[key] = str(resolved)
            continue
        if candidate.exists():
            continue
        sibling = spec_dir / candidate.name
        if sibling.exists():
            goal_spec[key] = str(sibling.resolve())
    return goal_spec


def _infer_segment_type_for_goal_dir(goal_dir: Path, metadata: Dict[str, Any]) -> str:
    segment_type = str(metadata.get("segment_type") or "")
    if segment_type:
        return segment_type
    goal_metadata = metadata.get("goal_metadata")
    if isinstance(goal_metadata, dict):
        segment_type = str(goal_metadata.get("segment_type") or "")
        if segment_type:
            return segment_type
        task_key = str(goal_metadata.get("task_key") or "")
        if task_key:
            task_specs = resolve_interaction_task_specs([task_key], env_source="rocket2_official")
            if task_specs and getattr(task_specs[0], "subtasks", None):
                return str(task_specs[0].subtasks[0].interaction_type)
    for ancestor in (goal_dir, *goal_dir.parents):
        name = ancestor.name.strip()
        if not name:
            continue
        try:
            task_specs = resolve_interaction_task_specs([name], env_source="rocket2_official")
        except Exception:
            continue
        if task_specs and getattr(task_specs[0], "subtasks", None):
            return str(task_specs[0].subtasks[0].interaction_type)
    raise KeyError(
        f"Could not infer segment_type for goal directory {goal_dir}. "
        "Pass a goal_spec.json or use a directory containing metadata with task_key."
    )


def _build_goal_spec_from_directory(goal_dir: Path) -> Dict[str, Any]:
    goal_image_path = goal_dir / "goal_image.png"
    goal_mask_path = goal_dir / "goal_mask.png"
    if not goal_image_path.exists() or not goal_mask_path.exists():
        raise FileNotFoundError(
            f"Goal directory {goal_dir} must contain goal_image.png and goal_mask.png"
        )
    metadata: Dict[str, Any] = {}
    metadata_path = goal_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    goal_image = Image.open(goal_image_path).convert("RGB")
    segment_type = _infer_segment_type_for_goal_dir(goal_dir, metadata)
    goal_spec: Dict[str, Any] = {
        "goal_image_path": str(goal_image_path.resolve()),
        "goal_mask_path": str(goal_mask_path.resolve()),
        "segment_type": segment_type,
        "cfg_coef": float(metadata.get("cfg_coef", 1.5)),
        "obs_size": [int(goal_image.width), int(goal_image.height)],
    }
    if "goal_metadata" in metadata and isinstance(metadata["goal_metadata"], dict):
        trimmed_goal_metadata = {
            str(k): v
            for k, v in metadata["goal_metadata"].items()
            if str(k) != "occupied_voxel_keys"
        }
        goal_spec.update(trimmed_goal_metadata)
    goal_mask_overlay_path = goal_dir / "goal_mask_overlay.png"
    goal_bbox_overlay_path = goal_dir / "goal_bbox_overlay.png"
    if goal_mask_overlay_path.exists():
        goal_spec["goal_mask_overlay_path"] = str(goal_mask_overlay_path.resolve())
    if goal_bbox_overlay_path.exists():
        goal_spec["goal_bbox_overlay_path"] = str(goal_bbox_overlay_path.resolve())
    return goal_spec


def _sanitize_debug_value(data):
    if isinstance(data, dict):
        return {str(k): _sanitize_debug_value(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_debug_value(v) for v in data]
    if isinstance(data, tuple):
        return [_sanitize_debug_value(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    return data


def _task_key(task_spec: Any) -> str:
    return str(getattr(task_spec, "task_key", "") or getattr(task_spec, "task_config_name", ""))


def _normalize_name(value: Any) -> str:
    return str(value or "").lower().replace("minecraft:", "")


def _distance3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _round_half_up(value: float) -> int:
    if float(value) >= 0.0:
        return int(math.floor(float(value) + 0.5))
    return int(math.ceil(float(value) - 0.5))


def _snap_support_height_offset(height_offset: float) -> float:
    return float(_round_half_up(float(height_offset)))


def _support_block_for_pose(pose: Dict[str, float]) -> Dict[str, Any]:
    feet_y = float(pose["y"])
    return {
        "x": int(math.floor(float(pose["x"]))),
        "y": int(math.floor(feet_y)) - 1,
        "z": int(math.floor(float(pose["z"]))),
        "feet_y": float(feet_y),
        "block_type": "minecraft:barrier",
    }


def _ensure_support_block(env, support_block: Dict[str, Any] | None) -> None:
    if not isinstance(support_block, dict):
        return
    if support_block.get("enabled") is False:
        return
    x = int(support_block["x"])
    y = int(support_block["y"])
    z = int(support_block["z"])
    block_type = str(support_block.get("block_type", "minecraft:barrier"))
    env.env.execute_cmd(f"/setblock {x} {y} {z} {block_type} keep")


def _voxel_world_center_from_player_pose(
    player_pose: Dict[str, float],
    raw_offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    # Malmo grid observations are centered on BlockPos(player.posX, posY, posZ),
    # not on the player's fractional coordinates. This matters at *.5 and for
    # negative coordinates, where using player_pose + offset shifts targets.
    return (
        math.floor(float(player_pose["x"])) + float(raw_offset[0]) + 0.5,
        math.floor(float(player_pose["y"])) + float(raw_offset[1]) + 0.5,
        math.floor(float(player_pose["z"])) + float(raw_offset[2]) + 0.5,
    )


def _voxel_key_from_world_center(
    player_pose: Dict[str, float],
    world_center: tuple[float, float, float],
) -> tuple[int, int, int]:
    return (
        int(round(float(world_center[0]) - math.floor(float(player_pose["x"])) - 0.5)),
        int(round(float(world_center[1]) - math.floor(float(player_pose["y"])) - 0.5)),
        int(round(float(world_center[2]) - math.floor(float(player_pose["z"])) - 0.5)),
    )


def _angle_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return min(delta, 360.0 - delta)


def _angular_error_deg(current: float, target: float) -> float:
    delta = (float(current) - float(target) + 180.0) % 360.0 - 180.0
    return abs(float(delta))


def _extract_mobs(info: Dict[str, Any]) -> list[Dict[str, Any]]:
    raw_mobs = info.get("mobs")
    if isinstance(raw_mobs, list):
        return [item for item in raw_mobs if isinstance(item, dict)]
    if isinstance(raw_mobs, dict):
        for value in raw_mobs.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _player_pose(info: Dict[str, Any]) -> Dict[str, float]:
    player_pos = info.get("player_pos") or {}
    if not isinstance(player_pos, dict):
        raise RuntimeError("player_pos missing from info; cannot synthesize auto goal")
    return {
        "x": float(player_pos.get("x", 0.0)),
        "y": float(player_pos.get("y", 0.0)),
        "z": float(player_pos.get("z", 0.0)),
        "pitch": float(player_pos.get("pitch", 0.0)),
        "yaw": float(player_pos.get("yaw", 0.0)),
    }


def _safe_player_pose(info: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(info, dict):
        return None
    try:
        return _player_pose(info)
    except Exception:
        return None


def _inventory_totals_from_info(info: Optional[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    if not isinstance(info, dict):
        return totals
    inventory = info.get("inventory", {}) or {}
    values = inventory.values() if isinstance(inventory, dict) else inventory
    for slot in values:
        if not isinstance(slot, dict):
            continue
        item_type = _normalize_name(slot.get("type", ""))
        quantity_raw = slot.get("quantity", 0)
        if isinstance(quantity_raw, np.ndarray):
            quantity = int(quantity_raw.item())
        else:
            try:
                quantity = int(quantity_raw)
            except Exception:
                quantity = 0
        if item_type and item_type != "air" and quantity > 0:
            totals[item_type] = int(totals.get(item_type, 0)) + int(quantity)
    return totals


def _mainhand_type_from_info(info: Optional[Dict[str, Any]]) -> str:
    if not isinstance(info, dict):
        return ""
    equipped_items = info.get("equipped_items") or {}
    if not isinstance(equipped_items, dict):
        return ""
    mainhand = equipped_items.get("mainhand") or {}
    if not isinstance(mainhand, dict):
        return ""
    return _normalize_name(mainhand.get("type", ""))


def _expected_spawn_positions(task_config: Dict[str, Any]) -> list[Dict[str, float]]:
    positions: list[Dict[str, float]] = []
    raw_positions = task_config.get("spawn_positions") or []
    if not isinstance(raw_positions, list):
        return positions
    for entry in raw_positions:
        if not isinstance(entry, dict):
            continue
        raw_position = entry.get("position")
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 3:
            continue
        try:
            positions.append(
                {
                    "x": float(raw_position[0]),
                    "y": float(raw_position[1]),
                    "z": float(raw_position[2]),
                    "yaw": float(entry.get("yaw", 0.0) or 0.0),
                    "pitch": float(entry.get("pitch", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return positions


def _expected_mainhand_from_task_config(task_config: Dict[str, Any]) -> str:
    init_inventory = task_config.get("init_inventory") or []
    if not isinstance(init_inventory, list):
        return ""
    for item in init_inventory:
        if not isinstance(item, dict):
            continue
        raw_slot = item.get("slot")
        slot_name = str(raw_slot).strip().lower()
        slot_index = None
        try:
            slot_index = int(raw_slot)
        except Exception:
            slot_index = None
        if slot_index == 0 or slot_name in {"mainhand", "weapon.mainhand"}:
            return _normalize_name(item.get("type", ""))
    return ""


def _best_spawn_match(
    player_pose: Optional[Dict[str, float]],
    expected_spawns: list[Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    if player_pose is None or not expected_spawns:
        return None
    best_match = None
    for index, expected in enumerate(expected_spawns):
        horizontal_drift = math.sqrt(
            (float(player_pose["x"]) - float(expected["x"])) ** 2
            + (float(player_pose["z"]) - float(expected["z"])) ** 2
        )
        vertical_drift = abs(float(player_pose["y"]) - float(expected["y"]))
        candidate = {
            "expected_spawn_index": int(index),
            "expected_spawn": {
                "x": float(expected["x"]),
                "y": float(expected["y"]),
                "z": float(expected["z"]),
                "yaw": float(expected.get("yaw", 0.0)),
                "pitch": float(expected.get("pitch", 0.0)),
            },
            "horizontal_drift": float(horizontal_drift),
            "vertical_drift": float(vertical_drift),
        }
        if best_match is None or (
            float(candidate["horizontal_drift"]),
            float(candidate["vertical_drift"]),
        ) < (
            float(best_match["horizontal_drift"]),
            float(best_match["vertical_drift"]),
        ):
            best_match = candidate
    return best_match


def _rollout_reset_invariant_report(task_config: Dict[str, Any], info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    player_pose = _safe_player_pose(info)
    expected_spawns = _expected_spawn_positions(task_config)
    expected_mainhand = _expected_mainhand_from_task_config(task_config)
    inventory_totals = _inventory_totals_from_info(info)
    mainhand_type = _mainhand_type_from_info(info)
    spawn_match = _best_spawn_match(player_pose, expected_spawns)
    failures: list[str] = []

    if player_pose is None:
        failures.append("player_pos_missing")
    if spawn_match is not None:
        if float(spawn_match["horizontal_drift"]) > float(ROLLOUT_RESET_MAX_HORIZONTAL_DRIFT):
            failures.append(
                "spawn_horizontal_drift="
                f"{float(spawn_match['horizontal_drift']):.3f}>{float(ROLLOUT_RESET_MAX_HORIZONTAL_DRIFT):.3f}"
            )
        if float(spawn_match["vertical_drift"]) > float(ROLLOUT_RESET_MAX_VERTICAL_DRIFT):
            failures.append(
                "spawn_vertical_drift="
                f"{float(spawn_match['vertical_drift']):.3f}>{float(ROLLOUT_RESET_MAX_VERTICAL_DRIFT):.3f}"
            )
    if expected_mainhand:
        expected_count = int(inventory_totals.get(expected_mainhand, 0))
        if mainhand_type != expected_mainhand and expected_count <= 0:
            failures.append(
                f"missing_expected_mainhand={expected_mainhand}"
                f"(mainhand={mainhand_type or 'air'}, inventory_count={expected_count})"
            )

    return {
        "ok": len(failures) == 0,
        "player_pose": player_pose,
        "expected_spawn_positions": expected_spawns,
        "spawn_match": spawn_match,
        "expected_mainhand": expected_mainhand,
        "mainhand_type": mainhand_type,
        "inventory_totals": inventory_totals,
        "failures": failures,
    }


def _format_rollout_reset_failure(stage: str, report: Dict[str, Any]) -> str:
    failures = report.get("failures") or []
    if not failures:
        return f"Rollout reset invariant failed at {stage}"
    return f"Rollout reset invariant failed at {stage}: {'; '.join(str(item) for item in failures)}"


def _pose_error(candidate_pose: Dict[str, float], expected_pose: Dict[str, float]) -> Dict[str, float]:
    return {
        "x": abs(float(candidate_pose["x"]) - float(expected_pose["x"])),
        "y": abs(float(candidate_pose["y"]) - float(expected_pose["y"])),
        "z": abs(float(candidate_pose["z"]) - float(expected_pose["z"])),
        "yaw": _angular_error_deg(float(candidate_pose["yaw"]), float(expected_pose["yaw"])),
        "pitch": _angular_error_deg(float(candidate_pose["pitch"]), float(expected_pose["pitch"])),
    }


def _pose_matches_expected(
    candidate_pose: Dict[str, float],
    expected_pose: Dict[str, float],
    *,
    pos_tol: float = 1e-3,
    angle_tol_deg: float = 0.05,
) -> bool:
    error = _pose_error(candidate_pose, expected_pose)
    return bool(
        float(error["x"]) <= float(pos_tol)
        and float(error["y"]) <= float(pos_tol)
        and float(error["z"]) <= float(pos_tol)
        and float(error["yaw"]) <= float(angle_tol_deg)
        and float(error["pitch"]) <= float(angle_tol_deg)
    )


def _teleport_to_pose(env, pose: Dict[str, float]) -> None:
    env.env.execute_cmd(
        "/tp @a "
        f"{float(pose['x']):.3f} {float(pose['y']):.3f} {float(pose['z']):.3f} "
        f"{float(pose['yaw']):.3f} {float(pose['pitch']):.3f}"
    )


def _capture_stable_frame(
    env,
    expected_pose: Dict[str, float],
    *,
    min_wait_steps: int,
    max_wait_steps: int,
    match_streak_required: int = 2,
    render_buffer_steps: int = 4,
):
    obs = None
    info = None
    terminated = False
    truncated = False
    streak = 0
    diagnostics: Dict[str, Any] = {
        "expected_pose": _sanitize_debug_value(expected_pose),
        "min_wait_steps": int(min_wait_steps),
        "max_wait_steps": int(max_wait_steps),
        "match_streak_required": int(match_streak_required),
        "render_buffer_steps": int(render_buffer_steps),
        "visual_streak_required": int(AUTO_GOAL_CAPTURE_VISUAL_STREAK_REQUIRED),
        "visual_diff_max": float(AUTO_GOAL_CAPTURE_VISUAL_DIFF_MAX),
        "frames_stepped": 0,
        "stabilized": False,
        "buffer_frames_applied": 0,
        "visual_match_streak": 0,
    }

    for step_idx in range(max(1, int(max_wait_steps))):
        obs, _, terminated, truncated, info = env.step(env.noop_action())
        diagnostics["frames_stepped"] = int(step_idx + 1)
        if info is not None:
            pose = _player_pose(info)
            diagnostics["last_pose"] = _sanitize_debug_value(pose)
            diagnostics["last_pose_error"] = _sanitize_debug_value(_pose_error(pose, expected_pose))
            if _pose_matches_expected(pose, expected_pose):
                streak += 1
            else:
                streak = 0
            diagnostics["match_streak"] = int(streak)
        if terminated or truncated:
            break
        if int(step_idx + 1) >= int(min_wait_steps) and int(streak) >= int(match_streak_required):
            diagnostics["stabilized"] = True
            prev_pov = None
            if info is not None:
                raw_pov = info.get("pov")
                if isinstance(raw_pov, np.ndarray):
                    prev_pov = np.asarray(raw_pov, dtype=np.uint8).copy()
            visual_streak = 0
            total_buffer_steps = max(0, int(render_buffer_steps)) + max(0, int(AUTO_GOAL_CAPTURE_VISUAL_EXTRA_BUFFER_STEPS))
            for buffer_idx in range(total_buffer_steps):
                obs, _, terminated, truncated, info = env.step(env.noop_action())
                diagnostics["buffer_frames_applied"] = int(diagnostics["buffer_frames_applied"]) + 1
                diagnostics["frames_stepped"] = int(diagnostics["frames_stepped"]) + 1
                if info is not None:
                    pose = _player_pose(info)
                    diagnostics["last_pose"] = _sanitize_debug_value(pose)
                    diagnostics["last_pose_error"] = _sanitize_debug_value(_pose_error(pose, expected_pose))
                    raw_pov = info.get("pov")
                    current_pov = np.asarray(raw_pov, dtype=np.uint8) if isinstance(raw_pov, np.ndarray) else None
                    if current_pov is not None and prev_pov is not None and current_pov.shape == prev_pov.shape:
                        frame_mean_abs_diff = float(
                            np.mean(np.abs(current_pov.astype(np.int16) - prev_pov.astype(np.int16)))
                        )
                        diagnostics["last_frame_mean_abs_diff"] = round(frame_mean_abs_diff, 6)
                        if frame_mean_abs_diff <= float(AUTO_GOAL_CAPTURE_VISUAL_DIFF_MAX):
                            visual_streak += 1
                        else:
                            visual_streak = 0
                        diagnostics["visual_match_streak"] = int(visual_streak)
                    else:
                        visual_streak = 0
                        diagnostics["visual_match_streak"] = 0
                    if current_pov is not None:
                        prev_pov = current_pov.copy()
                if terminated or truncated:
                    break
                if (
                    int(buffer_idx + 1) >= int(render_buffer_steps)
                    and int(visual_streak) >= int(AUTO_GOAL_CAPTURE_VISUAL_STREAK_REQUIRED)
                ):
                    break
            break

    return obs, info, terminated, truncated, diagnostics


def _camera_basis_variants(player_pose: Dict[str, float]) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    yaw = math.radians(float(player_pose["yaw"]))
    pitch = math.radians(float(player_pose["pitch"]))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    variants = []
    for x_sign, z_sign, pitch_sign in (
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
    ):
        forward = np.array(
            [
                x_sign * math.sin(yaw) * math.cos(pitch),
                pitch_sign * math.sin(pitch),
                z_sign * math.cos(yaw) * math.cos(pitch),
            ],
            dtype=np.float32,
        )
        forward /= max(1e-6, float(np.linalg.norm(forward)))
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-6:
            continue
        right /= float(np.linalg.norm(right))
        up = np.cross(right, forward)
        up /= max(1e-6, float(np.linalg.norm(up)))
        variants.append((f"x{x_sign:+.0f}_z{z_sign:+.0f}_p{pitch_sign:+.0f}", forward, right, up))
    return variants


def _project_world_point_with_basis(
    origin: np.ndarray,
    target: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> Optional[Dict[str, float]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    delta = target - origin
    cam_z = float(np.dot(delta, forward))
    if cam_z <= 1e-4:
        return None
    cam_x = float(np.dot(delta, right))
    cam_y = float(np.dot(delta, up))
    aspect = float(width) / max(1.0, float(height))
    fov_y = math.radians(float(fov_deg))
    fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect)
    ndc_x = cam_x / max(1e-6, cam_z * math.tan(fov_x / 2.0))
    ndc_y = cam_y / max(1e-6, cam_z * math.tan(fov_y / 2.0))
    x = (ndc_x + 1.0) * 0.5 * float(width)
    y = (1.0 - ndc_y) * 0.5 * float(height)
    return {
        "x": float(x),
        "y": float(y),
        "depth": float(cam_z),
        "ndc_x": float(ndc_x),
        "ndc_y": float(ndc_y),
    }


def _project_world_point(
    player_pose: Dict[str, float],
    target_world: tuple[float, float, float],
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    fov_deg: float = DEFAULT_FOV_DEG,
    eye_height: float = DEFAULT_EYE_HEIGHT,
) -> Optional[Dict[str, float]]:
    origin = np.array(
        [
            float(player_pose["x"]),
            float(player_pose["y"]) + float(eye_height),
            float(player_pose["z"]),
        ],
        dtype=np.float32,
    )
    target = np.array(target_world, dtype=np.float32)
    best_projection = None
    best_penalty = None
    width = int(image_shape[1])
    height = int(image_shape[0])
    for basis_name, forward, right, up in _camera_basis_variants(player_pose):
        projection = _project_world_point_with_basis(
            origin,
            target,
            image_shape,
            forward=forward,
            right=right,
            up=up,
            fov_deg=fov_deg,
        )
        if projection is None:
            continue
        penalty = (
            abs(float(projection["x"]) - (width / 2.0))
            + abs(float(projection["y"]) - (height / 2.0))
            + 0.1 * float(projection["depth"])
        )
        if best_projection is None or penalty < float(best_penalty):
            projection["basis_name"] = basis_name
            best_projection = projection
            best_penalty = penalty
    return best_projection


def _project_bbox(
    player_pose: Dict[str, float],
    corners: list[tuple[float, float, float]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> Optional[tuple[tuple[int, int, int, int], Dict[str, float]]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    origin = np.array(
        [
            float(player_pose["x"]),
            float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
            float(player_pose["z"]),
        ],
        dtype=np.float32,
    )
    targets = [np.array(corner, dtype=np.float32) for corner in corners]
    best_result = None
    best_score = None
    for basis_name, forward, right, up in _camera_basis_variants(player_pose):
        projections = []
        for target in targets:
            projection = _project_world_point_with_basis(
                origin,
                target,
                image_shape,
                forward=forward,
                right=right,
                up=up,
            )
            if projection is not None:
                projections.append(projection)
        if not projections:
            continue
        xs = [item["x"] for item in projections]
        ys = [item["y"] for item in projections]
        x0 = max(0, min(width - 1, int(math.floor(min(xs)))))
        y0 = max(0, min(height - 1, int(math.floor(min(ys)))))
        x1 = max(0, min(width - 1, int(math.ceil(max(xs)))))
        y1 = max(0, min(height - 1, int(math.ceil(max(ys)))))
        if x1 <= x0 or y1 <= y0:
            continue
        center = {
            "x": float(sum(xs) / len(xs)),
            "y": float(sum(ys) / len(ys)),
            "depth": float(sum(item["depth"] for item in projections) / len(projections)),
            "basis_name": basis_name,
        }
        area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
        score = float(area) - 0.2 * _score_candidate(center, image_shape)
        if best_result is None or score > float(best_score):
            best_result = ((x0, y0, x1, y1), center)
            best_score = score
    return best_result


def _get_basis_by_name(player_pose: Dict[str, float], basis_name: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if str(basis_name).lower() in {"minecraft", "minecraft_1_16_5", "canonical"}:
        return _minecraft_camera_basis(player_pose)
    for name, forward, right, up in _camera_basis_variants(player_pose):
        if name == basis_name:
            return forward, right, up
    return None


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    src_shape: tuple[int, int, int] | tuple[int, int],
    dst_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    src_h, src_w = int(src_shape[0]), int(src_shape[1])
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(dst_w - 1, int(round(x0 * dst_w / src_w)))),
        max(0, min(dst_h - 1, int(round(y0 * dst_h / src_h)))),
        max(0, min(dst_w - 1, int(round(x1 * dst_w / src_w)))),
        max(0, min(dst_h - 1, int(round(y1 * dst_h / src_h)))),
    )


def _make_bbox_mask(obs_shape: tuple[int, int, int] | tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = int(obs_shape[0]), int(obs_shape[1])
    x0, y0, x1, y1 = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def _resize_binary_mask(mask: np.ndarray, dst_shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])
    resized = cv2.resize(mask.astype(np.uint8), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _mask_prompt_point(mask: np.ndarray) -> tuple[float, float]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return (0.0, 0.0)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return _mask_center(mask_u8)
    ys, xs = np.where(distance >= max_distance - 1e-6)
    if len(xs) == 0 or len(ys) == 0:
        return _mask_center(mask_u8)
    return (float(xs.mean()), float(ys.mean()))


def _block_corners(center: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    corners = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                corners.append((cx + dx, cy + dy, cz + dz))
    return corners


def _is_non_occluding_goal_block(voxel_type: str) -> bool:
    normalized = _normalize_name(voxel_type)
    if not normalized:
        return True
    if normalized in {"air", "cave_air", "void_air"}:
        return True
    non_occluding_substrings = (
        "glass",
        "vine",
    )
    return any(token in normalized for token in non_occluding_substrings)


def _block_face_definitions(center: tuple[float, float, float]) -> list[tuple[np.ndarray, tuple[int, int, int], list[tuple[float, float, float]]]]:
    cx, cy, cz = center
    return [
        (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (1, 0, 0),
            [(cx + 0.5, cy - 0.5, cz - 0.5), (cx + 0.5, cy - 0.5, cz + 0.5), (cx + 0.5, cy + 0.5, cz + 0.5), (cx + 0.5, cy + 0.5, cz - 0.5)],
        ),
        (
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            (-1, 0, 0),
            [(cx - 0.5, cy - 0.5, cz + 0.5), (cx - 0.5, cy - 0.5, cz - 0.5), (cx - 0.5, cy + 0.5, cz - 0.5), (cx - 0.5, cy + 0.5, cz + 0.5)],
        ),
        (
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (0, 1, 0),
            [(cx - 0.5, cy + 0.5, cz - 0.5), (cx + 0.5, cy + 0.5, cz - 0.5), (cx + 0.5, cy + 0.5, cz + 0.5), (cx - 0.5, cy + 0.5, cz + 0.5)],
        ),
        (
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            (0, -1, 0),
            [(cx - 0.5, cy - 0.5, cz + 0.5), (cx + 0.5, cy - 0.5, cz + 0.5), (cx + 0.5, cy - 0.5, cz - 0.5), (cx - 0.5, cy - 0.5, cz - 0.5)],
        ),
        (
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            (0, 0, 1),
            [(cx - 0.5, cy - 0.5, cz + 0.5), (cx - 0.5, cy + 0.5, cz + 0.5), (cx + 0.5, cy + 0.5, cz + 0.5), (cx + 0.5, cy - 0.5, cz + 0.5)],
        ),
        (
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            (0, 0, -1),
            [(cx + 0.5, cy - 0.5, cz - 0.5), (cx + 0.5, cy + 0.5, cz - 0.5), (cx - 0.5, cy + 0.5, cz - 0.5), (cx - 0.5, cy - 0.5, cz - 0.5)],
        ),
    ]


FACE_LABELS = {
    0: "+x",
    1: "-x",
    2: "+y",
    3: "-y",
    4: "+z",
    5: "-z",
}


def _visible_face_projection_entries(
    *,
    player_pose: Dict[str, float],
    target_center: tuple[float, float, float],
    image_shape,
    basis_name: str,
    voxel_key: tuple[int, int, int],
    occupied_voxels: set[tuple[int, int, int]],
) -> list[Dict[str, Any]]:
    basis = _get_basis_by_name(player_pose, str(basis_name))
    if basis is None:
        return []
    forward, right, up = basis
    height, width = int(image_shape[0]), int(image_shape[1])
    origin = np.array(
        [
            float(player_pose["x"]),
            float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
            float(player_pose["z"]),
        ],
        dtype=np.float32,
    )
    entries: list[Dict[str, Any]] = []
    for face_index, (normal, neighbor_offset, vertices) in enumerate(_block_face_definitions(target_center)):
        neighbor_key = (
            int(voxel_key[0] + neighbor_offset[0]),
            int(voxel_key[1] + neighbor_offset[1]),
            int(voxel_key[2] + neighbor_offset[2]),
        )
        if neighbor_key in occupied_voxels:
            continue
        face_center = np.mean(np.asarray(vertices, dtype=np.float32), axis=0)
        to_face = face_center - origin
        if float(np.dot(normal, to_face)) >= -1e-4:
            continue
        projected_pts = []
        valid = True
        for vertex in vertices:
            projection = _project_world_point_with_basis(
                origin,
                np.asarray(vertex, dtype=np.float32),
                image_shape,
                forward=forward,
                right=right,
                up=up,
            )
            if projection is None:
                valid = False
                break
            projected_pts.append(
                [
                    max(0, min(width - 1, int(round(projection["x"])))),
                    max(0, min(height - 1, int(round(projection["y"])))),
                ]
            )
        if not valid or len(projected_pts) < 3:
            continue
        face_center_projection = _project_world_point_with_basis(
            origin,
            np.asarray(face_center, dtype=np.float32),
            image_shape,
            forward=forward,
            right=right,
            up=up,
        )
        polygon = np.asarray(projected_pts, dtype=np.int32)
        polygon_moments = cv2.moments(polygon.astype(np.float32))
        if abs(float(polygon_moments.get("m00", 0.0))) > 1e-6:
            polygon_center = {
                "x": float(polygon_moments["m10"] / polygon_moments["m00"]),
                "y": float(polygon_moments["m01"] / polygon_moments["m00"]),
                "basis_name": str(basis_name),
            }
        else:
            polygon_center = {
                "x": float(np.mean(polygon[:, 0])),
                "y": float(np.mean(polygon[:, 1])),
                "basis_name": str(basis_name),
            }
        entries.append(
            {
                "face_index": int(face_index),
                "face_label": str(FACE_LABELS.get(int(face_index), str(face_index))),
                "polygon": polygon,
                "area": float(abs(cv2.contourArea(polygon.astype(np.float32)))),
                "polygon_center": polygon_center,
                "center_projection": (
                    None
                    if face_center_projection is None
                    else {
                        "x": float(face_center_projection["x"]),
                        "y": float(face_center_projection["y"]),
                        "depth": float(face_center_projection["depth"]),
                        "basis_name": str(basis_name),
                    }
                ),
            }
        )
    return entries


def _visible_face_metadata(entries: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    metadata: list[Dict[str, Any]] = []
    for entry in entries:
        polygon = entry.get("polygon")
        polygon_vertices = []
        if isinstance(polygon, np.ndarray) and polygon.ndim == 2 and polygon.shape[1] == 2:
            polygon_vertices = [[int(point[0]), int(point[1])] for point in polygon.tolist()]
        metadata.append(
            {
                "face_index": int(entry.get("face_index", -1)),
                "face_label": str(entry.get("face_label", "")),
                "area": round(float(entry.get("area", 0.0)), 4),
                "polygon_vertices": polygon_vertices,
                "polygon_center": _sanitize_debug_value(entry.get("polygon_center")),
                "center_projection": _sanitize_debug_value(entry.get("center_projection")),
            }
        )
    return metadata


def _project_block_face_mask(
    player_pose: Dict[str, float],
    center: tuple[float, float, float],
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    basis_name: str,
    voxel_key: tuple[int, int, int],
    occupied_voxels: set[tuple[int, int, int]],
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    basis = _get_basis_by_name(player_pose, basis_name)
    if basis is None:
        return mask
    forward, right, up = basis
    origin = np.array(
        [
            float(player_pose["x"]),
            float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
            float(player_pose["z"]),
        ],
        dtype=np.float32,
    )
    for normal, neighbor_offset, vertices in _block_face_definitions(center):
        neighbor_key = (
            int(voxel_key[0] + neighbor_offset[0]),
            int(voxel_key[1] + neighbor_offset[1]),
            int(voxel_key[2] + neighbor_offset[2]),
        )
        if neighbor_key in occupied_voxels:
            continue
        face_center = np.mean(np.asarray(vertices, dtype=np.float32), axis=0)
        to_face = face_center - origin
        if float(np.dot(normal, to_face)) >= -1e-4:
            continue
        projected_pts = []
        valid = True
        for vertex in vertices:
            projection = _project_world_point_with_basis(
                origin,
                np.asarray(vertex, dtype=np.float32),
                image_shape,
                forward=forward,
                right=right,
                up=up,
            )
            if projection is None:
                valid = False
                break
            projected_pts.append(
                [
                    max(0, min(width - 1, int(round(projection["x"])))),
                    max(0, min(height - 1, int(round(projection["y"])))),
                ]
            )
        if not valid or len(projected_pts) < 3:
            continue
        polygon = np.asarray(projected_pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask


def _block_mask_signal(task_key: str, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pixels = image[mask > 0]
    if pixels.size == 0:
        return {
            "selection_score": float("-inf"),
            "signal_mean": 0.0,
            "signal_sum": 0.0,
            "signal_frac": 0.0,
            "mask_area": 0.0,
        }
    pixels = pixels.astype(np.float32)
    red = pixels[:, 0]
    green = pixels[:, 1]
    blue = pixels[:, 2]
    mask_area = float(len(pixels))
    if task_key == "mine_emerald":
        signal = np.maximum(0.0, green - np.maximum(red, blue))
        strong = signal >= 18.0
        signal_sum = float(signal.sum())
        signal_mean = float(signal.mean())
        signal_frac = float(np.count_nonzero(strong) / max(1, len(signal)))
        selection_score = (
            0.20 * signal_sum
            + 250.0 * signal_frac
            + 20.0 * signal_mean
            - 0.03 * mask_area
        )
        return {
            "selection_score": float(selection_score),
            "signal_mean": signal_mean,
            "signal_sum": signal_sum,
            "signal_frac": signal_frac,
            "mask_area": mask_area,
        }
    return {
        "selection_score": float(-0.03 * mask_area),
        "signal_mean": 0.0,
        "signal_sum": 0.0,
        "signal_frac": 0.0,
        "mask_area": mask_area,
    }


def _transform_binary_mask(mask: np.ndarray, scale: float, dx: int, dy: int) -> np.ndarray:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    transform = np.array(
        [
            [float(scale), 0.0, (1.0 - float(scale)) * cx + float(dx)],
            [0.0, float(scale), (1.0 - float(scale)) * cy + float(dy)],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0).astype(np.uint8)


def _block_refinement_score(task_key: str, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    signal = _block_mask_signal(task_key, image, mask)
    if not np.any(mask > 0):
        signal["refine_score"] = float("-inf")
        signal["ring_signal_sum"] = 0.0
        signal["ring_signal_frac"] = 0.0
        return signal
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1)
    ring = np.logical_and(dilated > 0, mask <= 0).astype(np.uint8)
    ring_signal = _block_mask_signal(task_key, image, ring)
    refine_score = (
        1.0 * float(signal["signal_sum"])
        + 1200.0 * float(signal["signal_frac"])
        - 0.85 * float(ring_signal["signal_sum"])
        - 900.0 * float(ring_signal["signal_frac"])
        - 0.02 * float(signal["mask_area"])
    )
    signal["refine_score"] = float(refine_score)
    signal["ring_signal_sum"] = float(ring_signal["signal_sum"])
    signal["ring_signal_frac"] = float(ring_signal["signal_frac"])
    return signal


def _refine_block_mask_image(task_key: str, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    base_metrics = _block_refinement_score(task_key, image, mask)
    best_mask = mask
    best_metrics = dict(base_metrics)
    best_params = {"dx": 0, "dy": 0, "scale": 1.0}
    for scale in AUTO_GOAL_REFINE_SCALE_VALUES:
        for dy in AUTO_GOAL_REFINE_SHIFT_VALUES:
            for dx in AUTO_GOAL_REFINE_SHIFT_VALUES:
                if scale == 1.0 and dx == 0 and dy == 0:
                    continue
                candidate_mask = _transform_binary_mask(mask, scale=scale, dx=dx, dy=dy)
                metrics = _block_refinement_score(task_key, image, candidate_mask)
                if float(metrics["refine_score"]) > float(best_metrics["refine_score"]):
                    best_mask = candidate_mask
                    best_metrics = dict(metrics)
                    best_params = {"dx": int(dx), "dy": int(dy), "scale": float(scale)}
    best_metrics.update(
        {
            "best_dx": int(best_params["dx"]),
            "best_dy": int(best_params["dy"]),
            "best_scale": float(best_params["scale"]),
            "base_refine_score": float(base_metrics["refine_score"]),
        }
    )
    return best_mask, best_metrics


def _refine_block_world_projection(
    task_key: str,
    player_pose: Dict[str, float],
    image: np.ndarray,
    center: tuple[float, float, float],
    *,
    basis_name: str,
    voxel_key: tuple[int, int, int],
    occupied_voxels: set[tuple[int, int, int]],
) -> tuple[tuple[float, float, float], np.ndarray, Dict[str, float]]:
    raw_mask = _project_block_face_mask(
        player_pose,
        center,
        image.shape,
        basis_name=basis_name,
        voxel_key=voxel_key,
        occupied_voxels=occupied_voxels,
    )
    best_center = center
    best_mask = raw_mask
    best_metrics = _block_refinement_score(task_key, image, raw_mask)
    best_offsets = (0.0, 0.0, 0.0)
    for dy in AUTO_GOAL_WORLD_REFINE_Y:
        for dx in AUTO_GOAL_WORLD_REFINE_XZ:
            for dz in AUTO_GOAL_WORLD_REFINE_XZ:
                if dx == 0.0 and dy == 0.0 and dz == 0.0:
                    continue
                candidate_center = (
                    float(center[0]) + float(dx),
                    float(center[1]) + float(dy),
                    float(center[2]) + float(dz),
                )
                candidate_mask = _project_block_face_mask(
                    player_pose,
                    candidate_center,
                    image.shape,
                    basis_name=basis_name,
                    voxel_key=voxel_key,
                    occupied_voxels=occupied_voxels,
                )
                metrics = _block_refinement_score(task_key, image, candidate_mask)
                if float(metrics["refine_score"]) > float(best_metrics["refine_score"]):
                    best_center = candidate_center
                    best_mask = candidate_mask
                    best_metrics = dict(metrics)
                    best_offsets = (float(dx), float(dy), float(dz))
    best_metrics.update(
        {
            "world_refine_dx": float(best_offsets[0]),
            "world_refine_dy": float(best_offsets[1]),
            "world_refine_dz": float(best_offsets[2]),
        }
    )
    return best_center, best_mask, best_metrics


def _entity_corners(center: tuple[float, float, float], half_width: float = 0.45, half_height: float = 0.7) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    corners = []
    for dx in (-half_width, half_width):
        for dy in (-half_height, half_height):
            for dz in (-half_width, half_width):
                corners.append((cx + dx, cy + dy, cz + dz))
    return corners


def _score_candidate(center: Dict[str, float], image_shape: tuple[int, int, int] | tuple[int, int]) -> float:
    height, width = int(image_shape[0]), int(image_shape[1])
    return (
        abs(center["x"] - (width / 2.0))
        + abs(center["y"] - (height / 2.0))
        + 0.2 * float(center["depth"])
    )


def _crossview_difference_metrics(
    reference_pose: Dict[str, float],
    reference_obs: np.ndarray,
    candidate_pose: Dict[str, float],
    candidate_obs: np.ndarray,
) -> Dict[str, float]:
    changed_mask = np.any(reference_obs != candidate_obs, axis=-1)
    return {
        "camera_displacement": float(
            _distance3(
                (reference_pose["x"], reference_pose["y"], reference_pose["z"]),
                (candidate_pose["x"], candidate_pose["y"], candidate_pose["z"]),
            )
        ),
        "yaw_delta_deg": float(_angle_delta_deg(reference_pose["yaw"], candidate_pose["yaw"])),
        "pitch_delta_deg": float(abs(reference_pose["pitch"] - candidate_pose["pitch"])),
        "mean_abs_diff": float(np.mean(np.abs(reference_obs.astype(np.int16) - candidate_obs.astype(np.int16)))),
        "changed_pixel_frac": float(np.count_nonzero(changed_mask) / max(1, changed_mask.size)),
    }


def _clip_bbox_to_image(
    obs_shape: tuple[int, int, int] | tuple[int, int],
    bbox: tuple[int, int, int, int] | list[int],
) -> tuple[int, int, int, int]:
    height, width = int(obs_shape[0]), int(obs_shape[1])
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return (
        max(0, min(width - 1, x0)),
        max(0, min(height - 1, y0)),
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
    )


def _goal_appearance_metrics(obs_image: np.ndarray, bbox_obs: tuple[int, int, int, int] | list[int]) -> Dict[str, float]:
    x0, y0, x1, y1 = _clip_bbox_to_image(obs_image.shape, bbox_obs)
    patch = obs_image[y0 : y1 + 1, x0 : x1 + 1]
    if patch.size == 0:
        patch = obs_image
    width_px = int(max(0, x1 - x0 + 1))
    height_px = int(max(0, y1 - y0 + 1))
    area_frac = float((width_px * height_px) / max(1.0, float(int(obs_image.shape[0]) * int(obs_image.shape[1]))))
    return {
        "image_mean": float(obs_image.mean()),
        "image_std": float(obs_image.std()),
        "bbox_mean": float(patch.mean()),
        "bbox_std": float(patch.std()),
        "bbox_width_px": width_px,
        "bbox_height_px": height_px,
        "bbox_area_frac": area_frac,
    }


def _is_valid_crossview_candidate(metrics: Dict[str, float]) -> bool:
    if float(metrics["camera_displacement"]) < AUTO_GOAL_MIN_CAMERA_DISPLACEMENT:
        return False
    return bool(
        float(metrics["yaw_delta_deg"]) >= AUTO_GOAL_MIN_YAW_DELTA_DEG
        or float(metrics["pitch_delta_deg"]) >= AUTO_GOAL_MIN_PITCH_DELTA_DEG
        or float(metrics["mean_abs_diff"]) >= AUTO_GOAL_MIN_MEAN_ABS_DIFF
        or float(metrics["changed_pixel_frac"]) >= AUTO_GOAL_MIN_CHANGED_PIXEL_FRAC
    )


def _is_valid_goal_appearance(task_key: str, metrics: Dict[str, float]) -> bool:
    base_valid = bool(
        float(metrics["image_mean"]) >= AUTO_GOAL_MIN_IMAGE_MEAN
        and float(metrics["bbox_mean"]) >= AUTO_GOAL_MIN_BBOX_MEAN
        and int(metrics.get("bbox_height_px", 0)) >= int(AUTO_GOAL_MIN_BBOX_HEIGHT_PX)
        and (
            int(AUTO_GOAL_MIN_BBOX_WIDTH_PX) <= 0
            or int(metrics.get("bbox_width_px", 0)) >= int(AUTO_GOAL_MIN_BBOX_WIDTH_PX)
        )
        and float(metrics.get("bbox_area_frac", 0.0)) >= float(AUTO_GOAL_MIN_BBOX_AREA_FRAC)
    )
    if not base_valid:
        return False
    if task_key in {"mine_coal", "mine_emerald"} and float(metrics.get("bbox_area_frac", 0.0)) < float(AUTO_GOAL_MIN_MINE_BBOX_AREA_FRAC):
        return False
    return True


def _goal_target_visibility_metrics(
    task_key: str,
    goal_image: np.ndarray,
    goal_mask: np.ndarray | None,
) -> Dict[str, float]:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return {}
    if not isinstance(goal_image, np.ndarray) or goal_image.ndim != 3:
        return {
            "mask_area_px": 0.0,
            "mask_gray_mean": 0.0,
            "mask_gray_std": 0.0,
            "ring_gray_mean": 0.0,
            "ring_gray_std": 0.0,
            "mask_vs_ring_gray_mean_gap": 0.0,
            "mask_vs_ring_gray_std_gap": 0.0,
            "dark_frac_60": 0.0,
            "dark_frac_80": 0.0,
            "green_excess_mean": 0.0,
            "green_dominant_frac": 0.0,
        }
    mask_array = np.asarray(goal_mask if goal_mask is not None else 0)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    mask = mask_array > 0
    if not np.any(mask):
        return {
            "mask_area_px": 0.0,
            "mask_gray_mean": 0.0,
            "mask_gray_std": 0.0,
            "ring_gray_mean": 0.0,
            "ring_gray_std": 0.0,
            "mask_vs_ring_gray_mean_gap": 0.0,
            "mask_vs_ring_gray_std_gap": 0.0,
            "dark_frac_60": 0.0,
            "dark_frac_80": 0.0,
            "green_excess_mean": 0.0,
            "green_dominant_frac": 0.0,
        }
    pixels = goal_image[mask].astype(np.float32)
    gray = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    green_excess = np.maximum(0.0, pixels[:, 1] - np.maximum(pixels[:, 0], pixels[:, 2]))
    ring = np.logical_and(
        cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1) > 0,
        np.logical_not(mask),
    )
    ring_pixels = goal_image[ring].astype(np.float32)
    if ring_pixels.size > 0:
        ring_gray = 0.299 * ring_pixels[:, 0] + 0.587 * ring_pixels[:, 1] + 0.114 * ring_pixels[:, 2]
        ring_gray_mean = float(ring_gray.mean())
        ring_gray_std = float(ring_gray.std())
    else:
        ring_gray_mean = float(gray.mean())
        ring_gray_std = float(gray.std())
    return {
        "mask_area_px": float(mask.sum()),
        "mask_gray_mean": float(gray.mean()),
        "mask_gray_std": float(gray.std()),
        "ring_gray_mean": ring_gray_mean,
        "ring_gray_std": ring_gray_std,
        "mask_vs_ring_gray_mean_gap": float(gray.mean() - ring_gray_mean),
        "mask_vs_ring_gray_std_gap": float(gray.std() - ring_gray_std),
        "dark_frac_60": float(np.count_nonzero(gray < 60.0) / max(1, len(gray))),
        "dark_frac_80": float(np.count_nonzero(gray < 80.0) / max(1, len(gray))),
        "green_excess_mean": float(green_excess.mean()),
        "green_dominant_frac": float(np.count_nonzero(green_excess > 8.0) / max(1, len(green_excess))),
    }


def _shift_binary_mask_y(mask: np.ndarray, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=np.uint8)
    if dy >= 0:
        if dy < int(mask.shape[0]):
            shifted[dy:, :] = mask[: int(mask.shape[0]) - dy, :]
    else:
        shift = -int(dy)
        if shift < int(mask.shape[0]):
            shifted[: int(mask.shape[0]) - shift, :] = mask[shift:, :]
    return shifted


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return np.logical_and(dilated > 0, eroded <= 0).astype(np.uint8)


def _mask_ring(mask: np.ndarray, *, inner_iter: int = 1, outer_iter: int = 4) -> np.ndarray:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    outer = cv2.dilate(mask_u8, kernel, iterations=max(1, int(outer_iter)))
    inner = cv2.dilate(mask_u8, kernel, iterations=max(1, int(inner_iter)))
    return np.logical_and(outer > 0, inner <= 0).astype(np.uint8)


def _coal_projection_signal_map(raw_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(raw_image, dtype=np.float32)
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    darkness = np.clip((120.0 - gray) / 120.0, 0.0, 1.0)
    green_excess = np.clip((green - np.maximum(red, blue)) / 80.0, 0.0, 1.0)
    coal_like = np.clip(darkness * (1.0 - 0.75 * green_excess), 0.0, 1.0)
    gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    edge_map = cv2.Canny(gray_u8, 60, 140).astype(np.float32) / 255.0
    return coal_like, green_excess, edge_map


def _crop_projection_consistency_inputs(
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    *,
    margin: int = AUTO_GOAL_PROJECTION_CONSISTENCY_CROP_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    mask_u8 = (np.asarray(raw_mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return np.asarray(raw_image), mask_u8
    x0, y0, x1, y1 = _mask_bbox(mask_u8)
    pad = max(
        int(margin),
        int(max(abs(int(dy)) for dy in AUTO_GOAL_PROJECTION_CONSISTENCY_DY_VALUES)) + 6,
    )
    height, width = int(mask_u8.shape[0]), int(mask_u8.shape[1])
    crop_x0 = max(0, x0 - pad)
    crop_y0 = max(0, y0 - pad)
    crop_x1 = min(width, x1 + pad + 1)
    crop_y1 = min(height, y1 + pad + 1)
    return (
        np.asarray(raw_image)[crop_y0:crop_y1, crop_x0:crop_x1].copy(),
        mask_u8[crop_y0:crop_y1, crop_x0:crop_x1].copy(),
    )


def _projection_consistency_score(
    shifted_mask: np.ndarray,
    *,
    edge_map: np.ndarray,
    coal_map: np.ndarray,
    green_map: np.ndarray,
) -> tuple[float, Dict[str, float]]:
    mask_u8 = (np.asarray(shifted_mask) > 0).astype(np.uint8)
    boundary = _mask_boundary(mask_u8)
    ring = _mask_ring(mask_u8)
    inside = mask_u8 > 0
    boundary_bool = boundary > 0
    ring_bool = ring > 0
    if not np.any(inside) or not np.any(boundary_bool) or not np.any(ring_bool):
        return float("-inf"), {
            "edge_align": 0.0,
            "inside_coal": 0.0,
            "ring_coal": 0.0,
            "inside_green": 0.0,
            "inside_px": float(np.count_nonzero(inside)),
            "boundary_px": float(np.count_nonzero(boundary_bool)),
            "ring_px": float(np.count_nonzero(ring_bool)),
        }
    edge_align = float(edge_map[boundary_bool].mean())
    inside_coal = float(coal_map[inside].mean())
    ring_coal = float(coal_map[ring_bool].mean())
    inside_green = float(green_map[inside].mean())
    score = (
        1.2 * edge_align
        + 1.0 * inside_coal
        - 0.8 * ring_coal
        - 0.6 * inside_green
    )
    return float(score), {
        "edge_align": edge_align,
        "inside_coal": inside_coal,
        "ring_coal": ring_coal,
        "inside_green": inside_green,
        "inside_px": float(np.count_nonzero(inside)),
        "boundary_px": float(np.count_nonzero(boundary_bool)),
        "ring_px": float(np.count_nonzero(ring_bool)),
    }


def _goal_projection_consistency_metrics(
    task_key: str,
    raw_image: np.ndarray,
    raw_mask: np.ndarray | None,
) -> Dict[str, Any]:
    if not bool(AUTO_GOAL_ENABLE_PROJECTION_CONSISTENCY):
        return {}
    if task_key != "mine_coal":
        return {}
    if not isinstance(raw_image, np.ndarray) or raw_image.ndim != 3:
        return {}
    if not isinstance(raw_mask, np.ndarray):
        return {}
    mask_u8 = (np.asarray(raw_mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return {}
    cropped_image, cropped_mask = _crop_projection_consistency_inputs(raw_image, mask_u8)
    coal_map, green_map, edge_map = _coal_projection_signal_map(cropped_image)
    best_entry = None
    zero_entry = None
    for dy in AUTO_GOAL_PROJECTION_CONSISTENCY_DY_VALUES:
        shifted_mask = _shift_binary_mask_y(cropped_mask, int(dy))
        score, score_metrics = _projection_consistency_score(
            shifted_mask,
            edge_map=edge_map,
            coal_map=coal_map,
            green_map=green_map,
        )
        entry = {
            "dy": int(dy),
            "score": float(score),
            "metrics": score_metrics,
        }
        if best_entry is None or float(entry["score"]) > float(best_entry["score"]):
            best_entry = entry
        if int(dy) == 0:
            zero_entry = entry
    if zero_entry is None or best_entry is None:
        return {}
    best_dy = int(best_entry["dy"])
    zero_score = float(zero_entry["score"])
    best_score = float(best_entry["score"])
    gain = float(best_score - zero_score)
    is_consistent = bool(
        abs(best_dy) <= int(AUTO_GOAL_PROJECTION_CONSISTENCY_MAX_ABS_DY)
        and gain <= float(AUTO_GOAL_PROJECTION_CONSISTENCY_MAX_GAIN)
    )
    return {
        "best_dy": int(best_dy),
        "zero_score": float(zero_score),
        "best_score": float(best_score),
        "gain": float(gain),
        "accept": bool(is_consistent),
        "best_metrics": {
            str(key): round(float(value), 6)
            for key, value in dict(best_entry.get("metrics") or {}).items()
        },
        "zero_metrics": {
            str(key): round(float(value), 6)
            for key, value in dict(zero_entry.get("metrics") or {}).items()
        },
    }


def _attach_goal_target_visibility_metadata(
    synthesis: Dict[str, Any],
    visibility_metrics: Dict[str, float],
) -> None:
    if not visibility_metrics:
        return
    metadata = synthesis.get("metadata")
    if not isinstance(metadata, dict):
        return
    metadata.update(
        {
            "target_visibility_mask_area_px": int(round(float(visibility_metrics.get("mask_area_px", 0.0)))),
            "target_visibility_mask_gray_mean": round(float(visibility_metrics.get("mask_gray_mean", 0.0)), 4),
            "target_visibility_mask_gray_std": round(float(visibility_metrics.get("mask_gray_std", 0.0)), 4),
            "target_visibility_ring_gray_mean": round(float(visibility_metrics.get("ring_gray_mean", 0.0)), 4),
            "target_visibility_ring_gray_std": round(float(visibility_metrics.get("ring_gray_std", 0.0)), 4),
            "target_visibility_mean_gap": round(float(visibility_metrics.get("mask_vs_ring_gray_mean_gap", 0.0)), 4),
            "target_visibility_std_gap": round(float(visibility_metrics.get("mask_vs_ring_gray_std_gap", 0.0)), 4),
            "target_visibility_dark_frac_60": round(float(visibility_metrics.get("dark_frac_60", 0.0)), 6),
            "target_visibility_dark_frac_80": round(float(visibility_metrics.get("dark_frac_80", 0.0)), 6),
            "target_visibility_green_excess_mean": round(float(visibility_metrics.get("green_excess_mean", 0.0)), 4),
            "target_visibility_green_dominant_frac": round(float(visibility_metrics.get("green_dominant_frac", 0.0)), 6),
        }
    )


def _attach_goal_projection_consistency_metadata(
    synthesis: Dict[str, Any],
    consistency_metrics: Dict[str, Any],
) -> None:
    if not consistency_metrics:
        return
    metadata = synthesis.get("metadata")
    if not isinstance(metadata, dict):
        return
    metadata.update(
        {
            "projection_consistency_accept": bool(consistency_metrics.get("accept", False)),
            "projection_consistency_best_dy": int(consistency_metrics.get("best_dy", 0)),
            "projection_consistency_zero_score": round(float(consistency_metrics.get("zero_score", 0.0)), 6),
            "projection_consistency_best_score": round(float(consistency_metrics.get("best_score", 0.0)), 6),
            "projection_consistency_gain": round(float(consistency_metrics.get("gain", 0.0)), 6),
            "projection_consistency_best_metrics": _sanitize_debug_value(consistency_metrics.get("best_metrics") or {}),
            "projection_consistency_zero_metrics": _sanitize_debug_value(consistency_metrics.get("zero_metrics") or {}),
        }
    )


def _is_valid_goal_face_visibility(
    task_key: str,
    synthesis: Dict[str, Any],
    *,
    preferred_face: str = "",
    blueprint_id: str = "",
    pose_family: str = "",
) -> bool:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return True
    metadata = synthesis.get("metadata") or {}
    if str(metadata.get("goal_source") or "").strip() == "pose_centerbox_fallback":
        return False
    preferred_face = str(preferred_face or "").strip()
    blueprint_id = str(blueprint_id or "").strip()
    pose_family = str(pose_family or "").strip()
    visible_faces = metadata.get("visible_faces") or []
    visible_labels = {
        str(entry.get("face_label") or "").strip()
        for entry in visible_faces
        if isinstance(entry, dict)
    }
    lateral_faces = {"+x", "-x", "+z", "-z"}
    metadata["visible_face_labels"] = sorted(label for label in visible_labels if label)
    if not preferred_face:
        return bool(visible_labels & lateral_faces)
    if preferred_face == "+y":
        return False
    allowed_faces = {preferred_face}
    # Corner and alcove layouts have two behaviorally valid goal views:
    # a branch-facing lateral side after completing the turn, and a
    # corridor-mouth frontal face while entering the branch.
    if (
        preferred_face in {"+x", "-x"}
        and (
            blueprint_id in {"turn_left", "turn_right", "side_alcove_left", "side_alcove_right"}
            or pose_family in {"corner_center", "corner_entry", "alcove_entry"}
        )
    ):
        allowed_faces.add("+z")
    metadata["allowed_preferred_faces"] = sorted(allowed_faces)
    if visible_labels & allowed_faces:
        return True
    # Reject top-only or mismatched-face goals for mine tasks. They tend to
    # produce ambiguous supervision even when the bbox itself looks valid.
    return False


def _should_reject_goal_collision_warning(task_key: str, collision_warning: bool) -> bool:
    return bool(collision_warning) and task_key in {"mine_coal", "mine_emerald"}


def _is_valid_goal_target_visibility(
    task_key: str,
    synthesis: Dict[str, Any],
    *,
    visibility_metrics: Dict[str, float] | None = None,
) -> bool:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return True
    metadata = synthesis.get("metadata") or {}
    if str(metadata.get("goal_source") or "").strip() == "pose_centerbox_fallback":
        return False
    if visibility_metrics is None:
        visibility_metrics = _goal_target_visibility_metrics(
            task_key,
            np.asarray(synthesis.get("goal_image")),
            synthesis.get("goal_mask"),
        )
    if task_key != "mine_coal":
        return True
    if float(visibility_metrics.get("mask_area_px", 0.0)) <= 0.0:
        return False
    if float(visibility_metrics.get("mask_gray_std", 0.0)) < float(AUTO_GOAL_MIN_COAL_MASK_STD):
        return False
    if (
        float(visibility_metrics.get("green_dominant_frac", 0.0)) >= float(AUTO_GOAL_MAX_COAL_GREEN_DOMINANT_FRAC)
        and float(visibility_metrics.get("green_excess_mean", 0.0)) >= float(AUTO_GOAL_MAX_COAL_GREEN_EXCESS_MEAN)
    ):
        return False
    if (
        float(visibility_metrics.get("dark_frac_80", 0.0)) >= float(AUTO_GOAL_MAX_COAL_DARK80_OCCLUDED)
        and float(visibility_metrics.get("mask_vs_ring_gray_mean_gap", 0.0)) <= 2.0
    ):
        return False
    return True


def _is_valid_goal_projection_consistency(
    task_key: str,
    synthesis: Dict[str, Any],
    *,
    consistency_metrics: Dict[str, Any] | None = None,
) -> bool:
    if task_key != "mine_coal":
        return True
    metadata = synthesis.get("metadata") or {}
    if str(metadata.get("goal_source") or "").strip() == "pose_centerbox_fallback":
        return False
    if consistency_metrics is None:
        consistency_metrics = {}
    if not consistency_metrics:
        return True
    return bool(consistency_metrics.get("accept", False))


def _configured_target_entries(task_key: str, task_config: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    if not isinstance(task_config, dict):
        return []
    if task_key in AUTO_GOAL_BLOCK_TYPES:
        target_names = AUTO_GOAL_BLOCK_TYPES[task_key]
        default_type = f"{target_names[0]}_ore"
    elif task_key in AUTO_GOAL_MOB_TYPES:
        target_names = AUTO_GOAL_MOB_TYPES[task_key]
        default_type = str(target_names[0])
    else:
        target_names = ()
        default_type = "target"

    procedural_layout = task_config.get("procedural_layout") if isinstance(task_config.get("procedural_layout"), dict) else {}
    task_preferred_face = str(
        task_config.get("goal_camera_preferred_face")
        or task_config.get("goal_preferred_face")
        or procedural_layout.get("goal_camera_preferred_face")
        or procedural_layout.get("goal_preferred_face")
        or task_config.get("preferred_face")
        or procedural_layout.get("preferred_face")
        or ""
    ).strip()

    entries = []
    single_center = task_config.get("target_world_center")
    if isinstance(single_center, (list, tuple)) and len(single_center) == 3:
        entries.append(
            {
                "target_type": _normalize_name(task_config.get("target_type") or default_type),
                "world_center": tuple(float(value) for value in single_center),
                "preferred_face": task_preferred_face,
            }
        )

    for key in ("target_blocks", "targets"):
        raw_items = task_config.get(key) or []
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            center = item.get("world_center") or item.get("target_world_center") or item.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) == 3):
                continue
            entries.append(
                {
                    "target_type": _normalize_name(item.get("type") or item.get("target_type") or default_type),
                    "world_center": tuple(float(value) for value in center),
                    "preferred_face": str(
                        item.get("goal_camera_preferred_face")
                        or item.get("goal_preferred_face")
                        or item.get("preferred_face")
                        or task_preferred_face
                        or ""
                    ).strip(),
                }
            )

    unique_entries = []
    seen: dict[tuple[str, tuple[float, float, float]], int] = {}
    for entry in entries:
        target_type = _normalize_name(entry.get("target_type"))
        if target_names and not any(name in target_type for name in target_names):
            continue
        key = (target_type, tuple(round(float(value), 4) for value in entry["world_center"]))
        preferred_face = str(entry.get("preferred_face") or "").strip()
        existing_index = seen.get(key)
        if existing_index is not None:
            if (not unique_entries[existing_index].get("preferred_face")) and preferred_face:
                unique_entries[existing_index]["preferred_face"] = preferred_face
            continue
        seen[key] = len(unique_entries)
        unique_entries.append(
            {
                "target_type": target_type,
                "world_center": tuple(float(value) for value in entry["world_center"]),
                "preferred_face": preferred_face,
            }
        )
    return unique_entries


def _lateral_face_axes(face_label: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    normalized = str(face_label or "").strip().lower()
    mapping = {
        "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "+z": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "-z": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    }
    return mapping.get(normalized)


def _goal_pose_face_distance(
    camera_position: tuple[float, float, float],
    target_world_center: tuple[float, float, float],
    face_label: str,
) -> float | None:
    axes = _lateral_face_axes(face_label)
    if axes is None:
        return None
    outward, _ = axes
    dx = float(camera_position[0]) - float(target_world_center[0])
    dz = float(camera_position[2]) - float(target_world_center[2])
    center_offset = float(dx) * float(outward[0]) + float(dz) * float(outward[2])
    return max(0.0, float(center_offset) - 0.5)


def _goal_pose_sampled_distance(
    camera_position: tuple[float, float, float],
    target_world_center: tuple[float, float, float],
    face_label: str,
) -> float:
    face_distance = _goal_pose_face_distance(camera_position, target_world_center, face_label)
    if face_distance is not None:
        return float(face_distance)
    return float(_distance3(camera_position, target_world_center))


def _eye_position_from_camera_feet(
    camera_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(camera_position[0]),
        float(camera_position[1]) + float(DEFAULT_EYE_HEIGHT),
        float(camera_position[2]),
    )


def _interleave_pose_candidates(
    primary: list[Dict[str, Any]],
    secondary: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if not primary:
        return list(secondary)
    if not secondary:
        return list(primary)
    merged: list[Dict[str, Any]] = []
    primary_index = 0
    secondary_index = 0
    while primary_index < len(primary) or secondary_index < len(secondary):
        if primary_index < len(primary):
            merged.append(primary[primary_index])
            primary_index += 1
        if secondary_index < len(secondary):
            merged.append(secondary[secondary_index])
            secondary_index += 1
    return merged


def _canonical_face_distance_samples(
    *,
    blueprint_id: str,
    goal_face_label: str,
    face_label: str,
) -> tuple[float, ...]:
    # Side-alcove layouts only carve a single free block in front of the target
    # along the +z approach lane. Reusing the generic 1.0/1.5/2.0 face-distance
    # samples places the canonical frontal cameras beyond that carved pocket and
    # into the surrounding shell, which then gets rejected as camera collision.
    if (
        str(blueprint_id).strip() in {"side_alcove_left", "side_alcove_right"}
        and str(goal_face_label).strip() == "+z"
        and str(face_label).strip() == "+z"
    ):
        return AUTO_GOAL_CANONICAL_FACE_DISTANCE_SAMPLES_TIGHT_ALCOVE
    return AUTO_GOAL_CANONICAL_FACE_DISTANCE_SAMPLES


def _canonical_visible_face_hint_candidates(
    task_key: str,
    reference_pose: Dict[str, float],
    task_config: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return []
    if not isinstance(task_config, dict):
        return []
    configured_entries = _configured_target_entries(task_key, task_config)
    if not configured_entries:
        return []
    procedural_layout = task_config.get("procedural_layout") if isinstance(task_config.get("procedural_layout"), dict) else {}
    blueprint_id = str(procedural_layout.get("blueprint_id") or "").strip()
    realized_metrics = procedural_layout.get("realized_metrics") if isinstance(procedural_layout.get("realized_metrics"), dict) else {}
    action_margin = realized_metrics.get("action_margin") if isinstance(realized_metrics.get("action_margin"), dict) else {}
    goal_face_label = str(
        action_margin.get("goal_face_label")
        or task_config.get("goal_camera_preferred_face")
        or task_config.get("goal_preferred_face")
        or procedural_layout.get("goal_camera_preferred_face")
        or procedural_layout.get("goal_preferred_face")
        or ""
    ).strip()

    ordered_faces: list[str] = []

    def _append_face(face_label: Any) -> None:
        label = str(face_label or "").strip()
        if label not in {"+x", "-x", "+z", "-z"}:
            return
        if label not in ordered_faces:
            ordered_faces.append(label)

    _append_face(
        action_margin.get("goal_face_label")
        or task_config.get("goal_camera_preferred_face")
        or task_config.get("goal_preferred_face")
        or procedural_layout.get("goal_camera_preferred_face")
        or procedural_layout.get("goal_preferred_face")
    )
    for label in action_margin.get("open_face_labels") or []:
        _append_face(label)
    for label in ("+z", "-z", "+x", "-x"):
        _append_face(label)

    candidates: list[Dict[str, Any]] = []
    seen_keys: set[tuple[float, float, float, str]] = set()
    for target_index, entry in enumerate(configured_entries):
        target_world_center = (
            float(entry["world_center"][0]),
            float(entry["world_center"][1]),
            float(entry["world_center"][2]),
        )
        feet_y = math.floor(float(target_world_center[1]))
        target_type = _normalize_name(entry.get("target_type") or "")
        for face_rank, face_label in enumerate(ordered_faces):
            axes = _lateral_face_axes(face_label)
            if axes is None:
                continue
            outward, _ = axes
            distance_samples = _canonical_face_distance_samples(
                blueprint_id=blueprint_id,
                goal_face_label=goal_face_label,
                face_label=face_label,
            )
            for distance_rank, face_distance in enumerate(distance_samples):
                center_distance = 0.5 + float(face_distance)
                camera_position = (
                    float(target_world_center[0]) + float(outward[0]) * float(center_distance),
                    float(feet_y),
                    float(target_world_center[2]) + float(outward[2]) * float(center_distance),
                )
                dedupe_key = (
                    round(float(camera_position[0]), 4),
                    round(float(camera_position[1]), 4),
                    round(float(camera_position[2]), 4),
                    str(target_type),
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                yaw, pitch = _look_angles_for_target(camera_position, target_world_center)
                support_block = _support_block_for_pose(
                    {
                        "x": float(camera_position[0]),
                        "y": float(camera_position[1]),
                        "z": float(camera_position[2]),
                    }
                )
                sampled_center_distance = _distance3(camera_position, target_world_center)
                face_token = face_label.replace("+", "plus_").replace("-", "minus_")
                candidates.append(
                    {
                        "anchor_index": int(target_index),
                        "goal_source": "hint",
                        "target_type": target_type,
                        "target_world_center": target_world_center,
                        "camera_position": camera_position,
                        "sampled_distance": float(face_distance),
                        "sampled_center_distance": float(sampled_center_distance),
                        "sampled_face_distance": float(face_distance),
                        "sampled_azimuth_deg": 0.0,
                        "requested_height_offset": 0.0,
                        "sampled_height_offset": 0.0,
                        "sampled_pitch_offset": 0.0,
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "support_block": _sanitize_debug_value(support_block),
                        "hint_name": f"visible_face_{face_token}_d{int(round(float(face_distance) * 10.0)):02d}",
                        # Do not force a layout-defined face here. The whole point of
                        # this candidate family is to accept whichever lateral face is
                        # actually visible from the frontal camera we just placed.
                        "preferred_face": "",
                        "candidate_face_label": str(face_label),
                        "pose_family": "frontal_visible_face",
                        "selection_bias": float(5200.0 - 180.0 * face_rank - 120.0 * distance_rank),
                        "goal_mode_hint": "auto_pose_visible_face_frontal",
                    }
                )
    return candidates


def _goal_pose_hint_candidates(
    task_key: str,
    reference_pose: Dict[str, float],
    task_config: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    if not isinstance(task_config, dict):
        return []
    canonical_candidates = _canonical_visible_face_hint_candidates(task_key, reference_pose, task_config)
    procedural_layout = task_config.get("procedural_layout") if isinstance(task_config.get("procedural_layout"), dict) else {}
    blueprint_id = str(procedural_layout.get("blueprint_id") or "").strip()
    raw_candidates: list[Dict[str, Any]] = []
    configured_entries = _configured_target_entries(task_key, task_config)
    default_target = configured_entries[0] if len(configured_entries) == 1 else None
    raw_hints = task_config.get("goal_pose_hints") or []
    if not isinstance(raw_hints, list):
        return canonical_candidates
    for hint_index, hint in enumerate(raw_hints):
        if not isinstance(hint, dict):
            continue
        position = hint.get("position") or hint.get("camera_position")
        if not (isinstance(position, (list, tuple)) and len(position) == 3):
            continue
        target_center = hint.get("target_world_center")
        if not (isinstance(target_center, (list, tuple)) and len(target_center) == 3):
            if default_target is not None:
                target_center = default_target["world_center"]
            else:
                continue
        look_target_center = hint.get("look_target_world_center")
        if not (isinstance(look_target_center, (list, tuple)) and len(look_target_center) == 3):
            look_target_center = target_center
        if not (isinstance(target_center, (list, tuple)) and len(target_center) == 3):
            continue
        camera_position = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        target_world_center = (
            float(target_center[0]),
            float(target_center[1]),
            float(target_center[2]),
        )
        look_target_world_center = (
            float(look_target_center[0]),
            float(look_target_center[1]),
            float(look_target_center[2]),
        )
        yaw = hint.get("yaw")
        pitch = hint.get("pitch")
        if yaw in (None, "") or pitch in (None, ""):
            yaw, pitch = _look_angles_for_target(camera_position, look_target_world_center)
        target_type = _normalize_name(
            hint.get("target_type") or (default_target or {}).get("target_type") or ""
        )
        preferred_face = str(
            hint.get("preferred_face") or (default_target or {}).get("preferred_face") or ""
        ).strip()
        support_block = hint.get("support_block")
        if isinstance(support_block, dict) and support_block.get("enabled") is False:
            support_block = None
        elif not isinstance(support_block, dict):
            support_block = _support_block_for_pose(
                {
                    "x": float(camera_position[0]),
                    "y": float(camera_position[1]),
                    "z": float(camera_position[2]),
                }
            )
        sampled_center_distance = _distance3(camera_position, target_world_center)
        sampled_face_distance = _goal_pose_face_distance(camera_position, target_world_center, preferred_face)
        sampled_distance = _goal_pose_sampled_distance(camera_position, target_world_center, preferred_face)
        raw_height_offset = hint.get("height_offset")
        if isinstance(raw_height_offset, (int, float)):
            sampled_height_offset = float(raw_height_offset)
        elif isinstance(support_block, dict) and support_block.get("feet_y") is not None:
            sampled_height_offset = float(camera_position[1]) - float(support_block["feet_y"])
        else:
            sampled_height_offset = float(camera_position[1]) - float(reference_pose["y"])
        raw_candidates.append(
            {
                "anchor_index": int(len(configured_entries) + hint_index),
                "goal_source": "hint",
                "target_type": target_type,
                "target_world_center": target_world_center,
                "camera_position": camera_position,
                "sampled_distance": float(sampled_distance),
                "sampled_center_distance": float(sampled_center_distance),
                "sampled_face_distance": None if sampled_face_distance is None else float(sampled_face_distance),
                "sampled_azimuth_deg": 0.0,
                "requested_height_offset": float(sampled_height_offset),
                "sampled_height_offset": float(sampled_height_offset),
                "sampled_pitch_offset": 0.0,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "support_block": _sanitize_debug_value(support_block),
                "hint_name": str(hint.get("name") or f"hint_{hint_index:03d}"),
                "preferred_face": preferred_face,
                "pose_family": str(hint.get("pose_family") or ""),
                "selection_bias": float(hint.get("selection_bias", 0.0) or 0.0),
                "goal_mode_hint": "auto_pose_hint_cross_view",
            }
        )
    candidate_mode = str(
        task_config.get("auto_goal_pose_candidate_mode")
        or task_config.get("goal_pose_candidate_mode")
        or ""
    ).strip().lower()
    if candidate_mode in {"hints_only", "raw_only", "no_canonical"}:
        return raw_candidates
    if candidate_mode in {"canonical_only", "visible_face_only"}:
        return canonical_candidates
    if blueprint_id in {"side_alcove_left", "side_alcove_right"} and raw_candidates:
        filtered_canonical_candidates = [
            candidate
            for candidate in canonical_candidates
            if str(candidate.get("candidate_face_label") or "").strip() != "+z"
        ]
        return _interleave_pose_candidates(raw_candidates, filtered_canonical_candidates)
    return _interleave_pose_candidates(canonical_candidates, raw_candidates)


def _extract_target_anchors(task_key: str, info: Dict[str, Any], task_config: Dict[str, Any] | None = None) -> list[Dict[str, Any]]:
    player_pose = _player_pose(info)
    player_center = (player_pose["x"], player_pose["y"], player_pose["z"])
    anchors = []
    configured_entries = _configured_target_entries(task_key, task_config)
    if configured_entries:
        for entry in configured_entries:
            world_center = entry["world_center"]
            anchors.append(
                {
                    "task_key": task_key,
                    "goal_source": "configured",
                    "target_type": entry["target_type"],
                    "world_center": world_center,
                    "preferred_face": str(entry.get("preferred_face") or "").strip(),
                    "sort_distance": _distance3(world_center, player_center),
                }
            )
        anchors.sort(key=lambda item: float(item["sort_distance"]))
        return anchors[:AUTO_GOAL_MAX_ANCHORS]
    if task_key in AUTO_GOAL_BLOCK_TYPES:
        target_types = AUTO_GOAL_BLOCK_TYPES[task_key]
        voxels = info.get("voxels") or []
        if not isinstance(voxels, list):
            return []
        for voxel in voxels:
            if not isinstance(voxel, dict):
                continue
            voxel_type = _normalize_name(voxel.get("type"))
            if not any(target_type in voxel_type for target_type in target_types):
                continue
            raw_offset = (
                float(voxel.get("x", 0.0)),
                float(voxel.get("y", 0.0)),
                float(voxel.get("z", 0.0)),
            )
            world_center = _voxel_world_center_from_player_pose(player_pose, raw_offset)
            anchors.append(
                {
                    "task_key": task_key,
                    "goal_source": "voxels",
                    "target_type": voxel_type,
                    "world_center": world_center,
                    "preferred_face": "",
                    "sort_distance": _distance3(world_center, player_center),
                }
            )
    elif task_key in AUTO_GOAL_MOB_TYPES:
        target_types = AUTO_GOAL_MOB_TYPES[task_key]
        mobs = _extract_mobs(info)
        for mob in mobs:
            name = _normalize_name(mob.get("name"))
            if not any(target_type in name for target_type in target_types):
                continue
            world_center = (
                float(mob.get("x", 0.0)),
                float(mob.get("y", 0.0)) + 0.7,
                float(mob.get("z", 0.0)),
            )
            anchors.append(
                {
                    "task_key": task_key,
                    "goal_source": "mobs",
                    "target_type": name,
                    "world_center": world_center,
                    "preferred_face": "",
                    "sort_distance": _distance3(world_center, player_center),
                }
            )
    anchors.sort(key=lambda item: float(item["sort_distance"]))
    return anchors[:AUTO_GOAL_MAX_ANCHORS]


def _camera_collides_with_nearby_voxel(info: Dict[str, Any], camera_position: tuple[float, float, float], margin: float = 0.05) -> bool:
    player_pose = _player_pose(info)
    voxels = info.get("voxels") or []
    if not isinstance(voxels, list):
        return False
    cam_x, cam_y, cam_z = [float(v) for v in camera_position]
    base_x = math.floor(float(player_pose["x"]))
    base_y = math.floor(float(player_pose["y"]))
    base_z = math.floor(float(player_pose["z"]))
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if not voxel_type or voxel_type in {"air", "cave_air", "void_air"}:
            continue
        # Grid voxels are anchored to BlockPos(floor(player_pos)), not to the
        # player's fractional coordinates. Using the raw pose here shifts every
        # AABB by up to half a block at *.5 positions and falsely marks valid
        # center-lane cameras as colliding.
        min_x = float(base_x) + float(voxel.get("x", 0.0)) - margin
        min_y = float(base_y) + float(voxel.get("y", 0.0)) - margin
        min_z = float(base_z) + float(voxel.get("z", 0.0)) - margin
        max_x = min_x + 1.0 + 2.0 * margin
        max_y = min_y + 1.0 + 2.0 * margin
        max_z = min_z + 1.0 + 2.0 * margin
        if min_x <= cam_x <= max_x and min_y <= cam_y <= max_y and min_z <= cam_z <= max_z:
            return True
    return False


def _adjust_pose_candidate_for_clearance(
    pose_candidate: Dict[str, Any],
    info: Dict[str, Any],
    *,
    step_size: float = 0.1,
    max_backoff_steps: int = 12,
) -> Dict[str, Any]:
    if not isinstance(pose_candidate, dict) or not isinstance(info, dict):
        return pose_candidate
    raw_camera_position = pose_candidate.get("camera_position")
    raw_target_world_center = pose_candidate.get("target_world_center")
    if not (
        isinstance(raw_camera_position, (list, tuple))
        and len(raw_camera_position) == 3
        and isinstance(raw_target_world_center, (list, tuple))
        and len(raw_target_world_center) == 3
    ):
        return pose_candidate
    face_label = str(
        pose_candidate.get("candidate_face_label") or pose_candidate.get("preferred_face") or ""
    ).strip()
    axes = _lateral_face_axes(face_label)
    if axes is None:
        return pose_candidate
    camera_position = (
        float(raw_camera_position[0]),
        float(raw_camera_position[1]),
        float(raw_camera_position[2]),
    )
    eye_position = _eye_position_from_camera_feet(camera_position)
    if not _camera_collides_with_nearby_voxel(info, eye_position):
        return pose_candidate
    outward, _ = axes
    target_world_center = (
        float(raw_target_world_center[0]),
        float(raw_target_world_center[1]),
        float(raw_target_world_center[2]),
    )
    base_x, base_y, base_z = camera_position
    for step_index in range(1, max(1, int(max_backoff_steps)) + 1):
        backoff_distance = float(step_size) * float(step_index)
        trial_camera_position = (
            float(base_x) + float(outward[0]) * float(backoff_distance),
            float(base_y),
            float(base_z) + float(outward[2]) * float(backoff_distance),
        )
        trial_eye_position = _eye_position_from_camera_feet(trial_camera_position)
        if _camera_collides_with_nearby_voxel(info, trial_eye_position):
            continue
        adjusted_candidate = dict(pose_candidate)
        yaw, pitch = _look_angles_for_target(trial_camera_position, target_world_center)
        adjusted_candidate.update(
            {
                "camera_position": trial_camera_position,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "support_block": _sanitize_debug_value(
                    _support_block_for_pose(
                        {
                            "x": float(trial_camera_position[0]),
                            "y": float(trial_camera_position[1]),
                            "z": float(trial_camera_position[2]),
                        }
                    )
                ),
                "sampled_distance": float(
                    _goal_pose_sampled_distance(trial_camera_position, target_world_center, face_label)
                ),
                "sampled_center_distance": float(_distance3(trial_camera_position, target_world_center)),
                "sampled_face_distance": _goal_pose_face_distance(
                    trial_camera_position,
                    target_world_center,
                    face_label,
                ),
                "clearance_backoff": float(backoff_distance),
            }
        )
        return adjusted_candidate
    return pose_candidate


def _describe_available_targets(task_key: str, info: Dict[str, Any]) -> str:
    if task_key in AUTO_GOAL_BLOCK_TYPES:
        voxels = info.get("voxels") or []
        if not isinstance(voxels, list):
            return "voxels unavailable"
        names = sorted({_normalize_name(voxel.get("type")) for voxel in voxels if isinstance(voxel, dict) and voxel.get("type")})
        preview = names[:12]
        return f"voxel_types={preview}"
    if task_key in AUTO_GOAL_MOB_TYPES:
        mobs = _extract_mobs(info)
        names = sorted({_normalize_name(mob.get('name')) for mob in mobs if isinstance(mob, dict) and mob.get('name')})
        preview = names[:12]
        return f"mob_names={preview}"
    return "no target description"


def _look_angles_for_target(
    camera_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    *,
    eye_height: float = DEFAULT_EYE_HEIGHT,
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


def _synthesize_block_goal(task_key: str, obs_image: np.ndarray, raw_image: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
    target_types = AUTO_GOAL_BLOCK_TYPES[task_key]
    player_pose = _player_pose(info)
    voxels = info.get("voxels") or []
    if not isinstance(voxels, list):
        raise RuntimeError(f"voxels missing or malformed for auto-goal task {task_key}")
    best = None
    matching_voxels = 0
    projected_voxels = 0
    occupied_voxels = set()
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if _is_non_occluding_goal_block(voxel_type):
            continue
        occupied_voxels.add(
            (
                int(round(float(voxel.get("x", 0.0)))),
                int(round(float(voxel.get("y", 0.0)))),
                int(round(float(voxel.get("z", 0.0)))),
            )
        )
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if not any(target_type in voxel_type for target_type in target_types):
            continue
        matching_voxels += 1
        raw_offset = (
            float(voxel.get("x", 0.0)),
            float(voxel.get("y", 0.0)),
            float(voxel.get("z", 0.0)),
        )
        voxel_key = (
            int(round(raw_offset[0])),
            int(round(raw_offset[1])),
            int(round(raw_offset[2])),
        )
        center = _voxel_world_center_from_player_pose(player_pose, raw_offset)
        candidate_found = False
        voxel_projected = False
        for basis_name, forward, right, up in _camera_basis_variants(player_pose):
            refined_center, raw_mask, world_refine = _refine_block_world_projection(
                task_key,
                player_pose,
                raw_image,
                center,
                basis_name=basis_name,
                voxel_key=voxel_key,
                occupied_voxels=occupied_voxels,
            )
            if int(raw_mask.sum()) <= 0:
                continue
            candidate_found = True
            voxel_projected = True
            raw_bbox = _mask_bbox(raw_mask)
            raw_cx, raw_cy = _mask_center(raw_mask)
            center_projection = _project_world_point_with_basis(
                np.array(
                    [
                        float(player_pose["x"]),
                        float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
                        float(player_pose["z"]),
                    ],
                    dtype=np.float32,
                ),
                np.asarray(refined_center, dtype=np.float32),
                raw_image.shape,
                forward=forward,
                right=right,
                up=up,
            )
            depth = float(center_projection["depth"]) if center_projection is not None else 0.0
            raw_center = {
                "x": float(raw_cx),
                "y": float(raw_cy),
                "depth": depth,
                "basis_name": basis_name,
            }
            signal = _block_mask_signal(task_key, raw_image, raw_mask)
            score = float(world_refine["refine_score"]) - 0.15 * _score_candidate(raw_center, raw_image.shape)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "world_center": refined_center,
                    "base_world_center": center,
                    "voxel_key": voxel_key,
                    "voxel_offset_raw": raw_offset,
                    "raw_bbox": raw_bbox,
                    "raw_center": raw_center,
                    "raw_mask": raw_mask,
                    "target_type": voxel_type,
                    "basis_name": basis_name,
                    "signal_mean": float(signal["signal_mean"]),
                    "signal_sum": float(signal["signal_sum"]),
                    "signal_frac": float(signal["signal_frac"]),
                    "world_refine_dx": float(world_refine["world_refine_dx"]),
                    "world_refine_dy": float(world_refine["world_refine_dy"]),
                    "world_refine_dz": float(world_refine["world_refine_dz"]),
                    "world_refine_score": float(world_refine["refine_score"]),
                }
        if voxel_projected:
            projected_voxels += 1
        if candidate_found:
            continue
        projected = _project_bbox(player_pose, _block_corners(center), raw_image.shape)
        if projected is None:
            continue
        projected_voxels += 1
        raw_bbox, raw_center = projected
        score = -_score_candidate(raw_center, raw_image.shape)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "world_center": center,
                "base_world_center": center,
                "voxel_key": voxel_key,
                "voxel_offset_raw": raw_offset,
                "raw_bbox": raw_bbox,
                "raw_center": raw_center,
                "raw_mask": None,
                "target_type": voxel_type,
                "basis_name": str(raw_center.get("basis_name", "")),
                "signal_mean": 0.0,
                "signal_sum": 0.0,
                "signal_frac": 0.0,
                "world_refine_dx": 0.0,
                "world_refine_dy": 0.0,
                "world_refine_dz": 0.0,
                "world_refine_score": float(score),
            }
    if best is None:
        raise RuntimeError(
            f"Failed to synthesize goal for {task_key}: no visible matching voxel found "
            f"(matching_voxels={matching_voxels}, projected_voxels={projected_voxels}, "
            f"player_yaw={player_pose['yaw']:.3f}, player_pitch={player_pose['pitch']:.3f})"
        )
    raw_mask = best.get("raw_mask")
    if raw_mask is None:
        raw_mask = _project_block_face_mask(
            player_pose,
            best["world_center"],
            raw_image.shape,
            basis_name=best["basis_name"],
            voxel_key=best["voxel_key"],
            occupied_voxels=occupied_voxels,
        )
    if int(raw_mask.sum()) > 0:
        obs_mask = _resize_binary_mask(raw_mask, obs_image.shape)
        refinement_metrics = None
        if task_key == "mine_emerald":
            obs_mask, refinement_metrics = _refine_block_mask_image(task_key, obs_image, obs_mask)
        obs_bbox = _mask_bbox(obs_mask)
        raw_mask_bbox = _mask_bbox(raw_mask)
    else:
        obs_bbox = _scale_bbox(best["raw_bbox"], raw_image.shape, obs_image.shape)
        obs_mask = _make_bbox_mask(obs_image.shape, obs_bbox)
        raw_mask_bbox = best["raw_bbox"]
        refinement_metrics = None
    if refinement_metrics is None:
        refinement_metadata = {
            "mask_refine_applied": False,
        }
    else:
        refinement_metadata = {
            "mask_refine_applied": True,
            "mask_refine_dx": int(refinement_metrics["best_dx"]),
            "mask_refine_dy": int(refinement_metrics["best_dy"]),
            "mask_refine_scale": round(float(refinement_metrics["best_scale"]), 4),
            "mask_refine_score_before": round(float(refinement_metrics["base_refine_score"]), 4),
            "mask_refine_score_after": round(float(refinement_metrics["refine_score"]), 4),
            "mask_refine_signal_sum": round(float(refinement_metrics["signal_sum"]), 4),
            "mask_refine_signal_frac": round(float(refinement_metrics["signal_frac"]), 6),
            "mask_refine_ring_signal_sum": round(float(refinement_metrics["ring_signal_sum"]), 4),
            "mask_refine_ring_signal_frac": round(float(refinement_metrics["ring_signal_frac"]), 6),
        }
    return {
        "goal_image": np.asarray(obs_image, dtype=np.uint8).copy(),
        "goal_mask": obs_mask,
        "segment_type": "Mine",
        "_debug_raw_mask": raw_mask.copy(),
        "metadata": {
            "goal_mode": "auto_current_view",
            "goal_source": "voxels",
            "task_key": task_key,
            "target_type": best["target_type"],
            "voxel_key": [int(best["voxel_key"][0]), int(best["voxel_key"][1]), int(best["voxel_key"][2])],
            "voxel_offset_raw": [round(float(best["voxel_offset_raw"][0]), 4), round(float(best["voxel_offset_raw"][1]), 4), round(float(best["voxel_offset_raw"][2]), 4)],
            "base_target_world_center": [round(float(value), 4) for value in best["base_world_center"]],
            "target_world_center": [round(float(value), 4) for value in best["world_center"]],
            "target_center_raw": {
                "x": round(float(best["raw_center"]["x"]), 3),
                "y": round(float(best["raw_center"]["y"]), 3),
                "depth": round(float(best["raw_center"]["depth"]), 3),
            },
            "bbox_raw": list(best["raw_bbox"]),
            "mask_bbox_raw": list(raw_mask_bbox),
            "bbox_obs": list(obs_bbox),
            "projection_basis": best["basis_name"],
            "matching_voxels": int(matching_voxels),
            "projected_voxels": int(projected_voxels),
            "mask_source": "projected_visible_faces" if int(raw_mask.sum()) > 0 else "bbox_fallback",
            "candidate_signal_mean": round(float(best["signal_mean"]), 4),
            "candidate_signal_sum": round(float(best["signal_sum"]), 4),
            "candidate_signal_frac": round(float(best["signal_frac"]), 6),
            "world_refine_dx": round(float(best["world_refine_dx"]), 4),
            "world_refine_dy": round(float(best["world_refine_dy"]), 4),
            "world_refine_dz": round(float(best["world_refine_dz"]), 4),
            "world_refine_score": round(float(best["world_refine_score"]), 4),
            **refinement_metadata,
        },
    }


def _synthesize_configured_block_goal(
    task_key: str,
    obs_image: np.ndarray,
    raw_image: np.ndarray,
    info: Dict[str, Any],
    *,
    target_world_center: tuple[float, float, float] | list[float],
    target_type: str = "",
) -> Dict[str, Any]:
    player_pose = _player_pose(info)
    voxels = info.get("voxels") or []
    if not isinstance(voxels, list):
        raise RuntimeError(f"voxels missing or malformed for configured auto-goal task {task_key}")
    occupied_voxels: set[tuple[int, int, int]] = set()
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if _is_non_occluding_goal_block(voxel_type):
            continue
        occupied_voxels.add(
            (
                int(round(float(voxel.get("x", 0.0)))),
                int(round(float(voxel.get("y", 0.0)))),
                int(round(float(voxel.get("z", 0.0)))),
            )
        )

    center = (float(target_world_center[0]), float(target_world_center[1]), float(target_world_center[2]))
    voxel_key = _voxel_key_from_world_center(player_pose, center)
    basis_name = "minecraft"
    visible_faces = _visible_face_projection_entries(
        player_pose=player_pose,
        target_center=center,
        image_shape=raw_image.shape,
        basis_name=basis_name,
        voxel_key=voxel_key,
        occupied_voxels=occupied_voxels,
    )
    raw_mask = _project_block_face_mask(
        player_pose,
        center,
        raw_image.shape,
        basis_name=basis_name,
        voxel_key=voxel_key,
        occupied_voxels=occupied_voxels,
    )
    if int(raw_mask.sum()) <= 0:
        raise RuntimeError(f"Configured target is not visible for task {task_key}")

    raw_bbox = _mask_bbox(raw_mask)
    if len(visible_faces) == 1:
        face_entry = visible_faces[0]
        polygon_center = face_entry.get("polygon_center")
        face_center_projection = face_entry.get("center_projection")
        if isinstance(polygon_center, dict):
            raw_cx = float(polygon_center["x"])
            raw_cy = float(polygon_center["y"])
            depth = float(face_center_projection["depth"]) if isinstance(face_center_projection, dict) else 0.0
        elif isinstance(face_center_projection, dict):
            raw_cx = float(face_center_projection["x"])
            raw_cy = float(face_center_projection["y"])
            depth = float(face_center_projection["depth"])
        else:
            raw_cx, raw_cy = _mask_center(raw_mask)
            depth = 0.0
    else:
        raw_cx, raw_cy = _mask_prompt_point(raw_mask)
        depth = 0.0
        basis = _get_basis_by_name(player_pose, basis_name)
        if basis is not None:
            forward, right, up = basis
            center_projection = _project_world_point_with_basis(
                np.array(
                    [
                        float(player_pose["x"]),
                        float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
                        float(player_pose["z"]),
                    ],
                    dtype=np.float32,
                ),
                np.asarray(center, dtype=np.float32),
                raw_image.shape,
                forward=forward,
                right=right,
                up=up,
            )
            if center_projection is not None:
                depth = float(center_projection["depth"])

    obs_mask = _resize_binary_mask(raw_mask, obs_image.shape)
    goal_image = np.asarray(obs_image, dtype=np.uint8).copy()
    if isinstance(raw_image, np.ndarray) and raw_image.ndim == 3 and isinstance(obs_image, np.ndarray):
        try:
            goal_image = cv2.resize(
                raw_image.astype(np.uint8),
                dsize=(int(obs_image.shape[1]), int(obs_image.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        except Exception:
            goal_image = np.asarray(obs_image, dtype=np.uint8).copy()
    refinement_metrics = None
    if task_key == "mine_emerald":
        obs_mask, refinement_metrics = _refine_block_mask_image(task_key, goal_image, obs_mask)
    obs_bbox = _mask_bbox(obs_mask)
    raw_mask_bbox = _mask_bbox(raw_mask)
    if refinement_metrics is None:
        refinement_metadata = {
            "mask_refine_applied": False,
        }
    else:
        refinement_metadata = {
            "mask_refine_applied": True,
            "mask_refine_dx": int(refinement_metrics["best_dx"]),
            "mask_refine_dy": int(refinement_metrics["best_dy"]),
            "mask_refine_scale": round(float(refinement_metrics["best_scale"]), 4),
            "mask_refine_score_before": round(float(refinement_metrics["base_refine_score"]), 4),
            "mask_refine_score_after": round(float(refinement_metrics["refine_score"]), 4),
            "mask_refine_signal_sum": round(float(refinement_metrics["signal_sum"]), 4),
            "mask_refine_signal_frac": round(float(refinement_metrics["signal_frac"]), 6),
            "mask_refine_ring_signal_sum": round(float(refinement_metrics["ring_signal_sum"]), 4),
            "mask_refine_ring_signal_frac": round(float(refinement_metrics["ring_signal_frac"]), 6),
        }

    normalized_target_type = _normalize_name(target_type)
    if not normalized_target_type:
        normalized_target_type = str(AUTO_GOAL_BLOCK_TYPES[task_key][0])

    return {
        "goal_image": goal_image,
        "goal_mask": obs_mask,
        "raw_mask": raw_mask.copy(),
        "segment_type": "Mine",
        "_debug_raw_mask": raw_mask.copy(),
        "metadata": {
            "goal_mode": "auto_current_view",
            "goal_source": "configured",
            "task_key": task_key,
            "target_type": normalized_target_type,
            "voxel_key": [int(voxel_key[0]), int(voxel_key[1]), int(voxel_key[2])],
            "base_target_world_center": [round(float(value), 4) for value in center],
            "target_world_center": [round(float(value), 4) for value in center],
            "target_center_raw": {
                "x": round(float(raw_cx), 3),
                "y": round(float(raw_cy), 3),
                "depth": round(float(depth), 3),
            },
            "bbox_raw": list(raw_bbox),
            "mask_bbox_raw": list(raw_mask_bbox),
            "bbox_obs": list(obs_bbox),
            "projection_basis": basis_name,
            "matching_voxels": 1,
            "projected_voxels": 1,
            "mask_source": "projected_visible_faces",
            "goal_image_source_buffer": "raw_pov_resized",
            "world_refine_dx": 0.0,
            "world_refine_dy": 0.0,
            "world_refine_dz": 0.0,
            "world_refine_score": 0.0,
            "num_visible_faces": int(len(visible_faces)),
            "visible_faces": _visible_face_metadata(visible_faces),
            **refinement_metadata,
        },
    }


def _synthesize_mob_goal(task_key: str, obs_image: np.ndarray, raw_image: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
    target_types = AUTO_GOAL_MOB_TYPES[task_key]
    player_pose = _player_pose(info)
    mobs = _extract_mobs(info)
    if not mobs:
        raise RuntimeError(f"mobs missing or malformed for auto-goal task {task_key}")
    best = None
    matching_mobs = 0
    projected_mobs = 0
    for mob in mobs:
        name = _normalize_name(mob.get("name"))
        if not any(target_type in name for target_type in target_types):
            continue
        matching_mobs += 1
        base_x = float(mob.get("x", 0.0))
        base_y = float(mob.get("y", 0.0))
        base_z = float(mob.get("z", 0.0))
        center = (base_x, base_y + 0.7, base_z)
        projected = _project_bbox(player_pose, _entity_corners(center), raw_image.shape)
        if projected is None:
            continue
        projected_mobs += 1
        raw_bbox, raw_center = projected
        score = _score_candidate(raw_center, raw_image.shape)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "name": name,
                "world_center": center,
                "raw_bbox": raw_bbox,
                "raw_center": raw_center,
                "basis_name": str(raw_center.get("basis_name", "")),
            }
    if best is None:
        raise RuntimeError(
            f"Failed to synthesize goal for {task_key}: no visible matching mob found "
            f"(matching_mobs={matching_mobs}, projected_mobs={projected_mobs}, "
            f"player_yaw={player_pose['yaw']:.3f}, player_pitch={player_pose['pitch']:.3f})"
        )
    obs_bbox = _scale_bbox(best["raw_bbox"], raw_image.shape, obs_image.shape)
    return {
        "goal_image": np.asarray(obs_image, dtype=np.uint8).copy(),
        "goal_mask": _make_bbox_mask(obs_image.shape, obs_bbox),
        "segment_type": "Hunt",
        "_debug_raw_mask": None,
        "metadata": {
            "goal_mode": "auto_current_view",
            "goal_source": "mobs",
            "task_key": task_key,
            "target_type": best["name"],
            "target_world_center": [round(float(value), 4) for value in best["world_center"]],
            "target_center_raw": {
                "x": round(float(best["raw_center"]["x"]), 3),
                "y": round(float(best["raw_center"]["y"]), 3),
                "depth": round(float(best["raw_center"]["depth"]), 3),
            },
            "bbox_raw": list(best["raw_bbox"]),
            "bbox_obs": list(obs_bbox),
            "projection_basis": best["basis_name"],
            "matching_mobs": int(matching_mobs),
            "projected_mobs": int(projected_mobs),
        },
    }


def synthesize_auto_goal(task_spec: Any, obs_image: np.ndarray, raw_image: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
    task_key = _task_key(task_spec)
    if task_key in AUTO_GOAL_BLOCK_TYPES:
        return _synthesize_block_goal(task_key, obs_image=obs_image, raw_image=raw_image, info=info)
    if task_key in AUTO_GOAL_MOB_TYPES:
        return _synthesize_mob_goal(task_key, obs_image=obs_image, raw_image=raw_image, info=info)
    supported = ", ".join(sorted(AUTO_GOAL_SUPPORTED_TASKS))
    raise NotImplementedError(f"Auto goal synthesis is not implemented for task '{task_key}'. Supported tasks: {supported}")


def synthesize_centerbox_goal_from_pose(task_spec: Any, obs_image: np.ndarray, pose_candidate: Dict[str, Any]) -> Dict[str, Any]:
    task_key = _task_key(task_spec)
    height, width = int(obs_image.shape[0]), int(obs_image.shape[1])
    segment_type = "Mine" if task_key in AUTO_GOAL_BLOCK_TYPES else "Hunt"
    distance = max(1.0, float(pose_candidate.get("sampled_distance", 3.0)))
    if segment_type == "Mine":
        box_w = int(round(np.clip(width * (0.8 / distance), 28, 96)))
        box_h = int(round(np.clip(height * (0.8 / distance), 28, 96)))
    else:
        box_w = int(round(np.clip(width * (1.0 / distance), 36, 120)))
        box_h = int(round(np.clip(height * (1.3 / distance), 42, 140)))
    cx = width // 2
    cy = height // 2
    x0 = max(0, cx - box_w // 2)
    y0 = max(0, cy - box_h // 2)
    x1 = min(width - 1, cx + box_w // 2)
    y1 = min(height - 1, cy + box_h // 2)
    return {
        "goal_image": np.asarray(obs_image, dtype=np.uint8).copy(),
        "goal_mask": _make_bbox_mask(obs_image.shape, (x0, y0, x1, y1)),
        "segment_type": segment_type,
        "metadata": {
            "goal_mode": "auto_pose_centerbox_cross_view",
            "goal_source": "pose_centerbox_fallback",
            "task_key": task_key,
            "target_type": str(pose_candidate.get("target_type", "")),
            "target_world_center": [round(float(value), 4) for value in pose_candidate["target_world_center"]],
            "target_center_raw": {
                "x": round(float(640 / 2.0), 3),
                "y": round(float(360 / 2.0), 3),
                "depth": round(float(distance), 3),
            },
            "bbox_obs": [int(x0), int(y0), int(x1), int(y1)],
            "fallback_reason": "projection_failed_after_pose_aim",
        },
    }


def _goal_candidate_score(synthesis: Dict[str, Any], *, prefer_alternate_view: bool) -> float:
    metadata = synthesis.get("metadata") or {}
    bbox_obs = metadata.get("bbox_obs") or [0, 0, 0, 0]
    if len(bbox_obs) != 4:
        bbox_obs = [0, 0, 0, 0]
    x0, y0, x1, y1 = [int(v) for v in bbox_obs]
    area = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    center = metadata.get("target_center_raw") or {}
    cx = float(center.get("x", 320.0))
    cy = float(center.get("y", 180.0))
    center_penalty = abs(cx - 320.0) + abs(cy - 180.0)
    alternate_bonus = 5000.0 if prefer_alternate_view else 0.0
    camera_displacement = float(metadata.get("camera_displacement", 0.0))
    yaw_delta = float(metadata.get("yaw_delta_deg", 0.0))
    pitch_delta = float(metadata.get("pitch_delta_deg", 0.0))
    changed_pixel_frac = float(metadata.get("changed_pixel_frac", 0.0))
    image_mean = float(metadata.get("image_mean", 0.0))
    bbox_mean = float(metadata.get("bbox_mean", 0.0))
    bbox_height_px = float(metadata.get("bbox_height_px", 0.0))
    bbox_width_px = float(metadata.get("bbox_width_px", 0.0))
    bbox_area_frac = float(metadata.get("bbox_area_frac", 0.0))
    sampled_distance = float(metadata.get("sampled_distance", 0.0))
    pose_family = str(metadata.get("pose_family") or "")
    selection_bias = float(metadata.get("selection_bias", 0.0))
    probe_name = str(metadata.get("probe_name") or "")
    collision_penalty = 1200.0 if bool(metadata.get("collision_warning", False)) else 0.0
    distance_penalty = max(0.0, sampled_distance - 3.0) * 900.0
    oblique_penalty = 0.0
    if pose_family.startswith("oblique") or probe_name in {"front_left", "front_right"}:
        oblique_penalty = 900.0
        if bbox_area_frac >= 0.09:
            oblique_penalty *= 0.35
    return (
        float(area)
        - 0.35 * float(center_penalty)
        + alternate_bonus
        + 50.0 * camera_displacement
        + 4.0 * yaw_delta
        + 6.0 * pitch_delta
        + 2000.0 * changed_pixel_frac
        + 8.0 * image_mean
        + 10.0 * bbox_mean
        + 25.0 * bbox_height_px
        + 10.0 * bbox_width_px
        + 8000.0 * bbox_area_frac
        + selection_bias
        - collision_penalty
        - distance_penalty
        - oblique_penalty
    )


def _sample_goal_view_candidates(task_spec: Any, info: Dict[str, Any], task_config: Dict[str, Any] | None = None) -> list[Dict[str, Any]]:
    task_key = _task_key(task_spec)
    player_pose = _player_pose(info)
    base_y = float(player_pose["y"])
    anchors = _extract_target_anchors(task_key, info, task_config=task_config)
    candidates = []
    for anchor_index, anchor in enumerate(anchors):
        target_center = anchor["world_center"]
        for distance in AUTO_GOAL_DISTANCE_SAMPLES:
            for azimuth_deg in AUTO_GOAL_AZIMUTH_DEG_SAMPLES:
                azimuth_rad = math.radians(float(azimuth_deg))
                for height_offset in AUTO_GOAL_HEIGHT_SAMPLES:
                    requested_height_offset = float(height_offset)
                    sampled_height_offset = _snap_support_height_offset(requested_height_offset)
                    camera_position = (
                        float(target_center[0]) + float(distance) * math.cos(azimuth_rad),
                        float(base_y) + float(sampled_height_offset),
                        float(target_center[2]) + float(distance) * math.sin(azimuth_rad),
                    )
                    base_yaw, base_pitch = _look_angles_for_target(camera_position, target_center)
                    support_block = _support_block_for_pose(
                        {
                            "x": float(camera_position[0]),
                            "y": float(camera_position[1]),
                            "z": float(camera_position[2]),
                        }
                    )
                    for pitch_offset in AUTO_GOAL_PITCH_OFFSET_SAMPLES:
                        candidates.append(
                            {
                                "anchor_index": int(anchor_index),
                                "goal_source": anchor["goal_source"],
                                "target_type": anchor["target_type"],
                                "target_world_center": target_center,
                                "preferred_face": str(anchor.get("preferred_face") or "").strip(),
                                "camera_position": camera_position,
                                "sampled_distance": float(distance),
                                "sampled_azimuth_deg": float(azimuth_deg),
                                "requested_height_offset": requested_height_offset,
                                "sampled_height_offset": float(sampled_height_offset),
                                "sampled_pitch_offset": float(pitch_offset),
                                "yaw": float(base_yaw),
                                "pitch": float(base_pitch + float(pitch_offset)),
                                "support_block": support_block,
                            }
                        )
    return candidates


class CrossViewSession:
    def __init__(
        self,
        model_path: str,
        name_file_mapping: dict,
        goal_image_path: Optional[str] = None,
        goal_mask_path: Optional[str] = None,
        goal_segment_type: Optional[str] = None,
        cfg_coef: float = 1.5,
        cfg_policy_mode: str = "full",
        cfg_base_ref_model_path: str = "",
        obs_size: Tuple[int, int] = (224, 224),
        auto_goal_task_spec: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.name_file_mapping = name_file_mapping
        self.cfg_coef = float(cfg_coef)
        self.cfg_policy_mode = str(cfg_policy_mode or "full").strip() or "full"
        if self.cfg_policy_mode not in {"full", "frozen_base"}:
            raise ValueError(f"Unsupported cfg_policy_mode: {self.cfg_policy_mode}")
        self.cfg_base_ref_model_path = str(cfg_base_ref_model_path or "").strip()
        self.obs_size = tuple(int(x) for x in obs_size)
        self.auto_goal_task_spec = auto_goal_task_spec
        self.image_history = []
        self.world_seed = 0
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_policy_action = None
        self.last_policy_logprob = None
        self.last_policy_value = None
        self.last_policy_value_raw = None
        self.last_memory_in = None
        self.last_model_input = None
        self.last_segment_area = 0
        self.last_action_summary = "none"
        self.num_steps = 0
        self.current_task_config = {}
        self.current_goal_metadata: Dict[str, Any] = {}
        self._fixed_goal_metadata: Dict[str, Any] = {}
        self._auto_goal_cache: Dict[tuple, Dict[str, Any]] = {}
        self.debug_assets: Dict[str, Any] = {}
        self._goal_image_path = str(goal_image_path or "")
        self._goal_mask_path = str(goal_mask_path or "")
        if goal_image_path and goal_mask_path and goal_segment_type:
            self.goal_segment_type = str(goal_segment_type)
            self.goal_image_224, self.goal_mask_224 = load_goal_assets(goal_image_path, goal_mask_path, obs_size=self.obs_size)
            self.current_goal_metadata = {
                "goal_mode": "fixed_asset",
                "goal_image_path": self._goal_image_path,
                "goal_mask_path": self._goal_mask_path,
                "segment_type": self.goal_segment_type,
            }
            self._fixed_goal_metadata = copy.deepcopy(self.current_goal_metadata)
        else:
            inferred_segment_type = (
                str(goal_segment_type)
                if goal_segment_type
                else str(auto_goal_task_spec.subtasks[0].interaction_type if auto_goal_task_spec and getattr(auto_goal_task_spec, "subtasks", None) else "None")
            )
            self.goal_segment_type = inferred_segment_type
            self.goal_image_224 = np.zeros((self.obs_size[1], self.obs_size[0], 3), dtype=np.uint8)
            self.goal_mask_224 = np.zeros((self.obs_size[1], self.obs_size[0]), dtype=np.uint8)
        self.current_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self._goal_callback = GoalConditionCallback(self.goal_image_224, self.goal_mask_224, self.goal_segment_type)
        self._reset_debug_assets()

    def _reset_debug_assets(self):
        self.debug_assets = {
            "probe": {},
            "rollout": {},
            "metadata": {},
            "selected_pose": {},
            "candidate_debug": [],
        }

    def _auto_goal_cache_key(self, env_name: str, warmup_noop_steps: int) -> tuple:
        return (
            str(Path(self.name_file_mapping[env_name]).resolve()),
            int(max(0, warmup_noop_steps)),
        )

    def _load_cached_auto_goal(self, env_name: str, warmup_noop_steps: int) -> Optional[Dict[str, Any]]:
        cached = self._auto_goal_cache.get(self._auto_goal_cache_key(env_name, warmup_noop_steps))
        if not isinstance(cached, dict):
            return None
        return {
            "goal_image": np.asarray(cached["goal_image"], dtype=np.uint8).copy(),
            "goal_mask": np.asarray(cached["goal_mask"], dtype=np.uint8).copy(),
            "segment_type": str(cached["segment_type"]),
            "metadata": copy.deepcopy(cached.get("metadata") or {}),
        }

    def _store_cached_auto_goal(self, env_name: str, warmup_noop_steps: int, synthesis: Dict[str, Any]) -> None:
        self._auto_goal_cache[self._auto_goal_cache_key(env_name, warmup_noop_steps)] = {
            "goal_image": np.asarray(synthesis["goal_image"], dtype=np.uint8).copy(),
            "goal_mask": (np.asarray(synthesis["goal_mask"]) > 0).astype(np.uint8),
            "segment_type": str(synthesis["segment_type"]),
            "metadata": copy.deepcopy(synthesis.get("metadata") or {}),
        }

    def _store_debug_snapshot(self, bucket: str, name: str, obs: Optional[Dict[str, Any]], info: Optional[Dict[str, Any]]):
        bucket_data = self.debug_assets.setdefault(bucket, {})
        if obs is not None and isinstance(obs, dict) and "image" in obs:
            try:
                bucket_data[f"{name}_obs"] = np.asarray(obs["image"], dtype=np.uint8).copy()
            except Exception:
                pass
        if info is not None and isinstance(info, dict) and "pov" in info:
            try:
                bucket_data[f"{name}_pov"] = np.asarray(info["pov"], dtype=np.uint8).copy()
            except Exception:
                pass
        if info is not None and isinstance(info, dict) and "player_pos" in info:
            bucket_data[f"{name}_player_pos"] = _sanitize_debug_value(info.get("player_pos"))
        if info is not None and isinstance(info, dict):
            bucket_data[f"{name}_mainhand_type"] = _mainhand_type_from_info(info)
            bucket_data[f"{name}_inventory_totals"] = _inventory_totals_from_info(info)

    def _append_rollout_reset_trace(
        self,
        stage: str,
        *,
        attempt: Optional[int] = None,
        info: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = self.debug_assets.setdefault("metadata", {})
        trace = metadata.setdefault("rollout_reset_trace", [])
        entry: Dict[str, Any] = {"stage": str(stage)}
        if attempt is not None:
            entry["attempt"] = int(attempt)
        player_pose = _safe_player_pose(info)
        if player_pose is not None:
            entry["player_pose"] = _sanitize_debug_value(player_pose)
        mainhand_type = _mainhand_type_from_info(info)
        if mainhand_type:
            entry["mainhand_type"] = str(mainhand_type)
        inventory_totals = _inventory_totals_from_info(info)
        if inventory_totals:
            entry["inventory_totals"] = _sanitize_debug_value(inventory_totals)
        if extra:
            entry.update(_sanitize_debug_value(extra))
        trace.append(_sanitize_debug_value(entry))

    def _store_pose_candidate_debug(
        self,
        *,
        candidate_index: int,
        obs_image: np.ndarray,
        raw_pov: Optional[np.ndarray],
        synthesis: Optional[Dict[str, Any]],
        pose_candidate: Dict[str, Any],
        candidate_pose: Dict[str, float],
        decision: str,
        rejection_reason: Optional[str],
        crossview_metrics: Dict[str, float],
        appearance_metrics: Optional[Dict[str, float]],
    ):
        candidate_debug = self.debug_assets.setdefault("candidate_debug", [])
        if len(candidate_debug) >= int(AUTO_GOAL_MAX_DEBUG_CANDIDATES):
            return
        entry: Dict[str, Any] = {
            "candidate_index": int(candidate_index),
            "decision": str(decision),
            "rejection_reason": str(rejection_reason or ""),
            "pose_candidate": _sanitize_debug_value(pose_candidate),
            "candidate_pose": _sanitize_debug_value(candidate_pose),
            "crossview_metrics": _sanitize_debug_value(crossview_metrics),
            "appearance_metrics": _sanitize_debug_value(appearance_metrics or {}),
            "candidate_obs": np.asarray(obs_image, dtype=np.uint8).copy(),
        }
        if isinstance(raw_pov, np.ndarray):
            entry["candidate_pov"] = np.asarray(raw_pov, dtype=np.uint8).copy()
        if synthesis is not None:
            synthesis_metadata = dict(synthesis.get("metadata") or {})
            entry["goal_metadata"] = _sanitize_debug_value(synthesis_metadata)
            goal_image = synthesis.get("goal_image")
            if isinstance(goal_image, np.ndarray):
                entry["goal_image"] = np.asarray(goal_image, dtype=np.uint8).copy()
            goal_mask = synthesis.get("goal_mask")
            if isinstance(goal_mask, np.ndarray):
                entry["goal_mask"] = np.asarray(goal_mask, dtype=np.uint8).copy()
            raw_mask = synthesis.get("raw_mask")
            if not isinstance(raw_mask, np.ndarray):
                raw_mask = synthesis.get("_debug_raw_mask")
            if isinstance(raw_mask, np.ndarray):
                entry["raw_mask"] = np.asarray(raw_mask, dtype=np.uint8).copy()
        candidate_debug.append(entry)

    def clear_agent_memory(self, reset_counters: bool = True):
        if reset_counters:
            self.num_steps = 0
            self.last_action_summary = "none"
        if hasattr(self, "agent"):
            self.state = self.agent.initial_state()
        self.last_policy_action = None
        self.last_policy_logprob = None
        self.last_policy_value = None
        self.last_policy_value_raw = None
        self.last_memory_in = None
        self.last_model_input = None

    def summarize_agent_action(self, action):
        try:
            action_copy = copy.deepcopy(action)
            env_action = self.env.agent_action_to_env_action(action_copy)
            active_buttons = [key for key, value in env_action.items() if key != "camera" and value == 1]
            camera = np.asarray(env_action.get("camera", np.array([0, 0]))).tolist()
            if not active_buttons:
                active_buttons = ["noop"]
            return f"buttons={'+'.join(active_buttons)}, camera={camera}"
        except Exception as exc:
            return f"unavailable ({exc})"

    def _detach_action_for_logging(self, action):
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().clone()
        if isinstance(action, dict):
            return {key: self._detach_action_for_logging(value) for key, value in action.items()}
        if isinstance(action, list):
            return [self._detach_action_for_logging(value) for value in action]
        if isinstance(action, tuple):
            return tuple(self._detach_action_for_logging(value) for value in action)
        return action

    def _detach_state_for_logging(self, state):
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, list):
            return [self._detach_state_for_logging(value) for value in state]
        if isinstance(state, tuple):
            return tuple(self._detach_state_for_logging(value) for value in state)
        if isinstance(state, dict):
            return {key: self._detach_state_for_logging(value) for key, value in state.items()}
        return state

    @staticmethod
    def _load_cross_view_agent_from_source(model_path: str):
        if model_path.startswith("hf:"):
            model_id = model_path.split(":", 1)[1]
            agent = CrossViewRocket.from_pretrained(model_id)
        elif Path(model_path).exists():
            agent = load_cross_view_rocket(model_path)
        else:
            agent = CrossViewRocket.from_pretrained(model_path)
        agent = agent.to("cuda")
        agent.eval()
        return agent

    def _load_agent(self):
        agent = self._load_cross_view_agent_from_source(self.model_path)
        base_agent = None
        if self.cfg_policy_mode == "frozen_base" and self.cfg_coef > 0.0:
            base_model_path = self.cfg_base_ref_model_path or self.model_path
            if str(base_model_path) == str(self.model_path):
                base_agent = agent
            else:
                base_agent = self._load_cross_view_agent_from_source(base_model_path)
            for param in base_agent.parameters():
                param.requires_grad = False
        return CFGWrapper(agent, k=self.cfg_coef, base_model=base_agent)

    def _build_callbacks(self, task_config: Dict[str, Any], *, include_prev_action: bool, include_goal_callback: bool):
        callbacks = list(load_callbacks_from_config(task_config))
        if include_prev_action and not any(isinstance(callback, PrevActionCallback) for callback in callbacks):
            callbacks.append(PrevActionCallback())
        if self.auto_goal_task_spec is not None:
            task_key = _task_key(self.auto_goal_task_spec)
            if task_key in AUTO_GOAL_BLOCK_TYPES and not any(isinstance(callback, VoxelsCallback) for callback in callbacks):
                callbacks.append(VoxelsCallback(AUTO_GOAL_VOXEL_BOUNDS))
            if task_key in AUTO_GOAL_MOB_TYPES and not any(isinstance(callback, NearbyMobsCallback) for callback in callbacks):
                callbacks.append(NearbyMobsCallback(AUTO_GOAL_MOBS_BOUNDS))
        if include_goal_callback:
            callbacks.append(self._goal_callback)
        return callbacks

    def _run_probe_sweep(self, env_name: str, warmup_noop_steps: int) -> Dict[str, Any]:
        if self.auto_goal_task_spec is None:
            raise RuntimeError("Probe sweep requested without auto_goal_task_spec")
        callbacks = self._build_callbacks(self.current_task_config, include_prev_action=False, include_goal_callback=False)
        probe_env = None
        candidates = []
        info: Dict[str, Any] = {}
        pose_candidates = []
        initial_anchor_count = 0
        pose_success_count = 0
        pose_fallback_count = 0
        sweep_success_count = 0
        pose_rejected_similarity_count = 0
        sweep_rejected_similarity_count = 0
        pose_rejected_appearance_count = 0
        sweep_rejected_appearance_count = 0
        pose_rejected_collision_count = 0
        sweep_rejected_collision_count = 0

        def _make_probe_env():
            return MinecraftSim(
                seed=self.world_seed,
                preferred_spawn_biome="plains",
                callbacks=callbacks,
                obs_size=self.obs_size,
                action_type="env",
            )

        try:
            last_reset_exc = None
            for reset_attempt in range(1, int(AUTO_GOAL_PROBE_RESET_RETRIES) + 1):
                try:
                    if probe_env is not None:
                        try:
                            probe_env.close()
                        except Exception:
                            pass
                    probe_env = _make_probe_env()
                    obs, info = probe_env.reset()
                    break
                except Exception as exc:
                    last_reset_exc = exc
                    print(
                        "[crossview-auto-goal] "
                        f"probe reset failed attempt={reset_attempt}/{int(AUTO_GOAL_PROBE_RESET_RETRIES)} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    try:
                        if probe_env is not None:
                            probe_env.close()
                    except Exception:
                        pass
                    probe_env = None
                    if reset_attempt >= int(AUTO_GOAL_PROBE_RESET_RETRIES):
                        raise
            if probe_env is None:
                raise last_reset_exc or RuntimeError("Probe environment failed to initialize")
            self._store_debug_snapshot("probe", "probe_reset", obs, info)
            for _ in range(int(max(0, warmup_noop_steps))):
                obs, _, _, _, info = probe_env.step(probe_env.noop_action())
            self._store_debug_snapshot("probe", "probe_after_warmup", obs, info)
            task_key = _task_key(self.auto_goal_task_spec)
            configured_entries = _configured_target_entries(task_key, self.current_task_config)
            configured_probe_target = None
            if task_key in AUTO_GOAL_BLOCK_TYPES and len(configured_entries) == 1:
                configured_probe_target = configured_entries[0]
            reference_pose = _player_pose(info)
            reference_obs = np.asarray(obs["image"], dtype=np.uint8).copy()
            reference_collision_info = info
            self.debug_assets["metadata"]["reference_probe_pose"] = _sanitize_debug_value(reference_pose)

            def probe_pose_candidates(pose_candidates_to_try, *, pose_name_prefix: str):
                nonlocal pose_success_count
                nonlocal pose_fallback_count
                nonlocal pose_rejected_similarity_count
                nonlocal pose_rejected_appearance_count
                nonlocal pose_rejected_collision_count
                nonlocal obs
                nonlocal info
                total_pose_candidates = len(pose_candidates_to_try)
                for pose_index, pose_candidate in enumerate(pose_candidates_to_try, start=1):
                    pose_candidate = _adjust_pose_candidate_for_clearance(pose_candidate, reference_collision_info)
                    probe_name = str(pose_candidate.get("hint_name") or f"{pose_name_prefix}_{pose_index:03d}")
                    is_hint_candidate = str(pose_candidate.get("goal_source") or "").strip().lower() == "hint"
                    print(
                        "[crossview-auto-goal] "
                        f"pose={pose_index:02d}/{total_pose_candidates:02d} probing name={probe_name}",
                        flush=True,
                    )
                    cam_x, cam_y, cam_z = pose_candidate["camera_position"]
                    yaw = pose_candidate["yaw"]
                    pitch = pose_candidate["pitch"]
                    expected_pose = {
                        "x": float(cam_x),
                        "y": float(cam_y),
                        "z": float(cam_z),
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                    }
                    try:
                        _ensure_support_block(probe_env, pose_candidate.get("support_block"))
                        _teleport_to_pose(probe_env, expected_pose)
                    except Exception:
                        continue
                    obs, info, terminated, truncated, capture_diagnostics = _capture_stable_frame(
                        probe_env,
                        expected_pose,
                        min_wait_steps=max(1, int(AUTO_GOAL_POST_TP_SETTLE_STEPS)),
                        max_wait_steps=max(8, int(AUTO_GOAL_POST_TP_SETTLE_STEPS) + 8),
                    )
                    if obs is None or info is None:
                        continue
                    if pose_index == 1 and not candidates:
                        self._store_debug_snapshot("probe", "probe_first_pose", obs, info)
                    candidate_obs = np.asarray(obs["image"], dtype=np.uint8)
                    candidate_pose = _player_pose(info)
                    crossview_metrics = _crossview_difference_metrics(reference_pose, reference_obs, candidate_pose, candidate_obs)
                    preview_synthesis = None
                    preview_appearance_metrics = None
                    try:
                        if configured_probe_target is not None:
                            preview_synthesis = _synthesize_configured_block_goal(
                                task_key=task_key,
                                obs_image=candidate_obs,
                                raw_image=np.asarray(info["pov"], dtype=np.uint8),
                                info=info,
                                target_world_center=pose_candidate["target_world_center"],
                                target_type=str(
                                    pose_candidate.get("target_type") or configured_probe_target.get("target_type") or ""
                                ),
                            )
                        else:
                            preview_synthesis = synthesize_auto_goal(
                                task_spec=self.auto_goal_task_spec,
                                obs_image=candidate_obs,
                                raw_image=np.asarray(info["pov"], dtype=np.uint8),
                                info=info,
                            )
                    except Exception:
                        preview_synthesis = synthesize_centerbox_goal_from_pose(
                            task_spec=self.auto_goal_task_spec,
                            obs_image=candidate_obs,
                            pose_candidate=pose_candidate,
                        )
                    if preview_synthesis is not None:
                        preview_synthesis = dict(preview_synthesis)
                        preview_synthesis["metadata"] = dict(preview_synthesis.get("metadata") or {})
                        preview_appearance_metrics = _goal_appearance_metrics(
                            candidate_obs,
                            preview_synthesis["metadata"].get("bbox_obs") or [0, 0, 0, 0],
                        )
                    collision_warning = _camera_collides_with_nearby_voxel(
                        info,
                        (candidate_pose["x"], candidate_pose["y"] + DEFAULT_EYE_HEIGHT, candidate_pose["z"]),
                    )
                    if collision_warning:
                        pose_rejected_collision_count += 1
                        if _should_reject_goal_collision_warning(task_key, collision_warning):
                            print(
                                "[crossview-auto-goal] "
                                f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=camera_collision_warning",
                                flush=True,
                            )
                            self._store_pose_candidate_debug(
                                candidate_index=pose_index,
                                obs_image=candidate_obs,
                                raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                                synthesis=preview_synthesis,
                                pose_candidate=pose_candidate,
                                candidate_pose=candidate_pose,
                                decision="rejected_collision",
                                rejection_reason="camera_collision_warning",
                                crossview_metrics=crossview_metrics,
                                appearance_metrics=preview_appearance_metrics,
                            )
                            continue
                    if (not is_hint_candidate) and (not _is_valid_crossview_candidate(crossview_metrics)):
                        pose_rejected_similarity_count += 1
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=crossview_difference_below_threshold",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=preview_synthesis,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_similarity",
                            rejection_reason="crossview_difference_below_threshold",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=preview_appearance_metrics,
                        )
                        continue
                    if preview_synthesis is None:
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=auto_goal_synthesis_failed",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=None,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_no_synthesis",
                            rejection_reason="auto_goal_synthesis_failed",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=preview_appearance_metrics,
                        )
                        continue
                    synthesis = preview_synthesis
                    appearance_metrics = preview_appearance_metrics or {}
                    if synthesis["metadata"].get("goal_source") == "pose_centerbox_fallback":
                        pose_fallback_count += 1
                    if not _is_valid_goal_appearance(task_key, appearance_metrics):
                        pose_rejected_appearance_count += 1
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=goal_appearance_below_threshold",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=synthesis,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_appearance",
                            rejection_reason="goal_appearance_below_threshold",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=appearance_metrics,
                        )
                        continue
                    target_visibility_metrics = _goal_target_visibility_metrics(
                        task_key,
                        np.asarray(synthesis.get("goal_image")),
                        synthesis.get("goal_mask"),
                    )
                    _attach_goal_target_visibility_metadata(synthesis, target_visibility_metrics)
                    preferred_face = str(
                        pose_candidate.get("preferred_face") or (configured_probe_target or {}).get("preferred_face") or ""
                    ).strip()
                    procedural_layout = (
                        self.current_task_config.get("procedural_layout")
                        if isinstance(self.current_task_config.get("procedural_layout"), dict)
                        else {}
                    )
                    blueprint_id = str(procedural_layout.get("blueprint_id") or "").strip()
                    pose_family = str(pose_candidate.get("pose_family") or "").strip()
                    if not _is_valid_goal_face_visibility(
                        task_key,
                        synthesis,
                        preferred_face=preferred_face,
                        blueprint_id=blueprint_id,
                        pose_family=pose_family,
                    ):
                        pose_rejected_appearance_count += 1
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=preferred_face_not_visible",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=synthesis,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_appearance",
                            rejection_reason="preferred_face_not_visible",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=appearance_metrics,
                        )
                        continue
                    if not _is_valid_goal_target_visibility(
                        task_key,
                        synthesis,
                        visibility_metrics=target_visibility_metrics,
                    ):
                        pose_rejected_appearance_count += 1
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=goal_target_visibility_below_threshold",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=synthesis,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_appearance",
                            rejection_reason="goal_target_visibility_below_threshold",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=appearance_metrics,
                        )
                        continue
                    projection_raw_mask = synthesis.get("raw_mask")
                    if not isinstance(projection_raw_mask, np.ndarray):
                        projection_raw_mask = synthesis.get("_debug_raw_mask")
                    projection_consistency_metrics = _goal_projection_consistency_metrics(
                        task_key,
                        np.asarray(info["pov"], dtype=np.uint8),
                        projection_raw_mask,
                    )
                    _attach_goal_projection_consistency_metadata(synthesis, projection_consistency_metrics)
                    if not _is_valid_goal_projection_consistency(
                        task_key,
                        synthesis,
                        consistency_metrics=projection_consistency_metrics,
                    ):
                        pose_rejected_appearance_count += 1
                        print(
                            "[crossview-auto-goal] "
                            f"pose={pose_index:02d}/{total_pose_candidates:02d} rejected=projection_image_mismatch",
                            flush=True,
                        )
                        self._store_pose_candidate_debug(
                            candidate_index=pose_index,
                            obs_image=candidate_obs,
                            raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                            synthesis=synthesis,
                            pose_candidate=pose_candidate,
                            candidate_pose=candidate_pose,
                            decision="rejected_appearance",
                            rejection_reason="projection_image_mismatch",
                            crossview_metrics=crossview_metrics,
                            appearance_metrics=appearance_metrics,
                        )
                        continue
                    synthesis["metadata"].update(
                        {
                            "probe_step": int(pose_index),
                            "probe_name": str(probe_name),
                            "probe_camera": [0.0, 0.0],
                            "goal_mode": str(pose_candidate.get("goal_mode_hint") or "auto_pose_sampled_cross_view"),
                            "sampled_camera_position": [round(float(cam_x), 4), round(float(cam_y), 4), round(float(cam_z), 4)],
                            "sampled_yaw": round(float(yaw), 4),
                            "sampled_pitch": round(float(pitch), 4),
                            "sampled_distance": float(pose_candidate["sampled_distance"]),
                            "sampled_center_distance": float(
                                pose_candidate.get("sampled_center_distance", _distance3((cam_x, cam_y, cam_z), pose_candidate["target_world_center"]))
                            ),
                            "sampled_face_distance": pose_candidate.get("sampled_face_distance"),
                            "sampled_azimuth_deg": float(pose_candidate["sampled_azimuth_deg"]),
                            "sampled_height_offset": float(pose_candidate["sampled_height_offset"]),
                            "requested_height_offset": float(
                                pose_candidate.get("requested_height_offset", pose_candidate["sampled_height_offset"])
                            ),
                            "sampled_pitch_offset": float(pose_candidate["sampled_pitch_offset"]),
                            "clearance_backoff": round(float(pose_candidate.get("clearance_backoff", 0.0) or 0.0), 4),
                            "anchor_index": int(pose_candidate["anchor_index"]),
                            "target_type": pose_candidate["target_type"],
                            "target_world_center": [round(float(value), 4) for value in pose_candidate["target_world_center"]],
                            "preferred_face": preferred_face,
                            "pose_family": str(pose_candidate.get("pose_family") or ""),
                            "selection_bias": round(float(pose_candidate.get("selection_bias", 0.0) or 0.0), 4),
                            "support_block": _sanitize_debug_value(pose_candidate.get("support_block")),
                            "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                            "probe_player_pose": _sanitize_debug_value(candidate_pose),
                            "camera_displacement": round(float(crossview_metrics["camera_displacement"]), 4),
                            "yaw_delta_deg": round(float(crossview_metrics["yaw_delta_deg"]), 4),
                            "pitch_delta_deg": round(float(crossview_metrics["pitch_delta_deg"]), 4),
                            "mean_abs_diff": round(float(crossview_metrics["mean_abs_diff"]), 4),
                            "changed_pixel_frac": round(float(crossview_metrics["changed_pixel_frac"]), 6),
                            "image_mean": round(float(appearance_metrics["image_mean"]), 4),
                            "image_std": round(float(appearance_metrics["image_std"]), 4),
                            "bbox_mean": round(float(appearance_metrics["bbox_mean"]), 4),
                            "bbox_std": round(float(appearance_metrics["bbox_std"]), 4),
                            "bbox_width_px": int(appearance_metrics["bbox_width_px"]),
                            "bbox_height_px": int(appearance_metrics["bbox_height_px"]),
                            "bbox_area_frac": round(float(appearance_metrics["bbox_area_frac"]), 6),
                            "collision_warning": bool(collision_warning),
                        }
                    )
                    synthesis["_debug_candidate_index"] = int(pose_index)
                    synthesis["_debug_candidate_obs"] = candidate_obs.copy()
                    synthesis["_debug_candidate_pov"] = np.asarray(info["pov"], dtype=np.uint8).copy()
                    synthesis["_debug_candidate_pose"] = _sanitize_debug_value(candidate_pose)
                    synthesis["_debug_pose_candidate"] = _sanitize_debug_value(pose_candidate)
                    synthesis["_debug_crossview_metrics"] = _sanitize_debug_value(crossview_metrics)
                    synthesis["_debug_appearance_metrics"] = _sanitize_debug_value(appearance_metrics)
                    synthesis["_score"] = _goal_candidate_score(synthesis, prefer_alternate_view=True) + 2500.0
                    print(
                        "[crossview-auto-goal] "
                        f"pose={pose_index:02d}/{total_pose_candidates:02d} accepted "
                        f"bbox_h={int(appearance_metrics['bbox_height_px'])} "
                        f"bbox_w={int(appearance_metrics['bbox_width_px'])}",
                        flush=True,
                    )
                    self._store_pose_candidate_debug(
                        candidate_index=pose_index,
                        obs_image=candidate_obs,
                        raw_pov=np.asarray(info["pov"], dtype=np.uint8),
                        synthesis=synthesis,
                        pose_candidate=pose_candidate,
                        candidate_pose=candidate_pose,
                        decision="accepted",
                        rejection_reason="",
                        crossview_metrics=crossview_metrics,
                        appearance_metrics=appearance_metrics,
                    )
                    candidates.append(synthesis)
                    pose_success_count += 1
                    if int(pose_success_count) >= int(AUTO_GOAL_MAX_POSE_SUCCESSES):
                        print(
                            "[crossview-auto-goal] "
                            f"early-stop pose probing after {pose_success_count} accepted candidates",
                            flush=True,
                        )
                        break

            hint_pose_candidates = _goal_pose_hint_candidates(task_key, reference_pose, self.current_task_config)
            self.debug_assets["metadata"]["goal_pose_hint_count"] = int(len(hint_pose_candidates))
            if hint_pose_candidates:
                pose_candidates = list(hint_pose_candidates[: int(AUTO_GOAL_MAX_POSE_CANDIDATES)])
                initial_anchor_count = 0
                print(
                    "[crossview-auto-goal] "
                    f"task={task_key} "
                    f"anchors={initial_anchor_count} "
                    f"hint_pose_candidates={len(pose_candidates)}"
                )
                probe_pose_candidates(pose_candidates, pose_name_prefix="hint_pose")
            else:
                pose_candidates = []

            if int(pose_success_count) == 0:
                initial_anchor_count = len(_extract_target_anchors(task_key, info, task_config=self.current_task_config))
                pose_candidates = _sample_goal_view_candidates(self.auto_goal_task_spec, info, task_config=self.current_task_config)
                if len(pose_candidates) > AUTO_GOAL_MAX_POSE_CANDIDATES:
                    pose_candidates = pose_candidates[:AUTO_GOAL_MAX_POSE_CANDIDATES]
                print(
                    "[crossview-auto-goal] "
                    f"task={task_key} "
                    f"anchors={initial_anchor_count} "
                    f"pose_candidates={len(pose_candidates)}"
                )
                probe_pose_candidates(pose_candidates, pose_name_prefix="sampled_pose")

            # Start scripted sweep from the original probe reference pose, not the
            # last teleported sampled pose. Otherwise a later "yaw_right_*" frame
            # is measured relative to an arbitrary sampled camera position.
            run_sweep = int(pose_success_count) == 0
            if not run_sweep:
                print(
                    "[crossview-auto-goal] "
                    f"skip_sweep pose_success_count={pose_success_count}",
                    flush=True,
                )
            if run_sweep:
                reset_command = (
                    f"/tp @a {reference_pose['x']:.3f} {reference_pose['y']:.3f} {reference_pose['z']:.3f} "
                    f"{reference_pose['yaw']:.3f} {reference_pose['pitch']:.3f}"
                )
                try:
                    probe_env.env.execute_cmd(reset_command)
                    for _ in range(AUTO_GOAL_POST_TP_SETTLE_STEPS):
                        obs, _, terminated, truncated, info = probe_env.step(probe_env.noop_action())
                        if terminated or truncated:
                            break
                    self._store_debug_snapshot("probe", "probe_before_sweep_reset", obs, info)
                except Exception:
                    pass

            def maybe_add_candidate(probe_step: int, probe_name: str, prefer_alternate_view: bool, probe_camera=None):
                nonlocal sweep_success_count
                nonlocal sweep_rejected_similarity_count
                nonlocal sweep_rejected_appearance_count
                nonlocal sweep_rejected_collision_count
                candidate_obs = np.asarray(obs["image"], dtype=np.uint8)
                candidate_pose = _player_pose(info)
                collision_warning = _camera_collides_with_nearby_voxel(
                    info,
                    (candidate_pose["x"], candidate_pose["y"] + DEFAULT_EYE_HEIGHT, candidate_pose["z"]),
                )
                if collision_warning:
                    sweep_rejected_collision_count += 1
                    if _should_reject_goal_collision_warning(task_key, collision_warning):
                        return False
                crossview_metrics = _crossview_difference_metrics(reference_pose, reference_obs, candidate_pose, candidate_obs)
                if not _is_valid_crossview_candidate(crossview_metrics):
                    sweep_rejected_similarity_count += 1
                    return False
                try:
                    if configured_probe_target is not None:
                        synthesis = _synthesize_configured_block_goal(
                            task_key=task_key,
                            obs_image=candidate_obs,
                            raw_image=np.asarray(info["pov"], dtype=np.uint8),
                            info=info,
                            target_world_center=configured_probe_target["world_center"],
                            target_type=str(configured_probe_target.get("target_type") or ""),
                        )
                    else:
                        synthesis = synthesize_auto_goal(
                            task_spec=self.auto_goal_task_spec,
                            obs_image=candidate_obs,
                            raw_image=np.asarray(info["pov"], dtype=np.uint8),
                            info=info,
                        )
                except Exception:
                    return False
                synthesis = dict(synthesis)
                synthesis["metadata"] = dict(synthesis.get("metadata") or {})
                appearance_metrics = _goal_appearance_metrics(candidate_obs, synthesis["metadata"].get("bbox_obs") or [0, 0, 0, 0])
                if not _is_valid_goal_appearance(task_key, appearance_metrics):
                    sweep_rejected_appearance_count += 1
                    return False
                target_visibility_metrics = _goal_target_visibility_metrics(
                    task_key,
                    np.asarray(synthesis.get("goal_image")),
                    synthesis.get("goal_mask"),
                )
                _attach_goal_target_visibility_metadata(synthesis, target_visibility_metrics)
                preferred_face = str((configured_probe_target or {}).get("preferred_face") or "").strip()
                procedural_layout = (
                    self.current_task_config.get("procedural_layout")
                    if isinstance(self.current_task_config.get("procedural_layout"), dict)
                    else {}
                )
                blueprint_id = str(procedural_layout.get("blueprint_id") or "").strip()
                if not _is_valid_goal_face_visibility(
                    task_key,
                    synthesis,
                    preferred_face=preferred_face,
                    blueprint_id=blueprint_id,
                ):
                    sweep_rejected_appearance_count += 1
                    return False
                if not _is_valid_goal_target_visibility(
                    task_key,
                    synthesis,
                    visibility_metrics=target_visibility_metrics,
                ):
                    sweep_rejected_appearance_count += 1
                    return False
                synthesis["metadata"].update(
                    {
                        "probe_step": int(probe_step),
                        "probe_name": str(probe_name),
                        "probe_camera": probe_camera,
                        "goal_mode": "auto_sweep_cross_view",
                        "probe_player_pose": _sanitize_debug_value(candidate_pose),
                        "camera_displacement": round(float(crossview_metrics["camera_displacement"]), 4),
                        "yaw_delta_deg": round(float(crossview_metrics["yaw_delta_deg"]), 4),
                        "pitch_delta_deg": round(float(crossview_metrics["pitch_delta_deg"]), 4),
                        "mean_abs_diff": round(float(crossview_metrics["mean_abs_diff"]), 4),
                        "changed_pixel_frac": round(float(crossview_metrics["changed_pixel_frac"]), 6),
                        "image_mean": round(float(appearance_metrics["image_mean"]), 4),
                        "image_std": round(float(appearance_metrics["image_std"]), 4),
                        "bbox_mean": round(float(appearance_metrics["bbox_mean"]), 4),
                        "bbox_std": round(float(appearance_metrics["bbox_std"]), 4),
                        "bbox_width_px": int(appearance_metrics["bbox_width_px"]),
                        "bbox_height_px": int(appearance_metrics["bbox_height_px"]),
                        "bbox_area_frac": round(float(appearance_metrics["bbox_area_frac"]), 6),
                        "collision_warning": bool(collision_warning),
                        "preferred_face": preferred_face,
                    }
                )
                synthesis["_score"] = _goal_candidate_score(synthesis, prefer_alternate_view=prefer_alternate_view)
                candidates.append(synthesis)
                sweep_success_count += 1
                return True

            if run_sweep:
                total_sweep_steps = len(AUTO_GOAL_SWEEP_SCRIPT)
                for probe_step, sweep_step in enumerate(AUTO_GOAL_SWEEP_SCRIPT, start=1):
                    print(
                        "[crossview-auto-goal] "
                        f"sweep={probe_step:02d}/{total_sweep_steps:02d} probing name={sweep_step['name']}",
                        flush=True,
                    )
                    action = probe_env.noop_action()
                    action["camera"] = np.asarray(sweep_step["camera"], dtype=np.float32)
                    obs, _, terminated, truncated, info = probe_env.step(action)
                    maybe_add_candidate(
                        probe_step=probe_step,
                        probe_name=str(sweep_step["name"]),
                        prefer_alternate_view=True,
                        probe_camera=[float(sweep_step["camera"][0]), float(sweep_step["camera"][1])],
                    )
                    if terminated or truncated:
                        break
            self._store_debug_snapshot("probe", "probe_last", obs, info)
        finally:
            if probe_env is not None:
                probe_env.close()

        if not candidates:
            task_key = _task_key(self.auto_goal_task_spec)
            self.debug_assets["metadata"]["probe_failure"] = {
                "task_key": task_key,
                "available_targets": _describe_available_targets(task_key, info),
                "anchor_count": int(initial_anchor_count),
                "pose_candidate_count": int(len(pose_candidates)),
                "pose_success_count": int(pose_success_count),
                "pose_fallback_count": int(pose_fallback_count),
                "sweep_success_count": int(sweep_success_count),
                "pose_rejected_similarity_count": int(pose_rejected_similarity_count),
                "sweep_rejected_similarity_count": int(sweep_rejected_similarity_count),
                "pose_rejected_appearance_count": int(pose_rejected_appearance_count),
                "sweep_rejected_appearance_count": int(sweep_rejected_appearance_count),
                "pose_rejected_collision_count": int(pose_rejected_collision_count),
                "sweep_rejected_collision_count": int(sweep_rejected_collision_count),
            }
            raise RuntimeError(
                f"Failed to synthesize any goal candidates for task {task_key}; "
                f"{_describe_available_targets(task_key, info)}; "
                f"anchor_count={initial_anchor_count}; "
                f"pose_candidate_count={len(pose_candidates)}; "
                f"pose_success_count={pose_success_count}; "
                f"pose_fallback_count={pose_fallback_count}; "
                f"sweep_success_count={sweep_success_count}; "
                f"pose_rejected_similarity_count={pose_rejected_similarity_count}; "
                f"sweep_rejected_similarity_count={sweep_rejected_similarity_count}; "
                f"pose_rejected_appearance_count={pose_rejected_appearance_count}; "
                f"sweep_rejected_appearance_count={sweep_rejected_appearance_count}; "
                f"pose_rejected_collision_count={pose_rejected_collision_count}; "
                f"sweep_rejected_collision_count={sweep_rejected_collision_count}"
            )
        alternate_candidates = [candidate for candidate in candidates if int(candidate["metadata"].get("probe_step", 0)) > 0]
        if not alternate_candidates:
            task_key = _task_key(self.auto_goal_task_spec)
            raise RuntimeError(
                f"Failed to synthesize a sufficiently different cross-view goal for task {task_key}; "
                f"anchor_count={initial_anchor_count}; "
                f"pose_candidate_count={len(pose_candidates)}; "
                f"pose_success_count={pose_success_count}; "
                f"pose_fallback_count={pose_fallback_count}; "
                f"sweep_success_count={sweep_success_count}; "
                f"pose_rejected_similarity_count={pose_rejected_similarity_count}; "
                f"sweep_rejected_similarity_count={sweep_rejected_similarity_count}; "
                f"pose_rejected_appearance_count={pose_rejected_appearance_count}; "
                f"sweep_rejected_appearance_count={sweep_rejected_appearance_count}; "
                f"pose_rejected_collision_count={pose_rejected_collision_count}; "
                f"sweep_rejected_collision_count={sweep_rejected_collision_count}"
            )
        best = max(alternate_candidates, key=lambda item: float(item.get("_score", 0.0)))
        best.pop("_score", None)
        best["metadata"]["num_probe_candidates"] = len(candidates)
        best["metadata"]["used_alternate_view"] = bool(int(best["metadata"].get("probe_step", 0)) > 0)
        best["metadata"]["num_pose_candidates"] = len(pose_candidates)
        best["metadata"]["initial_anchor_count"] = int(initial_anchor_count)
        best["metadata"]["pose_success_count"] = int(pose_success_count)
        best["metadata"]["pose_fallback_count"] = int(pose_fallback_count)
        best["metadata"]["sweep_success_count"] = int(sweep_success_count)
        best["metadata"]["pose_rejected_similarity_count"] = int(pose_rejected_similarity_count)
        best["metadata"]["sweep_rejected_similarity_count"] = int(sweep_rejected_similarity_count)
        best["metadata"]["pose_rejected_appearance_count"] = int(pose_rejected_appearance_count)
        best["metadata"]["sweep_rejected_appearance_count"] = int(sweep_rejected_appearance_count)
        best["metadata"]["pose_rejected_collision_count"] = int(pose_rejected_collision_count)
        best["metadata"]["sweep_rejected_collision_count"] = int(sweep_rejected_collision_count)
        self.debug_assets["metadata"]["selected_goal_metadata"] = _sanitize_debug_value(best["metadata"])
        self.debug_assets["selected_pose"] = {
            "candidate_index": int(best.get("_debug_candidate_index", 0)),
            "candidate_obs": np.asarray(best.get("_debug_candidate_obs"), dtype=np.uint8).copy(),
            "candidate_pov": np.asarray(best.get("_debug_candidate_pov"), dtype=np.uint8).copy(),
            "goal_image": np.asarray(best.get("goal_image"), dtype=np.uint8).copy(),
            "goal_mask": (np.asarray(best.get("goal_mask")) > 0).astype(np.uint8),
            "raw_mask": (
                (np.asarray(best.get("_debug_raw_mask")) > 0).astype(np.uint8)
                if isinstance(best.get("_debug_raw_mask"), np.ndarray)
                else None
            ),
            "goal_metadata": _sanitize_debug_value(best.get("metadata") or {}),
            "candidate_pose": _sanitize_debug_value(best.get("_debug_candidate_pose") or {}),
            "pose_candidate": _sanitize_debug_value(best.get("_debug_pose_candidate") or {}),
            "crossview_metrics": _sanitize_debug_value(best.get("_debug_crossview_metrics") or {}),
            "appearance_metrics": _sanitize_debug_value(best.get("_debug_appearance_metrics") or {}),
        }
        return best

    def reset(self, env_name: str, world_seed=None, warmup_noop_steps: int = 30):
        self.image_history = []
        self._reset_debug_assets()
        self.current_goal_metadata = copy.deepcopy(self._fixed_goal_metadata)
        if world_seed not in (None, ""):
            self.world_seed = int(world_seed)
        with open(self.name_file_mapping[env_name], "r", encoding="utf-8") as f:
            task_config = yaml.safe_load(f) or {}
        task_config.pop("reference_video", None)
        self.current_task_config = task_config
        if self.auto_goal_task_spec is not None:
            synthesis = self._load_cached_auto_goal(env_name=env_name, warmup_noop_steps=warmup_noop_steps)
            self.debug_assets["metadata"]["auto_goal_cache_hit"] = bool(synthesis is not None)
            if synthesis is None:
                synthesis = self._run_probe_sweep(env_name=env_name, warmup_noop_steps=warmup_noop_steps)
                self._store_cached_auto_goal(env_name=env_name, warmup_noop_steps=warmup_noop_steps, synthesis=synthesis)
            self.goal_image_224 = np.asarray(synthesis["goal_image"], dtype=np.uint8).copy()
            self.goal_mask_224 = (np.asarray(synthesis["goal_mask"]) > 0).astype(np.uint8)
            self.goal_segment_type = str(synthesis["segment_type"])
            self.current_goal_metadata = dict(synthesis.get("metadata") or {})
            self._goal_callback.update_goal(
                goal_image=self.goal_image_224,
                goal_mask=self.goal_mask_224,
                segment_type=self.goal_segment_type,
            )
        callbacks = self._build_callbacks(task_config, include_prev_action=True, include_goal_callback=True)
        last_reset_exc = None
        self.last_reset_warmup_steps = int(max(0, warmup_noop_steps))
        self.debug_assets["metadata"]["task_config_path"] = str(Path(self.name_file_mapping[env_name]).resolve())
        self.debug_assets["metadata"]["task_config_name"] = str(env_name)
        self.debug_assets["metadata"]["rollout_callback_names"] = [type(callback).__name__ for callback in callbacks]
        rollout_expectations = {
            "expected_spawn_positions": _expected_spawn_positions(task_config),
            "expected_mainhand": _expected_mainhand_from_task_config(task_config),
            "warmup_noop_steps": int(self.last_reset_warmup_steps),
        }
        self.debug_assets["metadata"]["rollout_reset_expectations"] = _sanitize_debug_value(rollout_expectations)
        self._append_rollout_reset_trace("callbacks_built", extra=rollout_expectations)
        for reset_attempt in range(1, int(ROLLOUT_RESET_RETRIES) + 1):
            try:
                reuse_existing_env = bool(
                    hasattr(self, "env") and getattr(self, "_active_env_name", None) == str(env_name)
                )
                if reuse_existing_env:
                    self.env.callbacks = callbacks
                    self.env.seed = self.world_seed
                    try:
                        self.env.env.seed(self.world_seed)
                    except Exception:
                        pass
                else:
                    if hasattr(self, "env"):
                        try:
                            self.env.close()
                        except Exception:
                            pass
                    self.env = MinecraftSim(seed=self.world_seed, preferred_spawn_biome="plains", callbacks=callbacks)
                    self._active_env_name = str(env_name)
                self.debug_assets["metadata"]["rollout_env_reused"] = bool(reuse_existing_env)
                self.obs, self.info = self.env.reset()
                self._store_debug_snapshot("rollout", "rollout_reset", self.obs, self.info)
                reset_report = _rollout_reset_invariant_report(task_config, self.info)
                self._append_rollout_reset_trace(
                    "after_reset",
                    attempt=reset_attempt,
                    info=self.info,
                    extra={"report": reset_report},
                )
                if not bool(reset_report.get("ok", False)):
                    raise RuntimeError(_format_rollout_reset_failure("after_reset", reset_report))
                warmup_terminated = False
                warmup_truncated = False
                warmup_terminal_step = 0
                for _ in range(self.last_reset_warmup_steps):
                    time.sleep(0.1)
                    noop_action = self.env.noop_action()
                    self.obs, self.reward, terminated, truncated, self.info = self.env.step(noop_action)
                    if terminated or truncated:
                        warmup_terminated = bool(terminated)
                        warmup_truncated = bool(truncated)
                        warmup_terminal_step += 1
                        break
                    warmup_terminal_step += 1
                self._store_debug_snapshot("rollout", "rollout_after_warmup", self.obs, self.info)
                warmup_report = _rollout_reset_invariant_report(task_config, self.info)
                self._append_rollout_reset_trace(
                    "after_warmup",
                    attempt=reset_attempt,
                    info=self.info,
                    extra={
                        "report": warmup_report,
                        "warmup_terminated": bool(warmup_terminated),
                        "warmup_truncated": bool(warmup_truncated),
                        "warmup_steps_completed": int(warmup_terminal_step),
                    },
                )
                if warmup_terminated or warmup_truncated:
                    raise RuntimeError(
                        "Rollout reset warmup terminated early: "
                        f"terminated={bool(warmup_terminated)} truncated={bool(warmup_truncated)} "
                        f"step={int(warmup_terminal_step)}"
                    )
                if not bool(warmup_report.get("ok", False)):
                    raise RuntimeError(_format_rollout_reset_failure("after_warmup", warmup_report))
                break
            except Exception as exc:
                last_reset_exc = exc
                self._append_rollout_reset_trace(
                    "attempt_failed",
                    attempt=reset_attempt,
                    info=getattr(self, "info", None),
                    extra={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                print(
                    "[interaction-crossview] "
                    f"rollout reset failed attempt={reset_attempt}/{int(ROLLOUT_RESET_RETRIES)} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                try:
                    if hasattr(self, "env"):
                        self.env.close()
                        del self.env
                except Exception:
                    pass
                if reset_attempt >= int(ROLLOUT_RESET_RETRIES):
                    raise
        if last_reset_exc is not None and not hasattr(self, "env"):
            raise last_reset_exc
        self.debug_assets["metadata"]["rollout_reset_status"] = "ok"
        self.reward = 0.0
        self.agent = self._load_agent()
        self.clear_agent_memory()
        self.current_image = self.info["pov"]
        self.image_history.append(self.current_image.copy())
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        return self.current_image

    def save_goal_assets(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        goal_image_path = output_dir / "goal_image.png"
        goal_mask_path = output_dir / "goal_mask.png"
        goal_bbox_overlay_path = output_dir / "goal_bbox_overlay.png"
        goal_mask_overlay_path = output_dir / "goal_mask_overlay.png"
        goal_spec_path = output_dir / "goal_spec.json"
        Image.fromarray(self.goal_image_224).save(goal_image_path)
        Image.fromarray((self.goal_mask_224 > 0).astype(np.uint8) * 255, mode="L").save(goal_mask_path)
        bbox_overlay = self.goal_image_224.copy()
        bbox_obs = (self.current_goal_metadata or {}).get("bbox_obs")
        if isinstance(bbox_obs, (list, tuple)) and len(bbox_obs) == 4:
            x0, y0, x1, y1 = [int(v) for v in bbox_obs]
            cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 64, 64), thickness=2)
        Image.fromarray(bbox_overlay).save(goal_bbox_overlay_path)
        mask_overlay = self.goal_image_224.copy()
        mask_bool = (self.goal_mask_224 > 0)
        if np.any(mask_bool):
            mask_overlay = mask_overlay.astype(np.float32)
            mask_overlay[mask_bool] = 0.55 * mask_overlay[mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
            mask_overlay = np.clip(mask_overlay, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(mask_overlay).save(goal_mask_overlay_path)
        basis_debug_dir = output_dir / "goal_basis_debug"
        basis_debug_paths = {}
        metadata = self.current_goal_metadata or {}
        task_key = str(metadata.get("task_key", ""))
        probe_player_pose = metadata.get("probe_player_pose")
        target_world_center = metadata.get("target_world_center")
        voxel_key = metadata.get("voxel_key")
        occupied_voxel_keys = metadata.get("occupied_voxel_keys")
        mask_refine_applied = bool(metadata.get("mask_refine_applied", False))
        mask_refine_dx = int(metadata.get("mask_refine_dx", 0))
        mask_refine_dy = int(metadata.get("mask_refine_dy", 0))
        mask_refine_scale = float(metadata.get("mask_refine_scale", 1.0))
        if (
            task_key in AUTO_GOAL_BLOCK_TYPES
            and isinstance(probe_player_pose, dict)
            and isinstance(target_world_center, (list, tuple))
            and len(target_world_center) == 3
            and isinstance(voxel_key, (list, tuple))
            and len(voxel_key) == 3
        ):
            basis_debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                base_vx, base_vy, base_vz = [int(v) for v in voxel_key]
                occupied_voxels = set()
                if isinstance(occupied_voxel_keys, list):
                    for item in occupied_voxel_keys:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            occupied_voxels.add((int(item[0]), int(item[1]), int(item[2])))
                for basis_name, *_ in _camera_basis_variants(probe_player_pose):
                    raw_mask = _project_block_face_mask(
                        probe_player_pose,
                        (float(target_world_center[0]), float(target_world_center[1]), float(target_world_center[2])),
                        self.current_image.shape if hasattr(self, "current_image") and self.current_image is not None else (360, 640, 3),
                        basis_name=basis_name,
                        voxel_key=(base_vx, base_vy, base_vz),
                        occupied_voxels=occupied_voxels,
                    )
                    obs_mask = _resize_binary_mask(raw_mask, self.goal_image_224.shape)
                    if mask_refine_applied:
                        obs_mask = _transform_binary_mask(
                            obs_mask,
                            scale=mask_refine_scale,
                            dx=mask_refine_dx,
                            dy=mask_refine_dy,
                        )
                    overlay = self.goal_image_224.copy().astype(np.float32)
                    mask_bool_basis = obs_mask > 0
                    if np.any(mask_bool_basis):
                        overlay[mask_bool_basis] = 0.55 * overlay[mask_bool_basis] + 0.45 * np.array([0.0, 255.0, 0.0], dtype=np.float32)
                    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
                    basis_path = basis_debug_dir / f"{basis_name}.png"
                    Image.fromarray(overlay).save(basis_path)
                    basis_debug_paths[basis_name] = str(basis_path.resolve())
            except Exception:
                basis_debug_paths = {}
        goal_spec = {
            "goal_image_path": str(goal_image_path.resolve()),
            "goal_mask_path": str(goal_mask_path.resolve()),
            "goal_bbox_overlay_path": str(goal_bbox_overlay_path.resolve()),
            "goal_mask_overlay_path": str(goal_mask_overlay_path.resolve()),
            "segment_type": self.goal_segment_type,
            "cfg_coef": float(self.cfg_coef),
        }
        goal_spec.update(self.current_goal_metadata)
        if basis_debug_paths:
            goal_spec["goal_basis_debug_dir"] = str(basis_debug_dir.resolve())
            goal_spec["goal_basis_debug_paths"] = basis_debug_paths
        goal_spec_path.write_text(json.dumps(goal_spec, indent=2, ensure_ascii=False), encoding="utf-8")
        result = {
            "goal_image_path": str(goal_image_path.resolve()),
            "goal_mask_path": str(goal_mask_path.resolve()),
            "goal_bbox_overlay_path": str(goal_bbox_overlay_path.resolve()),
            "goal_mask_overlay_path": str(goal_mask_overlay_path.resolve()),
            "goal_spec_path": str(goal_spec_path.resolve()),
        }
        if basis_debug_paths:
            result["goal_basis_debug_dir"] = str(basis_debug_dir.resolve())
        return result

    def save_debug_assets(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for bucket_name in ("probe", "rollout"):
            bucket = self.debug_assets.get(bucket_name) or {}
            for key, value in bucket.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 2:
                        path = output_dir / f"{key}.png"
                        Image.fromarray(value.astype(np.uint8), mode="L").save(path)
                        saved[f"{key}_path"] = str(path.resolve())
                    elif value.ndim == 3:
                        path = output_dir / f"{key}.png"
                        Image.fromarray(value.astype(np.uint8)).save(path)
                        saved[f"{key}_path"] = str(path.resolve())
                else:
                    saved[key] = _sanitize_debug_value(value)
        metadata = dict(self.debug_assets.get("metadata") or {})
        if self.current_goal_metadata:
            metadata["current_goal_metadata"] = _sanitize_debug_value(self.current_goal_metadata)
        metadata_path = output_dir / "debug_metadata.json"
        metadata_path.write_text(json.dumps(_sanitize_debug_value(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
        saved["debug_metadata_path"] = str(metadata_path.resolve())
        selected_pose = dict(self.debug_assets.get("selected_pose") or {})
        if selected_pose:
            selected_dir = output_dir / "selected_pose"
            selected_dir.mkdir(parents=True, exist_ok=True)
            obs_image = selected_pose.get("candidate_obs")
            raw_pov = selected_pose.get("candidate_pov")
            goal_image = selected_pose.get("goal_image")
            goal_mask = selected_pose.get("goal_mask")
            raw_mask = selected_pose.get("raw_mask")
            goal_metadata = dict(selected_pose.get("goal_metadata") or {})
            goal_bbox = goal_metadata.get("bbox_obs")
            raw_bbox = goal_metadata.get("bbox_raw")
            if isinstance(obs_image, np.ndarray):
                Image.fromarray(obs_image.astype(np.uint8)).save(selected_dir / "candidate_obs.png")
            if isinstance(raw_pov, np.ndarray):
                Image.fromarray(raw_pov.astype(np.uint8)).save(selected_dir / "candidate_pov.png")
                resized_from_pov = cv2.resize(
                    raw_pov.astype(np.uint8),
                    dsize=(obs_image.shape[1], obs_image.shape[0]) if isinstance(obs_image, np.ndarray) else (224, 224),
                    interpolation=cv2.INTER_LINEAR,
                )
                Image.fromarray(resized_from_pov.astype(np.uint8)).save(selected_dir / "candidate_pov_resized.png")
            if isinstance(goal_image, np.ndarray):
                Image.fromarray(goal_image.astype(np.uint8)).save(selected_dir / "goal_image.png")
            if isinstance(goal_mask, np.ndarray):
                Image.fromarray((goal_mask > 0).astype(np.uint8) * 255, mode="L").save(selected_dir / "goal_mask.png")
                overlay = (goal_image if isinstance(goal_image, np.ndarray) else obs_image).copy().astype(np.float32)
                mask_bool = goal_mask > 0
                if np.any(mask_bool):
                    overlay[mask_bool] = 0.55 * overlay[mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
                overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
                Image.fromarray(overlay).save(selected_dir / "goal_mask_overlay.png")
                if isinstance(goal_bbox, (list, tuple)) and len(goal_bbox) == 4:
                    bbox_overlay = (goal_image if isinstance(goal_image, np.ndarray) else obs_image).copy()
                    x0, y0, x1, y1 = [int(v) for v in goal_bbox]
                    cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 64, 64), thickness=2)
                    Image.fromarray(bbox_overlay).save(selected_dir / "goal_bbox_overlay.png")
            if isinstance(raw_pov, np.ndarray) and isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                raw_bbox_overlay = raw_pov.copy()
                x0, y0, x1, y1 = [int(v) for v in raw_bbox]
                cv2.rectangle(raw_bbox_overlay, (x0, y0), (x1, y1), (255, 64, 64), thickness=2)
                Image.fromarray(raw_bbox_overlay).save(selected_dir / "candidate_pov_bbox_overlay.png")
            if isinstance(raw_pov, np.ndarray) and isinstance(raw_mask, np.ndarray):
                Image.fromarray((raw_mask > 0).astype(np.uint8) * 255, mode="L").save(selected_dir / "candidate_pov_raw_mask.png")
                raw_mask_overlay = raw_pov.copy().astype(np.float32)
                raw_mask_bool = raw_mask > 0
                if np.any(raw_mask_bool):
                    raw_mask_overlay[raw_mask_bool] = 0.55 * raw_mask_overlay[raw_mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
                raw_mask_overlay = np.clip(raw_mask_overlay, 0.0, 255.0).astype(np.uint8)
                Image.fromarray(raw_mask_overlay).save(selected_dir / "candidate_pov_raw_mask_overlay.png")
            selected_json = {k: _sanitize_debug_value(v) for k, v in selected_pose.items() if not isinstance(v, np.ndarray)}
            (selected_dir / "metadata.json").write_text(json.dumps(selected_json, indent=2, ensure_ascii=False), encoding="utf-8")
            if isinstance(goal_image, np.ndarray) and isinstance(goal_mask, np.ndarray):
                selected_goal_spec = {
                    "goal_image_path": str((selected_dir / "goal_image.png").resolve()),
                    "goal_mask_path": str((selected_dir / "goal_mask.png").resolve()),
                    "goal_mask_overlay_path": str((selected_dir / "goal_mask_overlay.png").resolve()),
                    "segment_type": self.goal_segment_type,
                    "cfg_coef": float(self.cfg_coef),
                    "obs_size": [int(goal_image.shape[1]), int(goal_image.shape[0])],
                    "model_uri": self.model_path,
                }
                if isinstance(goal_bbox, (list, tuple)) and len(goal_bbox) == 4:
                    selected_goal_spec["goal_bbox_overlay_path"] = str((selected_dir / "goal_bbox_overlay.png").resolve())
                for key, value in goal_metadata.items():
                    if str(key) == "occupied_voxel_keys":
                        continue
                    selected_goal_spec[str(key)] = _sanitize_debug_value(value)
                (selected_dir / "goal_spec.json").write_text(
                    json.dumps(selected_goal_spec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            saved["selected_pose_dir"] = str(selected_dir.resolve())
        candidate_debug = list(self.debug_assets.get("candidate_debug") or [])
        if candidate_debug:
            candidates_dir = output_dir / "candidate_debug"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            for entry_index, entry in enumerate(candidate_debug, start=1):
                decision = str(entry.get("decision") or "candidate")
                rejection_reason = str(entry.get("rejection_reason") or "").strip()
                suffix = f"_{rejection_reason}" if rejection_reason else ""
                entry_dir = candidates_dir / f"{entry_index:03d}_{decision}{suffix}"
                entry_dir.mkdir(parents=True, exist_ok=True)
                obs_image = entry.get("candidate_obs")
                raw_pov = entry.get("candidate_pov")
                goal_image = entry.get("goal_image")
                goal_mask = entry.get("goal_mask")
                raw_mask = entry.get("raw_mask")
                goal_metadata = dict(entry.get("goal_metadata") or {})
                goal_bbox = goal_metadata.get("bbox_obs")
                raw_bbox = goal_metadata.get("bbox_raw")
                if isinstance(obs_image, np.ndarray):
                    Image.fromarray(obs_image.astype(np.uint8)).save(entry_dir / "candidate_obs.png")
                if isinstance(raw_pov, np.ndarray):
                    Image.fromarray(raw_pov.astype(np.uint8)).save(entry_dir / "candidate_pov.png")
                if isinstance(goal_image, np.ndarray):
                    Image.fromarray(goal_image.astype(np.uint8)).save(entry_dir / "goal_image.png")
                if isinstance(goal_mask, np.ndarray):
                    goal_mask_bool = goal_mask > 0
                    Image.fromarray(goal_mask_bool.astype(np.uint8) * 255, mode="L").save(entry_dir / "goal_mask.png")
                    overlay_source = goal_image if isinstance(goal_image, np.ndarray) else obs_image
                    if isinstance(overlay_source, np.ndarray):
                        overlay = overlay_source.copy().astype(np.float32)
                        if np.any(goal_mask_bool):
                            overlay[goal_mask_bool] = 0.55 * overlay[goal_mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
                        Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8)).save(entry_dir / "goal_mask_overlay.png")
                    if isinstance(goal_image, np.ndarray) and isinstance(goal_bbox, (list, tuple)) and len(goal_bbox) == 4:
                        bbox_overlay = goal_image.copy()
                        x0, y0, x1, y1 = [int(v) for v in goal_bbox]
                        cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 64, 64), thickness=2)
                        Image.fromarray(bbox_overlay).save(entry_dir / "goal_bbox_overlay.png")
                if isinstance(raw_pov, np.ndarray) and isinstance(raw_mask, np.ndarray):
                    raw_mask_bool = raw_mask > 0
                    Image.fromarray(raw_mask_bool.astype(np.uint8) * 255, mode="L").save(entry_dir / "candidate_pov_raw_mask.png")
                    raw_overlay = raw_pov.copy().astype(np.float32)
                    if np.any(raw_mask_bool):
                        raw_overlay[raw_mask_bool] = 0.55 * raw_overlay[raw_mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
                    Image.fromarray(np.clip(raw_overlay, 0.0, 255.0).astype(np.uint8)).save(entry_dir / "candidate_pov_raw_mask_overlay.png")
                    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                        raw_bbox_overlay = raw_pov.copy()
                        x0, y0, x1, y1 = [int(v) for v in raw_bbox]
                        cv2.rectangle(raw_bbox_overlay, (x0, y0), (x1, y1), (255, 64, 64), thickness=2)
                        Image.fromarray(raw_bbox_overlay).save(entry_dir / "candidate_pov_bbox_overlay.png")
                    if bool(AUTO_GOAL_ENABLE_PROJECTION_CONSISTENCY) and "projection_consistency_best_dy" in goal_metadata:
                        projection_best_dy = int(goal_metadata.get("projection_consistency_best_dy", 0) or 0)
                        shifted_raw_mask = _shift_binary_mask_y(raw_mask_bool.astype(np.uint8), projection_best_dy)
                        Image.fromarray((shifted_raw_mask > 0).astype(np.uint8) * 255, mode="L").save(
                            entry_dir / "candidate_pov_raw_mask_best_shift.png"
                        )
                        shifted_overlay = raw_pov.copy().astype(np.float32)
                        if np.any(shifted_raw_mask > 0):
                            shifted_overlay[shifted_raw_mask > 0] = (
                                0.55 * shifted_overlay[shifted_raw_mask > 0]
                                + 0.45 * np.array([0.0, 255.0, 255.0], dtype=np.float32)
                            )
                        Image.fromarray(np.clip(shifted_overlay, 0.0, 255.0).astype(np.uint8)).save(
                            entry_dir / "candidate_pov_raw_mask_best_shift_overlay.png"
                        )
                metadata = {k: _sanitize_debug_value(v) for k, v in entry.items() if not isinstance(v, np.ndarray)}
                (entry_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            saved["candidate_debug_dir"] = str(candidates_dir.resolve())
        return saved

    def step(self, input_action=None):
        if input_action is not None:
            action = input_action
            self.last_policy_action = None
            self.last_policy_logprob = None
            self.last_policy_value = None
            self.last_policy_value_raw = None
            self.last_memory_in = None
            self.last_model_input = None
        else:
            obj_id = torch.tensor(SEGMENT_MAPPING[self.goal_segment_type], dtype=torch.long)
            obj_mask = torch.from_numpy(self.goal_mask_224.copy()).to(dtype=torch.uint8)
            cross_view_image = self.goal_image_224.copy()
            obs = {
                "image": self.obs["image"],
                "env_prev_action": self.obs["env_prev_action"],
                "cross_view": {
                    "cross_view_image": cross_view_image,
                    "cross_view_obj_id": obj_id,
                    "cross_view_obj_mask": obj_mask,
                },
            }
            self.last_model_input = {
                "image": np.array(self.obs["image"], copy=True),
                "obj_mask": obj_mask.detach().cpu().clone(),
                "obj_id": int(obj_id.item()),
                "cross_view_image": torch.from_numpy(cross_view_image.copy()).to(dtype=torch.uint8),
                "cross_view_obj_mask": obj_mask.detach().cpu().clone(),
                "cross_view_obj_id": int(obj_id.item()),
                "env_prev_action": self._detach_action_for_logging(self.obs["env_prev_action"]),
            }
            self.last_memory_in = self._detach_state_for_logging(self.state)
            action, self.state = self.agent.get_action(obs, self.state, input_shape="*")
            self.last_policy_action = self._detach_action_for_logging(action)
            try:
                self.last_policy_logprob = float(self.agent.model.pi_head.logprob(action, self.agent.cache_latents["pi_logits"]).item())
            except Exception:
                self.last_policy_logprob = None
            try:
                cached_vpred = self.agent.cache_latents["vpred"]
                value_head_owner = getattr(self.agent, "model", self.agent)
                value_head = getattr(value_head_owner, "value_head", None)
                self.last_policy_value = float(cached_vpred.reshape(-1)[0].item())
                if value_head is not None:
                    with torch.no_grad():
                        raw_vpred = value_head.denormalize(cached_vpred)
                    self.last_policy_value_raw = float(raw_vpred.reshape(-1)[0].detach().cpu().item())
                else:
                    self.last_policy_value_raw = None
            except Exception:
                self.last_policy_value = None
                self.last_policy_value_raw = None

        self.last_action_summary = self.summarize_agent_action(action)
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.last_reward = float(self.reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.current_image = self.info["pov"]
        self.last_segment_area = int(self.goal_mask_224.sum())
        self.num_steps += 1
        self.image_history.append(self.current_image.copy())
        return self.current_image

    def estimate_current_policy_value(self) -> Optional[Dict[str, float]]:
        if not hasattr(self, "agent") or not hasattr(self, "obs"):
            return None
        try:
            obj_id = torch.tensor(SEGMENT_MAPPING[self.goal_segment_type], dtype=torch.long)
            obj_mask = torch.from_numpy(self.goal_mask_224.copy()).to(dtype=torch.uint8)
            cross_view_image = self.goal_image_224.copy()
            obs = {
                "image": self.obs["image"],
                "env_prev_action": self.obs["env_prev_action"],
                "cross_view": {
                    "cross_view_image": cross_view_image,
                    "cross_view_obj_id": obj_id,
                    "cross_view_obj_mask": obj_mask,
                },
            }
            cached_latents = getattr(self.agent, "cache_latents", None)
            cached_vpred = getattr(self.agent, "vpred", None)
            try:
                self.agent.get_action(obs, self.state, deterministic=True, input_shape="*")
                current_vpred = self.agent.cache_latents["vpred"]
                value_head_owner = getattr(self.agent, "model", self.agent)
                value_head = getattr(value_head_owner, "value_head", None)
                value_norm = float(current_vpred.reshape(-1)[0].item())
                if value_head is None:
                    return {
                        "value_norm": value_norm,
                        "value_raw": value_norm,
                    }
                with torch.no_grad():
                    value_raw = value_head.denormalize(current_vpred)
                return {
                    "value_norm": value_norm,
                    "value_raw": float(value_raw.reshape(-1)[0].detach().cpu().item()),
                }
            finally:
                if cached_latents is not None:
                    self.agent.cache_latents = cached_latents
                if cached_vpred is not None and hasattr(self.agent, "vpred"):
                    self.agent.vpred = cached_vpred
        except Exception:
            return None

    def close(self):
        if hasattr(self, "env"):
            self.env.close()
        if hasattr(self, "agent"):
            del self.agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
