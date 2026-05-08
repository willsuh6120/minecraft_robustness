import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml
from PIL import Image

from minestudio.benchmark import prepare_task_configs
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import VoxelsCallback, load_callbacks_from_config
from minestudio.tutorials.inference.evaluate_rocket.crossview_utils import (
    AUTO_GOAL_BLOCK_TYPES,
    AUTO_GOAL_DISTANCE_SAMPLES,
    AUTO_GOAL_HEIGHT_SAMPLES,
    AUTO_GOAL_MAX_POSE_CANDIDATES,
    AUTO_GOAL_MIN_IMAGE_MEAN,
    AUTO_GOAL_MOB_TYPES,
    AUTO_GOAL_MOBS_BOUNDS,
    AUTO_GOAL_PITCH_OFFSET_SAMPLES,
    AUTO_GOAL_POST_TP_SETTLE_STEPS,
    AUTO_GOAL_SUPPORTED_TASKS,
    AUTO_GOAL_VOXEL_BOUNDS,
    DEFAULT_EYE_HEIGHT,
    NearbyMobsCallback,
    _block_corners,
    _block_face_definitions,
    _camera_basis_variants,
    _crossview_difference_metrics,
    _distance3,
    _get_basis_by_name,
    _look_angles_for_target,
    _mask_bbox,
    _mask_center,
    _normalize_name,
    _player_pose,
    _project_block_face_mask,
    _project_bbox,
    _project_world_point_with_basis,
    _resize_binary_mask,
    _sanitize_debug_value,
    _voxel_world_center_from_player_pose,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_protocol import (
    apply_protocol_to_task_specs,
    resolve_interaction_protocol,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    resolve_interaction_task_specs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump cross-view goal image candidates for manual selection."
    )
    parser.add_argument("--env-source", type=str, default="rocket2_official")
    parser.add_argument("--task-group", type=str, default="rocket2_official")
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--task-group-path", type=str, default="/home/gyulab/envgen2/ROCKET-2/env_conf")
    parser.add_argument("--task", type=str, required=True, help="Task key, e.g. mine_coal.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--warmup-noop-steps", type=int, default=None)
    parser.add_argument("--max-pose-candidates", type=int, default=AUTO_GOAL_MAX_POSE_CANDIDATES)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=6,
        help="Number of initially visible target anchors to use for camera pose sampling.",
    )
    parser.add_argument(
        "--max-anchor-distance",
        type=float,
        default=12.0,
        help="Discard block anchors farther than this many blocks from the reset player pose. Use <=0 to disable.",
    )
    parser.add_argument(
        "--candidate-distances",
        type=str,
        default="4.0,5.5,7.0",
        help="Comma-separated camera distances from the target for candidate dumping.",
    )
    parser.add_argument(
        "--candidate-height-offsets",
        type=str,
        default="0.0,1.5,3.0",
        help="Comma-separated camera feet-y offsets from reset player y for candidate dumping.",
    )
    parser.add_argument(
        "--candidate-pitch-offsets",
        type=str,
        default="0.0,-8.0,8.0",
        help="Comma-separated pitch offsets for candidate dumping.",
    )
    parser.add_argument(
        "--min-target-bbox-frac",
        type=float,
        default=0.002,
        help="Reject candidates whose projected target bbox covers less than this image fraction.",
    )
    parser.add_argument(
        "--max-target-bbox-frac",
        type=float,
        default=0.22,
        help="Reject candidates whose projected target bbox covers more than this image fraction.",
    )
    parser.add_argument(
        "--min-target-bbox-height-px",
        type=int,
        default=18,
        help="Reject candidates whose projected target bbox height in raw pixels is below this threshold.",
    )
    parser.add_argument(
        "--min-target-bbox-width-px",
        type=int,
        default=0,
        help="Reject candidates whose projected target bbox width in raw pixels is below this threshold. Set 0 to disable.",
    )
    parser.add_argument(
        "--reject-edge-clipped-bbox",
        action="store_true",
        help="Reject candidates whose target bbox touches the raw POV image border.",
    )
    parser.add_argument(
        "--allow-edge-clipped-anchors",
        action="store_true",
        help="Allow target anchors whose projected bbox touches the raw POV border.",
    )
    parser.add_argument(
        "--max-anchor-center-y-frac",
        type=float,
        default=0.86,
        help="Discard target anchors whose projected center is below this raw-image y fraction.",
    )
    parser.add_argument(
        "--pose-selection",
        type=str,
        default="diverse",
        choices=["diverse", "first"],
        help="How to select max-pose-candidates from the full pose set.",
    )
    parser.add_argument(
        "--candidate-source",
        type=str,
        default="reference_sweep",
        choices=["reference_sweep", "target_teleport"],
        help="Use safe camera sweeps from reset pose, or teleport around target.",
    )
    parser.add_argument(
        "--projection-basis",
        type=str,
        default="minecraft",
        help=(
            "Camera projection basis to use for target points/bboxes. "
            "Use 'minecraft' for the canonical Minecraft camera model; "
            "use 'auto' only for debugging because it can produce plausible but wrong points."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Print per-pose timing breakdown for tp/settle/projection/save.",
    )
    parser.add_argument(
        "--reset-retries",
        type=int,
        default=3,
        help="Retry Minecraft/Malmo env reset this many times before failing.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/gyulab/envgen2/Minestudio/outputs/evaluate_rocket/crossview_goal_candidates",
    )
    return parser.parse_args()


def build_callbacks(task_key: str, task_config: Dict[str, Any]):
    callbacks = list(load_callbacks_from_config(task_config))
    if task_key in AUTO_GOAL_BLOCK_TYPES and not any(isinstance(cb, VoxelsCallback) for cb in callbacks):
        callbacks.append(VoxelsCallback(AUTO_GOAL_VOXEL_BOUNDS))
    if task_key in AUTO_GOAL_MOB_TYPES and not any(isinstance(cb, NearbyMobsCallback) for cb in callbacks):
        callbacks.append(NearbyMobsCallback(AUTO_GOAL_MOBS_BOUNDS))
    return callbacks


def image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    return {
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "non_black_frac": float(np.count_nonzero(np.any(image > 8, axis=-1)) / max(1, image.shape[0] * image.shape[1])),
    }


def parse_float_csv(value: str, *, name: str) -> tuple[float, ...]:
    items = []
    for piece in str(value or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        items.append(float(piece))
    if not items:
        raise ValueError(f"{name} must contain at least one float")
    return tuple(items)


def _clip_bbox_to_image(image_shape, bbox):
    if not bbox or len(bbox) != 4:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def target_patch_metrics(task_key: str, image: np.ndarray, bbox) -> Dict[str, float]:
    clipped = _clip_bbox_to_image(image.shape, bbox)
    if clipped is None:
        return {
            "target_patch_area": 0.0,
            "target_patch_mean": 0.0,
            "target_signal_score": float("-inf"),
            "target_dark_frac": 0.0,
            "target_green_excess": 0.0,
        }
    x0, y0, x1, y1 = clipped
    patch = image[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    area = float(patch.shape[0] * patch.shape[1])
    luma = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
    patch_mean = float(luma.mean())
    dark_frac = float(np.count_nonzero(luma < 85.0) / max(1.0, area))
    very_dark_frac = float(np.count_nonzero(luma < 55.0) / max(1.0, area))
    green_excess = np.maximum(0.0, patch[..., 1] - np.maximum(patch[..., 0], patch[..., 2]))
    green_score = float(green_excess.mean())

    pad = 8
    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(int(image.shape[1]) - 1, x1 + pad)
    ry1 = min(int(image.shape[0]) - 1, y1 + pad)
    ring = image[ry0 : ry1 + 1, rx0 : rx1 + 1].astype(np.float32)
    ring_mask = np.ones((ring.shape[0], ring.shape[1]), dtype=bool)
    ring_mask[(y0 - ry0) : (y1 - ry0 + 1), (x0 - rx0) : (x1 - rx0 + 1)] = False
    ring_pixels = ring[ring_mask]
    if len(ring_pixels) > 0:
        ring_luma = 0.299 * ring_pixels[:, 0] + 0.587 * ring_pixels[:, 1] + 0.114 * ring_pixels[:, 2]
        ring_mean = float(ring_luma.mean())
        ring_dark_frac = float(np.count_nonzero(ring_luma < 85.0) / max(1, len(ring_luma)))
    else:
        ring_mean = patch_mean
        ring_dark_frac = dark_frac

    if task_key == "mine_emerald":
        signal_score = 50.0 * green_score
    elif task_key == "mine_coal":
        # Coal ore is mostly stone with black flecks; prefer locally darker patches.
        signal_score = (
            650.0 * max(0.0, dark_frac - ring_dark_frac)
            + 350.0 * very_dark_frac
            + 1.5 * max(0.0, ring_mean - patch_mean)
        )
    else:
        signal_score = 0.0
    return {
        "target_patch_area": area,
        "target_patch_mean": patch_mean,
        "target_ring_mean": ring_mean,
        "target_dark_frac": dark_frac,
        "target_ring_dark_frac": ring_dark_frac,
        "target_very_dark_frac": very_dark_frac,
        "target_green_excess": green_score,
        "target_signal_score": float(signal_score),
    }


def target_projection_quality(raw_shape, bbox_raw) -> Dict[str, Any]:
    clipped = _clip_bbox_to_image(raw_shape, bbox_raw)
    if clipped is None:
        return {
            "target_bbox_valid": False,
            "target_bbox_area_frac": 0.0,
            "target_bbox_edge_clipped": True,
            "target_bbox_width_px": 0,
            "target_bbox_height_px": 0,
        }
    x0, y0, x1, y1 = clipped
    h, w = int(raw_shape[0]), int(raw_shape[1])
    width_px = int(x1 - x0 + 1)
    height_px = int(y1 - y0 + 1)
    area = float(width_px * height_px)
    area_frac = area / max(1.0, float(h * w))
    edge_clipped = bool(x0 <= 0 or y0 <= 0 or x1 >= w - 1 or y1 >= h - 1)
    return {
        "target_bbox_valid": True,
        "target_bbox_area_frac": float(area_frac),
        "target_bbox_edge_clipped": edge_clipped,
        "target_bbox_width_px": width_px,
        "target_bbox_height_px": height_px,
    }


def is_valid_image_candidate(metrics: Dict[str, float]) -> bool:
    if float(metrics["image_mean"]) < float(AUTO_GOAL_MIN_IMAGE_MEAN):
        return False
    if float(metrics["non_black_frac"]) < 0.2:
        return False
    return True


def image_candidate_score(
    *,
    image_metrics: Dict[str, float],
    crossview_metrics: Dict[str, float],
    target_center_raw: Dict[str, Any] | None = None,
) -> float:
    center = target_center_raw or {}
    cx = float(center.get("x", 320.0))
    cy = float(center.get("y", 180.0))
    center_penalty = abs(cx - 320.0) + abs(cy - 180.0)
    return (
        50.0 * float(crossview_metrics["camera_displacement"])
        + 4.0 * float(crossview_metrics["yaw_delta_deg"])
        + 6.0 * float(crossview_metrics["pitch_delta_deg"])
        + 2000.0 * float(crossview_metrics["changed_pixel_frac"])
        + 8.0 * float(image_metrics["image_mean"])
        + 2.0 * float(image_metrics["image_std"])
        + 500.0 * float(image_metrics["non_black_frac"])
        - 0.35 * float(center_penalty)
    )


def is_valid_dump_crossview(metrics: Dict[str, float], *, require_camera_displacement: bool) -> bool:
    if require_camera_displacement and float(metrics["camera_displacement"]) < 2.5:
        return False
    return bool(
        float(metrics["yaw_delta_deg"]) >= 10.0
        or float(metrics["pitch_delta_deg"]) >= 8.0
        or float(metrics["mean_abs_diff"]) >= 6.0
        or float(metrics["changed_pixel_frac"]) >= 0.01
    )


def _angular_error_deg(current: float, target: float) -> float:
    delta = (float(current) - float(target) + 180.0) % 360.0 - 180.0
    return abs(float(delta))


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
    x = int(support_block["x"])
    y = int(support_block["y"])
    z = int(support_block["z"])
    block_type = str(support_block.get("block_type", "minecraft:barrier"))
    env.env.execute_cmd(f"/setblock {x} {y} {z} {block_type} keep")


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
        "frames_stepped": 0,
        "stabilized": False,
        "buffer_frames_applied": 0,
    }

    for step_idx in range(max(1, int(max_wait_steps))):
        obs, _, terminated, truncated, info = env.step(env.noop_action())
        diagnostics["frames_stepped"] = int(step_idx + 1)
        if info is not None:
            pose = _player_pose(info)
            pose_error = _pose_error(pose, expected_pose)
            diagnostics["last_pose"] = _sanitize_debug_value(pose)
            diagnostics["last_pose_error"] = _sanitize_debug_value(pose_error)
            if _pose_matches_expected(pose, expected_pose):
                streak += 1
            else:
                streak = 0
            diagnostics["match_streak"] = int(streak)
        if terminated or truncated:
            break
        if int(step_idx + 1) >= int(min_wait_steps) and int(streak) >= int(match_streak_required):
            diagnostics["stabilized"] = True
            for _ in range(max(0, int(render_buffer_steps))):
                obs, _, terminated, truncated, info = env.step(env.noop_action())
                diagnostics["buffer_frames_applied"] = int(diagnostics["buffer_frames_applied"]) + 1
                diagnostics["frames_stepped"] = int(diagnostics["frames_stepped"]) + 1
                if info is not None:
                    pose = _player_pose(info)
                    diagnostics["last_pose"] = _sanitize_debug_value(pose)
                    diagnostics["last_pose_error"] = _sanitize_debug_value(_pose_error(pose, expected_pose))
                if terminated or truncated:
                    break
            break

    return obs, info, terminated, truncated, diagnostics


def _project_bbox_fixed_basis(
    player_pose: Dict[str, float],
    corners: list[tuple[float, float, float]],
    image_shape,
    *,
    basis_name: str | None,
):
    if not basis_name or str(basis_name).lower() == "auto":
        return _project_bbox(player_pose, corners, image_shape)

    basis = _get_basis_by_name(player_pose, str(basis_name))
    if basis is None:
        return None
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
    projections = []
    for corner in corners:
        projection = _project_world_point_with_basis(
            origin,
            np.asarray(corner, dtype=np.float32),
            image_shape,
            forward=forward,
            right=right,
            up=up,
        )
        if projection is not None:
            projections.append(projection)
    if not projections:
        return None
    xs = [float(item["x"]) for item in projections]
    ys = [float(item["y"]) for item in projections]
    x0 = max(0, min(width - 1, int(np.floor(min(xs)))))
    y0 = max(0, min(height - 1, int(np.floor(min(ys)))))
    x1 = max(0, min(width - 1, int(np.ceil(max(xs)))))
    y1 = max(0, min(height - 1, int(np.ceil(max(ys)))))
    if x1 <= x0 or y1 <= y0:
        return None
    center = {
        "x": float(sum(xs) / len(xs)),
        "y": float(sum(ys) / len(ys)),
        "depth": float(sum(float(item["depth"]) for item in projections) / len(projections)),
        "basis_name": str(basis_name),
    }
    return (x0, y0, x1, y1), center


def _occupied_voxels_from_info(info: Dict[str, Any] | None) -> set[tuple[int, int, int]]:
    occupied_voxels: set[tuple[int, int, int]] = set()
    if not isinstance(info, dict):
        return occupied_voxels
    voxels = info.get("voxels") or []
    if not isinstance(voxels, list):
        return occupied_voxels
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if not voxel_type or voxel_type in {"air", "cave_air", "void_air"}:
            continue
        occupied_voxels.add(
            (
                int(round(float(voxel.get("x", 0.0)))),
                int(round(float(voxel.get("y", 0.0)))),
                int(round(float(voxel.get("z", 0.0)))),
            )
        )
    return occupied_voxels


def _voxel_key_from_world_center(
    player_pose: Dict[str, float],
    world_center: tuple[float, float, float],
) -> tuple[int, int, int]:
    return (
        int(round(float(world_center[0]) - math.floor(float(player_pose["x"])) - 0.5)),
        int(round(float(world_center[1]) - math.floor(float(player_pose["y"])) - 0.5)),
        int(round(float(world_center[2]) - math.floor(float(player_pose["z"])) - 0.5)),
    )


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


def _project_visible_target_with_basis(
    *,
    player_pose: Dict[str, float],
    info: Dict[str, Any] | None,
    target_center: tuple[float, float, float],
    image_shape,
    basis_name: str | None,
    voxel_key: tuple[int, int, int] | None = None,
):
    if not basis_name or str(basis_name).lower() == "auto":
        projected = _project_bbox_fixed_basis(
            player_pose,
            _block_corners(target_center),
            image_shape,
            basis_name=basis_name,
        )
        if projected is None:
            return None
        bbox_raw, center_raw = projected
        return {
            "bbox_raw": bbox_raw,
            "target_center_raw": center_raw,
            "raw_mask": None,
            "voxel_key": None,
            "mask_source": "cube_bbox_fallback",
            "mask_area_raw": 0,
        }

    basis = _get_basis_by_name(player_pose, str(basis_name))
    if basis is None:
        return None
    if voxel_key is None:
        voxel_key = _voxel_key_from_world_center(player_pose, target_center)
    occupied_voxels = _occupied_voxels_from_info(info)
    visible_faces = _visible_face_projection_entries(
        player_pose=player_pose,
        target_center=target_center,
        image_shape=image_shape,
        basis_name=str(basis_name),
        voxel_key=voxel_key,
        occupied_voxels=occupied_voxels,
    )
    raw_mask = _project_block_face_mask(
        player_pose,
        target_center,
        image_shape,
        basis_name=str(basis_name),
        voxel_key=voxel_key,
        occupied_voxels=occupied_voxels,
    )
    if int(raw_mask.sum()) <= 0:
        return None
    bbox_raw = _mask_bbox(raw_mask)
    if bbox_raw[2] <= bbox_raw[0] or bbox_raw[3] <= bbox_raw[1]:
        return None

    if len(visible_faces) == 1:
        face_entry = visible_faces[0]
        polygon_center = face_entry.get("polygon_center")
        face_center_projection = face_entry.get("center_projection")
        if isinstance(polygon_center, dict):
            center_projection = {
                "x": float(polygon_center["x"]),
                "y": float(polygon_center["y"]),
                "depth": float(face_center_projection["depth"]) if isinstance(face_center_projection, dict) else 0.0,
                "basis_name": str(basis_name),
            }
        elif isinstance(face_center_projection, dict):
            center_projection = dict(face_center_projection)
        else:
            center_projection = {
                "x": float((bbox_raw[0] + bbox_raw[2]) / 2.0),
                "y": float((bbox_raw[1] + bbox_raw[3]) / 2.0),
                "depth": 0.0,
                "basis_name": str(basis_name),
            }
    else:
        point_x, point_y = _mask_prompt_point(raw_mask)
        forward, right, up = basis
        origin = np.array(
            [
                float(player_pose["x"]),
                float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
                float(player_pose["z"]),
            ],
            dtype=np.float32,
        )
        world_center_projection = _project_world_point_with_basis(
            origin,
            np.asarray(target_center, dtype=np.float32),
            image_shape,
            forward=forward,
            right=right,
            up=up,
        )
        depth = float(world_center_projection["depth"]) if world_center_projection is not None else 0.0
        center_projection = {
            "x": float(point_x),
            "y": float(point_y),
            "depth": float(depth),
            "basis_name": str(basis_name),
        }
    return {
        "bbox_raw": bbox_raw,
        "target_center_raw": center_projection,
        "raw_mask": raw_mask,
        "voxel_key": [int(voxel_key[0]), int(voxel_key[1]), int(voxel_key[2])],
        "mask_source": "projected_visible_faces",
        "mask_area_raw": int(raw_mask.sum()),
        "num_visible_faces": int(len(visible_faces)),
        "visible_faces": _visible_face_metadata(visible_faces),
    }


def exact_target_projection(
    pose_candidate: Dict[str, Any],
    candidate_pose: Dict[str, Any],
    candidate_info: Dict[str, Any] | None,
    raw_image: np.ndarray,
    obs_image: np.ndarray,
):
    target_center = pose_candidate.get("target_world_center")
    if not target_center or len(target_center) != 3:
        return None
    projected = _project_visible_target_with_basis(
        player_pose=candidate_pose,
        info=candidate_info,
        target_center=tuple(float(v) for v in target_center),
        image_shape=raw_image.shape,
        basis_name=pose_candidate.get("projection_basis"),
    )
    if projected is None:
        return None
    bbox_raw = projected["bbox_raw"]
    center_raw = projected["target_center_raw"]
    raw_mask = projected.get("raw_mask")
    if isinstance(raw_mask, np.ndarray) and int(raw_mask.sum()) > 0:
        obs_mask = _resize_binary_mask(raw_mask, obs_image.shape)
        bbox_obs = list(_mask_bbox(obs_mask))
        obs_point_x, obs_point_y = _mask_prompt_point(obs_mask)
        center_obs = {"x": float(obs_point_x), "y": float(obs_point_y)}
    else:
        bbox_obs = _scale_bbox(bbox_raw, raw_image.shape, obs_image.shape)
        center_obs = _scale_point(center_raw, raw_image.shape, obs_image.shape)
    if bbox_obs is None or center_obs is None:
        return None
    return {
        "bbox_raw": bbox_raw,
        "target_center_raw": center_raw,
        "bbox_obs": bbox_obs,
        "target_center_obs": center_obs,
        "raw_mask": raw_mask,
        "mask_source": projected.get("mask_source"),
        "mask_area_raw": int(projected.get("mask_area_raw", 0)),
        "voxel_key": projected.get("voxel_key"),
        "num_visible_faces": int(projected.get("num_visible_faces", 0)),
        "visible_faces": _sanitize_debug_value(projected.get("visible_faces", [])),
    }


def _configured_target_entries(task_key: str, task_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_types = AUTO_GOAL_BLOCK_TYPES.get(task_key, ())
    default_type = f"{target_types[0]}_ore" if target_types else "target"
    entries: List[Dict[str, Any]] = []

    for key in ("target_world_center", "target_center_world", "target_center"):
        value = task_config.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            entries.append({"target_type": default_type, "world_center": tuple(float(item) for item in value)})

    for key in ("target_blocks", "targets"):
        raw_entries = task_config.get(key) or []
        if isinstance(raw_entries, dict):
            raw_entries = [raw_entries]
        if not isinstance(raw_entries, list):
            continue
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            center = item.get("world_center") or item.get("target_world_center") or item.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) == 3):
                continue
            target_type = _normalize_name(item.get("type") or item.get("target_type") or default_type)
            if target_types and not any(target_type_key in target_type for target_type_key in target_types):
                continue
            entries.append({"target_type": target_type, "world_center": tuple(float(value) for value in center)})

    unique_entries: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        key = (entry["target_type"], tuple(round(float(value), 4) for value in entry["world_center"]))
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
    return unique_entries


def configured_target_anchors(
    task_key: str,
    task_config: Dict[str, Any],
    info: Dict[str, Any],
    raw_image: np.ndarray,
    *,
    projection_basis: str | None,
    allow_edge_clipped: bool,
    max_center_y_frac: float,
) -> List[Dict[str, Any]]:
    player_pose = _player_pose(info)
    player_center = (float(player_pose["x"]), float(player_pose["y"]), float(player_pose["z"]))
    anchors: List[Dict[str, Any]] = []
    for entry in _configured_target_entries(task_key, task_config):
        world_center = tuple(float(value) for value in entry["world_center"])
        projected = _project_visible_target_with_basis(
            player_pose=player_pose,
            info=info,
            target_center=world_center,
            image_shape=raw_image.shape,
            basis_name=projection_basis,
        )
        if projected is None:
            continue
        bbox_raw = projected["bbox_raw"]
        center_raw = projected["target_center_raw"]
        projection_quality = target_projection_quality(raw_image.shape, bbox_raw)
        if not bool(allow_edge_clipped) and bool(projection_quality["target_bbox_edge_clipped"]):
            continue
        if float(max_center_y_frac) > 0.0:
            center_y_frac = float(center_raw.get("y", 0.0)) / max(1.0, float(raw_image.shape[0]))
            if center_y_frac > float(max_center_y_frac):
                continue
        metrics = target_patch_metrics(task_key, raw_image, bbox_raw)
        area = float(metrics["target_patch_area"])
        if area <= 4:
            continue
        center_penalty = abs(float(center_raw.get("x", 320.0)) - 320.0) + abs(float(center_raw.get("y", 180.0)) - 180.0)
        distance = _distance3(world_center, player_center)
        score = (
            1.0 * float(metrics["target_signal_score"])
            + 0.10 * area
            - 0.40 * center_penalty
            - 80.0 * distance
        )
        anchors.append(
            {
                "task_key": task_key,
                "goal_source": "configured_target_world_center",
                "target_type": entry["target_type"],
                "world_center": world_center,
                "projection_basis": str(center_raw.get("basis_name", projection_basis or "auto")),
                "voxel_offset_raw": None,
                "voxel_key": projected.get("voxel_key"),
                "bbox_raw": bbox_raw,
                "target_center_raw": center_raw,
                "mask_source": projected.get("mask_source"),
                "mask_area_raw": int(projected.get("mask_area_raw", 0)),
                "target_patch_metrics": metrics,
                "target_projection_quality": projection_quality,
                "sort_distance": distance,
                "anchor_score": float(score),
            }
        )
    anchors.sort(key=lambda item: float(item["anchor_score"]), reverse=True)
    return anchors


def extract_visible_target_anchors(
    task_key: str,
    info: Dict[str, Any],
    raw_image: np.ndarray,
    limit: int,
    *,
    projection_basis: str | None,
    allow_edge_clipped: bool,
    max_center_y_frac: float,
) -> List[Dict[str, Any]]:
    player_pose = _player_pose(info)
    player_center = (float(player_pose["x"]), float(player_pose["y"]), float(player_pose["z"]))
    target_types = AUTO_GOAL_BLOCK_TYPES.get(task_key, ())
    voxels = info.get("voxels") or []
    anchors: List[Dict[str, Any]] = []
    if not isinstance(voxels, list):
        return anchors
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
        projected = _project_visible_target_with_basis(
            player_pose=player_pose,
            info={"voxels": voxels},
            target_center=world_center,
            image_shape=raw_image.shape,
            basis_name=projection_basis,
            voxel_key=(
                int(round(raw_offset[0])),
                int(round(raw_offset[1])),
                int(round(raw_offset[2])),
            ),
        )
        if projected is None:
            continue
        bbox_raw = projected["bbox_raw"]
        center_raw = projected["target_center_raw"]
        projection_quality = target_projection_quality(raw_image.shape, bbox_raw)
        if not bool(allow_edge_clipped) and bool(projection_quality["target_bbox_edge_clipped"]):
            continue
        if float(max_center_y_frac) > 0.0:
            center_y_frac = float(center_raw.get("y", 0.0)) / max(1.0, float(raw_image.shape[0]))
            if center_y_frac > float(max_center_y_frac):
                continue
        metrics = target_patch_metrics(task_key, raw_image, bbox_raw)
        area = float(metrics["target_patch_area"])
        if area <= 4:
            continue
        center_penalty = abs(float(center_raw.get("x", 320.0)) - 320.0) + abs(float(center_raw.get("y", 180.0)) - 180.0)
        distance = _distance3(world_center, player_center)
        score = (
            1.0 * float(metrics["target_signal_score"])
            + 0.10 * area
            - 0.40 * center_penalty
            - 80.0 * distance
        )
        anchors.append(
            {
                "task_key": task_key,
                "goal_source": "visible_voxel_projection",
                "target_type": voxel_type,
                "world_center": world_center,
                "projection_basis": str(center_raw.get("basis_name", projection_basis or "auto")),
                "voxel_offset_raw": raw_offset,
                "voxel_key": projected.get("voxel_key"),
                "bbox_raw": bbox_raw,
                "target_center_raw": center_raw,
                "mask_source": projected.get("mask_source"),
                "mask_area_raw": int(projected.get("mask_area_raw", 0)),
                "target_patch_metrics": metrics,
                "target_projection_quality": projection_quality,
                "sort_distance": distance,
                "anchor_score": float(score),
            }
        )
    anchors.sort(key=lambda item: float(item["anchor_score"]), reverse=True)
    return anchors[: max(1, int(limit))]


def sample_pose_candidates_from_anchors(
    anchors: List[Dict[str, Any]],
    info: Dict[str, Any],
    *,
    distances: tuple[float, ...],
    height_offsets: tuple[float, ...],
    pitch_offsets: tuple[float, ...],
) -> List[Dict[str, Any]]:
    player_pose = _player_pose(info)
    base_y = float(player_pose["y"])
    candidates: List[Dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        target_center = anchor["world_center"]
        for distance in distances:
            for azimuth_deg in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
                azimuth_rad = np.radians(float(azimuth_deg))
                for height_offset in height_offsets:
                    snapped_height_offset = _snap_support_height_offset(float(height_offset))
                    camera_position = (
                        float(target_center[0]) + float(distance) * float(np.cos(azimuth_rad)),
                        float(base_y) + float(snapped_height_offset),
                        float(target_center[2]) + float(distance) * float(np.sin(azimuth_rad)),
                    )
                    base_yaw, base_pitch = _look_angles_for_target(camera_position, target_center)
                    support_block = _support_block_for_pose(
                        {
                            "x": float(camera_position[0]),
                            "y": float(camera_position[1]),
                            "z": float(camera_position[2]),
                        }
                    )
                    for pitch_offset in pitch_offsets:
                        candidates.append(
                            {
                                "anchor_index": int(anchor_index),
                                "goal_source": anchor["goal_source"],
                                "target_type": anchor["target_type"],
                                "target_world_center": target_center,
                                "projection_basis": anchor.get("projection_basis"),
                                "anchor_score": float(anchor.get("anchor_score", 0.0)),
                                "anchor_bbox_raw": anchor.get("bbox_raw"),
                                "anchor_target_center_raw": anchor.get("target_center_raw"),
                                "camera_position": camera_position,
                                "sampled_distance": float(distance),
                                "sampled_azimuth_deg": float(azimuth_deg),
                                "requested_height_offset": float(height_offset),
                                "sampled_height_offset": float(snapped_height_offset),
                                "sampled_pitch_offset": float(pitch_offset),
                                "yaw": float(base_yaw),
                                "pitch": float(base_pitch + float(pitch_offset)),
                                "support_block": support_block,
                            }
                        )
    return candidates


def sample_reference_sweep_candidates(
    anchors: List[Dict[str, Any]],
    reference_pose: Dict[str, float],
    max_count: int,
) -> List[Dict[str, Any]]:
    sweep_sequences = [
        {"name": "yaw_right_1", "yaw_delta": 12.0, "pitch_delta": 0.0},
        {"name": "yaw_left_1", "yaw_delta": -12.0, "pitch_delta": 0.0},
        {"name": "pitch_up_1", "yaw_delta": 0.0, "pitch_delta": -8.0},
        {"name": "pitch_down_1", "yaw_delta": 0.0, "pitch_delta": 8.0},
        {"name": "yaw_right_2", "yaw_delta": 24.0, "pitch_delta": 0.0},
        {"name": "yaw_left_2", "yaw_delta": -24.0, "pitch_delta": 0.0},
        {"name": "up_right", "yaw_delta": 12.0, "pitch_delta": -8.0},
        {"name": "up_left", "yaw_delta": -12.0, "pitch_delta": -8.0},
        {"name": "down_right", "yaw_delta": 12.0, "pitch_delta": 8.0},
        {"name": "down_left", "yaw_delta": -12.0, "pitch_delta": 8.0},
        {"name": "yaw_right_3", "yaw_delta": 36.0, "pitch_delta": 0.0},
        {"name": "yaw_left_3", "yaw_delta": -36.0, "pitch_delta": 0.0},
    ]
    candidates: List[Dict[str, Any]] = []
    for sweep_index, sweep in enumerate(sweep_sequences, start=1):
        yaw_delta = float(sweep["yaw_delta"])
        pitch_delta = float(sweep["pitch_delta"])
        target_yaw = float(reference_pose["yaw"]) + yaw_delta
        target_pitch = float(np.clip(float(reference_pose["pitch"]) + pitch_delta, -89.0, 89.0))
        for anchor_index, anchor in enumerate(anchors):
            candidates.append(
                {
                    "source": "reference_sweep",
                    "sweep_name": sweep["name"],
                    "sweep_index": int(sweep_index),
                    "sweep_actions": [],
                    "sweep_yaw_delta": yaw_delta,
                    "sweep_pitch_delta": pitch_delta,
                    "anchor_index": int(anchor_index),
                    "goal_source": anchor["goal_source"],
                    "target_type": anchor["target_type"],
                    "target_world_center": anchor["world_center"],
                    "projection_basis": anchor.get("projection_basis"),
                    "anchor_score": float(anchor.get("anchor_score", 0.0)),
                    "anchor_bbox_raw": anchor.get("bbox_raw"),
                    "anchor_target_center_raw": anchor.get("target_center_raw"),
                    "sampled_distance": 0.0,
                    "sampled_azimuth_deg": yaw_delta,
                    "sampled_height_offset": 0.0,
                    "sampled_pitch_offset": pitch_delta,
                    "camera_position": (
                        float(reference_pose["x"]),
                        float(reference_pose["y"]),
                        float(reference_pose["z"]),
                    ),
                    "yaw": target_yaw,
                    "pitch": target_pitch,
                }
            )
            if 0 < int(max_count) <= len(candidates):
                return candidates
    return candidates


def filter_anchors_by_distance(anchors: List[Dict[str, Any]], max_distance: float) -> List[Dict[str, Any]]:
    if float(max_distance) <= 0:
        return list(anchors)
    filtered = [anchor for anchor in anchors if float(anchor.get("sort_distance", float("inf"))) <= float(max_distance)]
    return filtered or list(anchors[:1])


def select_pose_candidates(candidates: List[Dict[str, Any]], max_count: int, mode: str) -> List[Dict[str, Any]]:
    if int(max_count) <= 0 or len(candidates) <= int(max_count):
        return list(candidates)
    if mode == "first":
        return list(candidates[: int(max_count)])
    # Interleave anchors first, then azimuths. This avoids max_count=3 taking
    # three nearby views from only the first anchor.
    groups: Dict[tuple[float, int], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        key = (float(candidate.get("sampled_azimuth_deg", 0.0)), int(candidate.get("anchor_index", 0)))
        groups.setdefault(key, []).append(candidate)
    ordered_keys = sorted(groups.keys(), key=lambda key: (key[0], key[1]))
    selected: List[Dict[str, Any]] = []
    round_index = 0
    while len(selected) < int(max_count):
        added = False
        for key in ordered_keys:
            group = groups[key]
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) >= int(max_count):
                    break
        if not added:
            break
        round_index += 1
    return selected


def _scale_point(point: Dict[str, Any] | None, src_shape, dst_shape):
    if not isinstance(point, dict):
        return None
    if "x" not in point or "y" not in point:
        return None
    src_h, src_w = int(src_shape[0]), int(src_shape[1])
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])
    if src_w <= 0 or src_h <= 0:
        return None
    x = float(point["x"]) * float(dst_w) / float(src_w)
    y = float(point["y"]) * float(dst_h) / float(src_h)
    return {"x": float(x), "y": float(y)}


def _scale_bbox(bbox, src_shape, dst_shape):
    if not bbox or len(bbox) != 4:
        return None
    src_h, src_w = int(src_shape[0]), int(src_shape[1])
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])
    if src_w <= 0 or src_h <= 0:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [
        int(round(x0 * float(dst_w) / float(src_w))),
        int(round(y0 * float(dst_h) / float(src_h))),
        int(round(x1 * float(dst_w) / float(src_w))),
        int(round(y1 * float(dst_h) / float(src_h))),
    ]


def overlay_bbox_and_centroid(image: np.ndarray, bbox=None, centroid=None) -> np.ndarray:
    overlay = image.copy()
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if isinstance(centroid, dict) and "x" in centroid and "y" in centroid:
        cx, cy = int(round(float(centroid["x"]))), int(round(float(centroid["y"])))
        cv2.drawMarker(
            overlay,
            (cx, cy),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return overlay


def overlay_visible_faces(image: np.ndarray, visible_faces=None, bbox=None, centroid=None) -> np.ndarray:
    overlay = overlay_bbox_and_centroid(image, bbox=bbox, centroid=centroid)
    palette = [
        (255, 180, 0),
        (0, 220, 255),
        (255, 80, 180),
        (120, 255, 120),
    ]
    if not isinstance(visible_faces, list):
        return overlay
    for face_idx, face in enumerate(visible_faces):
        vertices = face.get("polygon_vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        try:
            polygon = np.asarray([[int(pt[0]), int(pt[1])] for pt in vertices], dtype=np.int32)
        except Exception:
            continue
        color = palette[face_idx % len(palette)]
        cv2.polylines(overlay, [polygon.reshape(-1, 1, 2)], True, color, 2, lineType=cv2.LINE_AA)
        for vertex_index, point in enumerate(polygon.tolist(), start=1):
            px, py = int(point[0]), int(point[1])
            cv2.circle(overlay, (px, py), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(vertex_index),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
        polygon_center = face.get("polygon_center")
        if isinstance(polygon_center, dict) and "x" in polygon_center and "y" in polygon_center:
            cx = int(round(float(polygon_center["x"])))
            cy = int(round(float(polygon_center["y"])))
            cv2.circle(overlay, (cx, cy), 4, color, 1, lineType=cv2.LINE_AA)
            label = str(face.get("face_label", f"face{face_idx}"))
            cv2.putText(
                overlay,
                label,
                (cx + 6, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
    return overlay


def progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(round(width * float(current) / float(total)))
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total}"


def timing_summary(timing: Dict[str, float]) -> str:
    keys = ("tp_s", "settle_s", "capture_s", "crossview_s", "synthesize_s", "quality_s", "save_s", "total_s")
    parts = []
    for key in keys:
        if key in timing:
            parts.append(f"{key[:-2]}={float(timing[key]):.3f}s")
    return " ".join(parts)


def reason_slug(reason: str) -> str:
    keep = []
    for ch in str(reason).lower():
        if ch.isalnum():
            keep.append(ch)
        elif keep and keep[-1] != "_":
            keep.append("_")
    slug = "".join(keep).strip("_")
    return slug[:64] or "unknown"


def write_candidate_snapshot(
    *,
    candidate_dir: Path,
    task_key: str,
    segment_type: str,
    pose_index: int,
    status: str,
    goal_image: np.ndarray | None = None,
    candidate_pov: np.ndarray | None = None,
    bbox_obs=None,
    bbox_raw=None,
    center_obs=None,
    center_raw=None,
    score: float | None = None,
    metadata: Dict[str, Any] | None = None,
):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    paths: Dict[str, str] = {}

    if goal_image is not None:
        goal_image = np.asarray(goal_image, dtype=np.uint8)
        goal_image_path = candidate_dir / "goal_image.png"
        bbox_overlay_path = candidate_dir / "target_bbox_overlay.png"
        centroid_overlay_path = candidate_dir / "target_centroid_bbox_overlay.png"
        Image.fromarray(goal_image).save(goal_image_path)
        Image.fromarray(overlay_bbox_and_centroid(goal_image, bbox_obs, None)).save(bbox_overlay_path)
        Image.fromarray(overlay_bbox_and_centroid(goal_image, bbox_obs, center_obs)).save(centroid_overlay_path)
        paths.update(
            {
                "goal_image_path": str(goal_image_path.resolve()),
                "target_bbox_overlay_path": str(bbox_overlay_path.resolve()),
                "target_centroid_bbox_overlay_path": str(centroid_overlay_path.resolve()),
            }
        )

    if candidate_pov is not None:
        candidate_pov = np.asarray(candidate_pov, dtype=np.uint8)
        candidate_pov_path = candidate_dir / "candidate_pov.png"
        candidate_pov_overlay_path = candidate_dir / "candidate_pov_centroid_bbox_overlay.png"
        candidate_pov_faces_overlay_path = candidate_dir / "candidate_pov_visible_faces_overlay.png"
        Image.fromarray(candidate_pov).save(candidate_pov_path)
        Image.fromarray(overlay_bbox_and_centroid(candidate_pov, bbox_raw, center_raw)).save(candidate_pov_overlay_path)
        visible_faces = metadata.get("visible_faces")
        Image.fromarray(overlay_visible_faces(candidate_pov, visible_faces=visible_faces, bbox=bbox_raw, centroid=center_raw)).save(
            candidate_pov_faces_overlay_path
        )
        paths.update(
            {
                "candidate_pov_path": str(candidate_pov_path.resolve()),
                "candidate_pov_centroid_bbox_overlay_path": str(candidate_pov_overlay_path.resolve()),
                "candidate_pov_visible_faces_overlay_path": str(candidate_pov_faces_overlay_path.resolve()),
            }
        )

    if goal_image is not None and (center_obs is not None or bbox_obs is not None):
        sam_prompt_path = candidate_dir / "sam_prompt.json"
        sam_prompt = {
            "image_space": "obs_224",
            "point": center_obs,
            "box": bbox_obs,
            "raw_pov_space": "pov",
            "raw_point": center_raw,
            "raw_box": bbox_raw,
            "task_key": task_key,
            "segment_type": segment_type,
            "goal_image_path": paths.get("goal_image_path"),
        }
        sam_prompt_path.write_text(json.dumps(_sanitize_debug_value(sam_prompt), indent=2, ensure_ascii=False), encoding="utf-8")
        paths["sam_prompt_path"] = str(sam_prompt_path.resolve())

    snapshot = {
        "task_key": task_key,
        "segment_type": segment_type,
        "pose_index": int(pose_index),
        "status": str(status),
        "score": None if score is None else float(score),
        **paths,
        "goal_metadata": _sanitize_debug_value(metadata),
    }
    metadata_path = candidate_dir / "metadata.json"
    metadata_path.write_text(json.dumps(_sanitize_debug_value(snapshot), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["metadata_path"] = str(metadata_path.resolve())
    return paths


def write_projection_basis_debug(
    *,
    candidate_dir: Path,
    candidate_pov: np.ndarray,
    candidate_pose: Dict[str, float],
    pose_candidate: Dict[str, Any],
    candidate_info: Dict[str, Any] | None = None,
) -> None:
    target_center = pose_candidate.get("target_world_center")
    if not target_center or len(target_center) != 3:
        return
    debug_dir = candidate_dir / "projection_basis_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    target_center = tuple(float(value) for value in target_center)
    entries = []
    basis_names = ["minecraft"] + [basis_name for basis_name, _, _, _ in _camera_basis_variants(candidate_pose)]
    for basis_name in dict.fromkeys(basis_names):
        projected = _project_visible_target_with_basis(
            player_pose=candidate_pose,
            info=candidate_info,
            target_center=target_center,
            image_shape=candidate_pov.shape,
            basis_name=basis_name,
        )
        if projected is None:
            entries.append({"basis_name": basis_name, "projected": False})
            continue
        bbox_raw = projected["bbox_raw"]
        center_raw = projected["target_center_raw"]
        visible_faces = projected.get("visible_faces", [])
        overlay_path = debug_dir / f"{basis_name}.png"
        Image.fromarray(
            overlay_visible_faces(candidate_pov, visible_faces=visible_faces, bbox=bbox_raw, centroid=center_raw)
        ).save(overlay_path)
        entries.append(
            {
                "basis_name": basis_name,
                "projected": True,
                "bbox_raw": list(bbox_raw),
                "target_center_raw": _sanitize_debug_value(center_raw),
                "mask_source": projected.get("mask_source"),
                "mask_area_raw": int(projected.get("mask_area_raw", 0)),
                "voxel_key": _sanitize_debug_value(projected.get("voxel_key")),
                "num_visible_faces": int(projected.get("num_visible_faces", 0)),
                "visible_faces": _sanitize_debug_value(visible_faces),
                "overlay_path": str(overlay_path.resolve()),
            }
        )
    (debug_dir / "projection_basis_debug.json").write_text(
        json.dumps(
            {
                "target_world_center": _sanitize_debug_value(target_center),
                "candidate_player_pose": _sanitize_debug_value(candidate_pose),
                "pose_candidate": _sanitize_debug_value(pose_candidate),
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    protocol = resolve_interaction_protocol(args.protocol)
    warmup_noop_steps = protocol.warmup_noop_steps if args.warmup_noop_steps is None else int(args.warmup_noop_steps)
    task_specs = resolve_interaction_task_specs([args.task], env_source=args.env_source)
    task_specs = apply_protocol_to_task_specs(task_specs, protocol_name=args.protocol)
    task_spec = task_specs[0]
    task_key = task_spec.task_key or args.task
    if task_key not in AUTO_GOAL_SUPPORTED_TASKS:
        supported = ", ".join(sorted(AUTO_GOAL_SUPPORTED_TASKS))
        raise NotImplementedError(f"Supported auto-goal candidate tasks: {supported}. Got {task_key}.")

    refresh_task_configs = bool(Path(args.task_group_path).is_dir())
    file_list = prepare_task_configs(args.task_group, path=args.task_group_path, refresh=refresh_task_configs)
    name_file_mapping = {name: file for name, file in file_list.items()}
    if task_spec.task_config_name not in name_file_mapping:
        raise KeyError(f"Task config {task_spec.task_config_name} not found in prepared task configs.")

    task_path = Path(name_file_mapping[task_spec.task_config_name])
    with task_path.open("r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f) or {}
    task_config.pop("reference_video", None)

    run_dir = Path(args.out_dir) / task_key / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(task_key, task_config)
    env = None
    obs = None
    info = None
    last_reset_exc = None
    reset_retries = max(1, int(args.reset_retries))
    for reset_attempt in range(1, reset_retries + 1):
        env = MinecraftSim(
            seed=int(args.base_seed),
            preferred_spawn_biome="plains",
            callbacks=callbacks,
            obs_size=(224, 224),
            action_type="env",
        )
        try:
            print(f"[candidate-dump] env reset attempt {reset_attempt}/{reset_retries}", flush=True)
            obs, info = env.reset()
            break
        except Exception as exc:
            last_reset_exc = exc
            print(
                f"[candidate-dump] env reset failed attempt {reset_attempt}/{reset_retries}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                env.close()
            except Exception:
                pass
            env = None
            time.sleep(2.0)
    if env is None or obs is None or info is None:
        raise last_reset_exc or RuntimeError("Failed to reset Minecraft env")

    try:
        for _ in range(int(max(0, warmup_noop_steps))):
            obs, _, terminated, truncated, info = env.step(env.noop_action())
            if terminated or truncated:
                break
        reference_obs = np.asarray(obs["image"], dtype=np.uint8).copy()
        reference_pov = np.asarray(info["pov"], dtype=np.uint8).copy()
        reference_pose = _player_pose(info)

        Image.fromarray(reference_obs).save(run_dir / "reference_obs.png")
        Image.fromarray(reference_pov).save(run_dir / "reference_pov.png")

        projection_basis = None if str(args.projection_basis).lower() == "auto" else str(args.projection_basis)
        configured_anchors = configured_target_anchors(
            task_key,
            task_config,
            info,
            reference_pov,
            projection_basis=projection_basis,
            allow_edge_clipped=bool(args.allow_edge_clipped_anchors),
            max_center_y_frac=float(args.max_anchor_center_y_frac),
        )
        if configured_anchors:
            raw_anchors = configured_anchors
        else:
            raw_anchors = extract_visible_target_anchors(
                task_key,
                info,
                reference_pov,
                int(max(args.anchor_top_k, 12)),
                projection_basis=projection_basis,
                allow_edge_clipped=bool(args.allow_edge_clipped_anchors),
                max_center_y_frac=float(args.max_anchor_center_y_frac),
            )
        anchors = filter_anchors_by_distance(raw_anchors, float(args.max_anchor_distance))[: max(1, int(args.anchor_top_k))]
        if not anchors:
            raise RuntimeError(f"No projected target anchors found for task={task_key} seed={int(args.base_seed)}")
        (run_dir / "reference_target_anchors_all.json").write_text(
            json.dumps(_sanitize_debug_value(raw_anchors), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "reference_target_anchors.json").write_text(
            json.dumps(_sanitize_debug_value(anchors), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for anchor_rank, anchor in enumerate(anchors, start=1):
            overlay = overlay_bbox_and_centroid(
                reference_pov,
                anchor.get("bbox_raw"),
                anchor.get("target_center_raw"),
            )
            Image.fromarray(overlay).save(run_dir / f"reference_anchor_{anchor_rank:02d}_overlay.png")

        candidate_distances = parse_float_csv(args.candidate_distances, name="candidate-distances")
        candidate_height_offsets = parse_float_csv(args.candidate_height_offsets, name="candidate-height-offsets")
        candidate_pitch_offsets = parse_float_csv(args.candidate_pitch_offsets, name="candidate-pitch-offsets")
        if args.candidate_source == "reference_sweep":
            all_pose_candidates = sample_reference_sweep_candidates(anchors, reference_pose, int(args.max_pose_candidates))
            pose_candidates = all_pose_candidates
        else:
            all_pose_candidates = sample_pose_candidates_from_anchors(
                anchors,
                info,
                distances=candidate_distances,
                height_offsets=candidate_height_offsets,
                pitch_offsets=candidate_pitch_offsets,
            )
            pose_candidates = select_pose_candidates(
                all_pose_candidates,
                int(args.max_pose_candidates),
                str(args.pose_selection),
            )

        progress_dir = run_dir / "candidate_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        total_poses = len(pose_candidates)
        print(
            f"[candidate-dump] run_dir={run_dir} task={task_key} seed={int(args.base_seed)} anchors={len(anchors)}/{len(raw_anchors)} anchor_source={'configured' if configured_anchors else 'voxels'} max_anchor_distance={float(args.max_anchor_distance)} allow_edge_clipped_anchors={bool(args.allow_edge_clipped_anchors)} max_anchor_center_y_frac={float(args.max_anchor_center_y_frac)} poses={total_poses}/{len(all_pose_candidates)} source={args.candidate_source} projection_basis={args.projection_basis} distances={candidate_distances} height_offsets={candidate_height_offsets} selection={args.pose_selection}",
            flush=True,
        )

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for pose_index, pose_candidate in enumerate(pose_candidates, start=1):
            pose_t0 = time.perf_counter()
            timing: Dict[str, float] = {}
            capture_diagnostics: Dict[str, Any] = {}
            print(
                f"[candidate-dump] {progress_bar(pose_index - 1, total_poses)} pose={pose_index:03d} probing",
                flush=True,
            )
            source = str(pose_candidate.get("source", "target_teleport"))
            expected_pose = None
            if source == "reference_sweep":
                expected_pose = {
                    "x": float(reference_pose["x"]),
                    "y": float(reference_pose["y"]),
                    "z": float(reference_pose["z"]),
                    "yaw": float(pose_candidate["yaw"]),
                    "pitch": float(pose_candidate["pitch"]),
                }
            else:
                expected_pose = {
                    "x": float(pose_candidate["camera_position"][0]),
                    "y": float(pose_candidate["camera_position"][1]),
                    "z": float(pose_candidate["camera_position"][2]),
                    "yaw": float(pose_candidate["yaw"]),
                    "pitch": float(pose_candidate["pitch"]),
                }
            t0 = time.perf_counter()
            try:
                if source == "reference_sweep":
                    reference_reset_pose = {
                        "x": float(reference_pose["x"]),
                        "y": float(reference_pose["y"]),
                        "z": float(reference_pose["z"]),
                        "yaw": float(reference_pose["yaw"]),
                        "pitch": float(reference_pose["pitch"]),
                    }
                    _teleport_to_pose(env, reference_reset_pose)
                    ref_obs, ref_info, terminated, truncated, ref_diag = _capture_stable_frame(
                        env,
                        reference_reset_pose,
                        min_wait_steps=max(1, int(AUTO_GOAL_POST_TP_SETTLE_STEPS)),
                        max_wait_steps=max(6, int(AUTO_GOAL_POST_TP_SETTLE_STEPS) + 6),
                    )
                    capture_diagnostics["reference_reset"] = _sanitize_debug_value(ref_diag)
                    if terminated or truncated or ref_obs is None or ref_info is None:
                        raise RuntimeError("reference_reset_failed_or_terminated")
                    _teleport_to_pose(env, expected_pose)
                else:
                    support_block = pose_candidate.get("support_block")
                    _ensure_support_block(env, support_block)
                    capture_diagnostics["support_block"] = _sanitize_debug_value(support_block)
                    _teleport_to_pose(env, expected_pose)
            except Exception as exc:
                timing["tp_s"] = time.perf_counter() - t0
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": f"tp_failed: {exc}",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_tp_failed"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status="rejected_tp_failed",
                    metadata={
                            "reason": f"tp_failed: {exc}",
                            "pose_candidate": _sanitize_debug_value(pose_candidate),
                            "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                            "timing": _sanitize_debug_value(timing),
                        },
                    )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected tp_failed saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue
            timing["tp_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            obs, info, terminated, truncated, probe_diag = _capture_stable_frame(
                env,
                expected_pose,
                min_wait_steps=max(1, int(AUTO_GOAL_POST_TP_SETTLE_STEPS)),
                max_wait_steps=max(8, int(AUTO_GOAL_POST_TP_SETTLE_STEPS) + 8),
            )
            capture_diagnostics["probe_capture"] = _sanitize_debug_value(probe_diag)
            timing["settle_s"] = time.perf_counter() - t0
            if terminated or truncated or obs is None or info is None:
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": "capture_stabilization_failed_or_terminated",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_capture"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status="rejected_capture_stabilization_failed_or_terminated",
                    metadata={
                        "reason": "capture_stabilization_failed_or_terminated",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    },
                )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected capture saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue

            t0 = time.perf_counter()
            candidate_obs = np.asarray(obs["image"], dtype=np.uint8).copy()
            candidate_pov = np.asarray(info["pov"], dtype=np.uint8).copy()
            candidate_pose = _player_pose(info)
            cam_x, cam_y, cam_z = float(candidate_pose["x"]), float(candidate_pose["y"]), float(candidate_pose["z"])
            yaw = float(candidate_pose["yaw"])
            pitch = float(candidate_pose["pitch"])
            timing["capture_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            crossview_metrics = _crossview_difference_metrics(
                reference_pose,
                reference_obs,
                candidate_pose,
                candidate_obs,
            )
            timing["crossview_s"] = time.perf_counter() - t0
            if not is_valid_dump_crossview(crossview_metrics, require_camera_displacement=(source != "reference_sweep")):
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": "crossview_difference_below_threshold",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_crossview"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status="rejected_crossview_difference_below_threshold",
                    goal_image=candidate_obs,
                    candidate_pov=candidate_pov,
                    metadata={
                        "reason": "crossview_difference_below_threshold",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    },
                )
                write_projection_basis_debug(
                    candidate_dir=candidate_dir,
                    candidate_pov=candidate_pov,
                    candidate_pose=candidate_pose,
                    pose_candidate=pose_candidate,
                    candidate_info={"voxels": info.get("voxels") if isinstance(info, dict) else None},
                )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected crossview saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue

            t0 = time.perf_counter()
            target_projection = exact_target_projection(
                pose_candidate=pose_candidate,
                candidate_pose=candidate_pose,
                candidate_info=info,
                raw_image=candidate_pov,
                obs_image=candidate_obs,
            )
            timing["synthesize_s"] = time.perf_counter() - t0
            if target_projection is None:
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": "target_projection_failed",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_target"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status="rejected_target_not_visible_or_projection_failed",
                    goal_image=candidate_obs,
                    candidate_pov=candidate_pov,
                    metadata={
                        "reason": "target_projection_failed",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    },
                )
                write_projection_basis_debug(
                    candidate_dir=candidate_dir,
                    candidate_pov=candidate_pov,
                    candidate_pose=candidate_pose,
                    pose_candidate=pose_candidate,
                    candidate_info={"voxels": info.get("voxels") if isinstance(info, dict) else None},
                )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected target saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue

            t0 = time.perf_counter()
            appearance_metrics = image_quality_metrics(candidate_obs)
            center_raw = target_projection.get("target_center_raw")
            center_obs = target_projection.get("target_center_obs")
            bbox_raw = target_projection.get("bbox_raw")
            bbox_obs = target_projection.get("bbox_obs")
            target_metrics = target_patch_metrics(task_key, candidate_pov, bbox_raw)
            projection_quality = target_projection_quality(candidate_pov.shape, bbox_raw)
            timing["quality_s"] = time.perf_counter() - t0
            projection_rejection_reason = None
            if not bool(projection_quality["target_bbox_valid"]):
                projection_rejection_reason = "target_bbox_invalid"
            elif float(projection_quality["target_bbox_area_frac"]) < float(args.min_target_bbox_frac):
                projection_rejection_reason = "target_bbox_too_small"
            elif float(projection_quality["target_bbox_area_frac"]) > float(args.max_target_bbox_frac):
                projection_rejection_reason = "target_bbox_too_large"
            elif int(projection_quality["target_bbox_height_px"]) < int(args.min_target_bbox_height_px):
                projection_rejection_reason = "target_bbox_too_short"
            elif int(args.min_target_bbox_width_px) > 0 and int(projection_quality["target_bbox_width_px"]) < int(
                args.min_target_bbox_width_px
            ):
                projection_rejection_reason = "target_bbox_too_narrow"
            elif bool(args.reject_edge_clipped_bbox) and bool(projection_quality["target_bbox_edge_clipped"]):
                projection_rejection_reason = "target_bbox_edge_clipped"
            if projection_rejection_reason is not None:
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": projection_rejection_reason,
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "target_projection": _sanitize_debug_value(target_projection),
                        "target_patch_metrics": _sanitize_debug_value(target_metrics),
                        "target_projection_quality": _sanitize_debug_value(projection_quality),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_{reason_slug(projection_rejection_reason)}"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status=f"rejected_{projection_rejection_reason}",
                    goal_image=candidate_obs,
                    candidate_pov=candidate_pov,
                    bbox_obs=bbox_obs,
                    bbox_raw=bbox_raw,
                    center_obs=center_obs,
                    center_raw=center_raw,
                    metadata={
                        "reason": projection_rejection_reason,
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "target_projection": _sanitize_debug_value(target_projection),
                        "target_patch_metrics": _sanitize_debug_value(target_metrics),
                        "target_projection_quality": _sanitize_debug_value(projection_quality),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    },
                )
                write_projection_basis_debug(
                    candidate_dir=candidate_dir,
                    candidate_pov=candidate_pov,
                    candidate_pose=candidate_pose,
                    pose_candidate=pose_candidate,
                    candidate_info={"voxels": info.get("voxels") if isinstance(info, dict) else None},
                )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected {projection_rejection_reason} saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue
            if not is_valid_image_candidate(appearance_metrics):
                timing["total_s"] = time.perf_counter() - pose_t0
                rejected.append(
                    {
                        "pose_index": int(pose_index),
                        "reason": "image_quality_below_threshold",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "appearance_metrics": _sanitize_debug_value(appearance_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    }
                )
                candidate_dir = progress_dir / f"pose_{pose_index:03d}_rejected_image"
                t_save = time.perf_counter()
                write_candidate_snapshot(
                    candidate_dir=candidate_dir,
                    task_key=task_key,
                    segment_type=str(task_spec.subtasks[0].interaction_type),
                    pose_index=pose_index,
                    status="rejected_image_quality_below_threshold",
                    goal_image=candidate_obs,
                    candidate_pov=candidate_pov,
                    bbox_obs=bbox_obs,
                    bbox_raw=bbox_raw,
                    center_obs=center_obs,
                    center_raw=center_raw,
                    metadata={
                        "reason": "image_quality_below_threshold",
                        "pose_candidate": _sanitize_debug_value(pose_candidate),
                        "crossview_metrics": _sanitize_debug_value(crossview_metrics),
                        "appearance_metrics": _sanitize_debug_value(appearance_metrics),
                        "target_projection": _sanitize_debug_value(target_projection),
                        "target_patch_metrics": _sanitize_debug_value(target_metrics),
                        "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                        "timing": _sanitize_debug_value(timing),
                    },
                )
                write_projection_basis_debug(
                    candidate_dir=candidate_dir,
                    candidate_pov=candidate_pov,
                    candidate_pose=candidate_pose,
                    pose_candidate=pose_candidate,
                    candidate_info={"voxels": info.get("voxels") if isinstance(info, dict) else None},
                )
                timing["save_s"] = time.perf_counter() - t_save
                print(
                    f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} rejected image saved={candidate_dir}"
                    + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                    flush=True,
                )
                continue

            metadata = {
                "task_key": task_key,
                "segment_type": str(task_spec.subtasks[0].interaction_type),
                "goal_mode": "manual_image_selection",
                "goal_source": str(pose_candidate.get("source", "target_teleport")),
                "probe_name": f"sampled_pose_{pose_index:03d}",
                "probe_step": int(pose_index),
                "candidate_source": str(pose_candidate.get("source", "target_teleport")),
                "pose_candidate": _sanitize_debug_value(pose_candidate),
                "sweep_name": pose_candidate.get("sweep_name"),
                "sweep_actions": _sanitize_debug_value(pose_candidate.get("sweep_actions")),
                "sweep_yaw_delta": _sanitize_debug_value(pose_candidate.get("sweep_yaw_delta")),
                "sweep_pitch_delta": _sanitize_debug_value(pose_candidate.get("sweep_pitch_delta")),
                "projection_basis": _sanitize_debug_value(pose_candidate.get("projection_basis")),
                "intended_yaw": round(float(pose_candidate.get("yaw", yaw)), 4),
                "intended_pitch": round(float(pose_candidate.get("pitch", pitch)), 4),
                "sampled_camera_position": [round(float(cam_x), 4), round(float(cam_y), 4), round(float(cam_z), 4)],
                "sampled_yaw": round(float(yaw), 4),
                "sampled_pitch": round(float(pitch), 4),
                "sampled_distance": float(pose_candidate["sampled_distance"]),
                "sampled_azimuth_deg": float(pose_candidate["sampled_azimuth_deg"]),
                "sampled_height_offset": float(pose_candidate["sampled_height_offset"]),
                "requested_height_offset": float(
                    pose_candidate.get("requested_height_offset", pose_candidate["sampled_height_offset"])
                ),
                "sampled_pitch_offset": float(pose_candidate["sampled_pitch_offset"]),
                "support_block": _sanitize_debug_value(pose_candidate.get("support_block")),
                "target_type": str(pose_candidate.get("target_type", "")),
                "target_world_center": _sanitize_debug_value(
                    pose_candidate.get("target_world_center", (0.0, 0.0, 0.0))
                ),
                "target_center_raw": _sanitize_debug_value(center_raw),
                "target_center_obs": _sanitize_debug_value(center_obs),
                "bbox_raw": _sanitize_debug_value(bbox_raw),
                "mask_bbox_raw": _sanitize_debug_value(bbox_raw),
                "bbox_obs": _sanitize_debug_value(bbox_obs),
                "bbox_obs_from_raw": _sanitize_debug_value(
                    _scale_bbox(
                        bbox_raw,
                        candidate_pov.shape,
                        candidate_obs.shape,
                    )
                ),
                "sam_point_prompt_obs": _sanitize_debug_value(center_obs),
                "sam_box_prompt_obs": _sanitize_debug_value(bbox_obs),
                "sam_point_prompt_raw": _sanitize_debug_value(center_raw),
                "sam_box_prompt_raw": _sanitize_debug_value(bbox_raw),
                "mask_source": _sanitize_debug_value(target_projection.get("mask_source")),
                "mask_area_raw": int(target_projection.get("mask_area_raw", 0)),
                "voxel_key": _sanitize_debug_value(target_projection.get("voxel_key")),
                "num_visible_faces": int(target_projection.get("num_visible_faces", 0)),
                "visible_faces": _sanitize_debug_value(target_projection.get("visible_faces", [])),
                "camera_displacement": round(float(crossview_metrics["camera_displacement"]), 4),
                "yaw_delta_deg": round(float(crossview_metrics["yaw_delta_deg"]), 4),
                "pitch_delta_deg": round(float(crossview_metrics["pitch_delta_deg"]), 4),
                "mean_abs_diff": round(float(crossview_metrics["mean_abs_diff"]), 4),
                "changed_pixel_frac": round(float(crossview_metrics["changed_pixel_frac"]), 6),
                "image_mean": round(float(appearance_metrics["image_mean"]), 4),
                "image_std": round(float(appearance_metrics["image_std"]), 4),
                "non_black_frac": round(float(appearance_metrics["non_black_frac"]), 6),
                "target_patch_metrics": _sanitize_debug_value(target_metrics),
                "target_projection_quality": _sanitize_debug_value(projection_quality),
                "candidate_player_pose": _sanitize_debug_value(candidate_pose),
                "capture_diagnostics": _sanitize_debug_value(capture_diagnostics),
                "timing": _sanitize_debug_value(timing),
            }
            score = image_candidate_score(
                image_metrics=appearance_metrics,
                crossview_metrics=crossview_metrics,
                target_center_raw=center_raw,
            ) + 5.0 * float(target_metrics.get("target_signal_score", 0.0))
            candidate_dir = progress_dir / f"pose_{pose_index:03d}_accepted"
            timing["total_s"] = time.perf_counter() - pose_t0
            metadata["timing"] = _sanitize_debug_value(timing)
            t_save = time.perf_counter()
            write_candidate_snapshot(
                candidate_dir=candidate_dir,
                task_key=task_key,
                segment_type=str(task_spec.subtasks[0].interaction_type),
                pose_index=pose_index,
                status="accepted",
                goal_image=candidate_obs,
                candidate_pov=candidate_pov,
                bbox_obs=bbox_obs,
                bbox_raw=bbox_raw,
                center_obs=center_obs,
                center_raw=center_raw,
                score=score,
                metadata=metadata,
            )
            write_projection_basis_debug(
                candidate_dir=candidate_dir,
                candidate_pov=candidate_pov,
                candidate_pose=candidate_pose,
                pose_candidate=pose_candidate,
                candidate_info={"voxels": info.get("voxels") if isinstance(info, dict) else None},
            )
            timing["save_s"] = time.perf_counter() - t_save
            print(
                f"[candidate-dump] {progress_bar(pose_index, total_poses)} pose={pose_index:03d} accepted score={score:.2f} saved={candidate_dir}"
                + (f" {timing_summary(timing)}" if args.profile_timing else ""),
                flush=True,
            )
            accepted.append(
                {
                    "pose_index": int(pose_index),
                    "score": float(score),
                    "goal_image": candidate_obs,
                    "candidate_pov": candidate_pov,
                    "metadata": metadata,
                    "crossview_metrics": crossview_metrics,
                    "appearance_metrics": appearance_metrics,
                    "bbox_obs": bbox_obs,
                    "bbox_raw": bbox_raw,
                    "target_center_raw": center_raw,
                    "candidate_voxels": info.get("voxels") if isinstance(info, dict) else None,
                }
            )
    finally:
        env.close()

    accepted.sort(key=lambda row: float(row["score"]), reverse=True)
    kept = accepted[: max(1, int(args.top_k))]

    manifest = {
        "task_key": task_key,
        "task_config_name": task_spec.task_config_name,
        "seed": int(args.base_seed),
        "warmup_noop_steps": int(warmup_noop_steps),
        "candidate_source": str(args.candidate_source),
        "projection_basis": str(args.projection_basis),
        "allow_edge_clipped_anchors": bool(args.allow_edge_clipped_anchors),
        "max_anchor_center_y_frac": float(args.max_anchor_center_y_frac),
        "candidate_progress_dir": str((run_dir / "candidate_progress").resolve()),
        "num_pose_candidates": int(len(pose_candidates)),
        "num_accepted": int(len(accepted)),
        "num_saved": int(len(kept)),
        "candidates": [],
        "rejected": rejected,
    }

    for rank, candidate in enumerate(kept, start=1):
        candidate_dir = run_dir / f"candidate_{rank:02d}_pose_{candidate['pose_index']:03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        goal_image = np.asarray(candidate["goal_image"], dtype=np.uint8)

        goal_image_path = candidate_dir / "goal_image.png"
        bbox_overlay_path = candidate_dir / "target_bbox_overlay.png"
        centroid_overlay_path = candidate_dir / "target_centroid_bbox_overlay.png"
        candidate_pov_path = candidate_dir / "candidate_pov.png"
        candidate_pov_overlay_path = candidate_dir / "candidate_pov_centroid_bbox_overlay.png"
        sam_prompt_path = candidate_dir / "sam_prompt.json"
        metadata_path = candidate_dir / "metadata.json"

        Image.fromarray(goal_image).save(goal_image_path)
        candidate_pov = np.asarray(candidate["candidate_pov"], dtype=np.uint8)
        goal_metadata = dict(candidate["metadata"])
        center_obs = goal_metadata.get("target_center_obs")
        center_raw = goal_metadata.get("target_center_raw")
        bbox_obs = candidate.get("bbox_obs")
        bbox_raw = candidate.get("bbox_raw")
        Image.fromarray(overlay_bbox_and_centroid(goal_image, bbox_obs, None)).save(bbox_overlay_path)
        Image.fromarray(overlay_bbox_and_centroid(goal_image, bbox_obs, center_obs)).save(centroid_overlay_path)
        Image.fromarray(candidate_pov).save(candidate_pov_path)
        Image.fromarray(overlay_bbox_and_centroid(candidate_pov, bbox_raw, center_raw)).save(candidate_pov_overlay_path)

        sam_prompt = {
            "image_space": "obs_224",
            "point": center_obs,
            "box": bbox_obs,
            "raw_pov_space": "pov",
            "raw_point": center_raw,
            "raw_box": bbox_raw,
            "task_key": task_key,
            "segment_type": str(task_spec.subtasks[0].interaction_type),
            "goal_image_path": str(goal_image_path.resolve()),
        }
        sam_prompt_path.write_text(json.dumps(_sanitize_debug_value(sam_prompt), indent=2, ensure_ascii=False), encoding="utf-8")

        metadata = {
            "rank": int(rank),
            "pose_index": int(candidate["pose_index"]),
            "score": float(candidate["score"]),
            "task_key": task_key,
            "segment_type": str(task_spec.subtasks[0].interaction_type),
            "goal_image_path": str(goal_image_path.resolve()),
            "target_bbox_overlay_path": str(bbox_overlay_path.resolve()),
            "target_centroid_bbox_overlay_path": str(centroid_overlay_path.resolve()),
            "candidate_pov_path": str(candidate_pov_path.resolve()),
            "candidate_pov_centroid_bbox_overlay_path": str(candidate_pov_overlay_path.resolve()),
            "sam_prompt_path": str(sam_prompt_path.resolve()),
            "goal_metadata": _sanitize_debug_value(candidate["metadata"]),
            "crossview_metrics": _sanitize_debug_value(candidate["crossview_metrics"]),
            "appearance_metrics": _sanitize_debug_value(candidate["appearance_metrics"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        write_projection_basis_debug(
            candidate_dir=candidate_dir,
            candidate_pov=candidate_pov,
            candidate_pose=goal_metadata.get("candidate_player_pose", {}),
            pose_candidate=goal_metadata.get("pose_candidate", goal_metadata),
            candidate_info={"voxels": candidate.get("candidate_voxels")},
        )
        manifest["candidates"].append(
            {
                "rank": int(rank),
                "pose_index": int(candidate["pose_index"]),
                "score": float(candidate["score"]),
                "candidate_dir": str(candidate_dir.resolve()),
                "goal_image_path": str(goal_image_path.resolve()),
                "target_bbox_overlay_path": str(bbox_overlay_path.resolve()),
                "target_centroid_bbox_overlay_path": str(centroid_overlay_path.resolve()),
                "candidate_pov_path": str(candidate_pov_path.resolve()),
                "candidate_pov_centroid_bbox_overlay_path": str(candidate_pov_overlay_path.resolve()),
                "sam_prompt_path": str(sam_prompt_path.resolve()),
            }
        )

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **{k: manifest[k] for k in ("task_key", "seed", "num_pose_candidates", "num_accepted", "num_saved")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
