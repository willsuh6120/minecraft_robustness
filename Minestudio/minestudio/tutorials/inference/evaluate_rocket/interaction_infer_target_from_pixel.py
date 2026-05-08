import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from minestudio.benchmark import prepare_task_configs
from minestudio.simulator import MinecraftSim
from minestudio.tutorials.inference.evaluate_rocket.crossview_utils import (
    AUTO_GOAL_BLOCK_TYPES,
    DEFAULT_EYE_HEIGHT,
    DEFAULT_FOV_DEG,
    _get_basis_by_name,
    _normalize_name,
    _player_pose,
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
from minestudio.tutorials.inference.evaluate_rocket.interaction_dump_crossview_candidates import (
    _project_visible_target_with_basis,
    build_callbacks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer a target block 3D world coordinate from a clicked raw-image pixel."
    )
    parser.add_argument("--env-source", type=str, default="rocket2_official")
    parser.add_argument("--task-group", type=str, default="rocket2_official")
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--task-group-path", type=str, default="/home/gyulab/envgen2/ROCKET-2/env_conf")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--warmup-noop-steps", type=int, default=None)
    parser.add_argument("--pixel-x", type=float, required=True)
    parser.add_argument("--pixel-y", type=float, required=True)
    parser.add_argument(
        "--polygon",
        type=str,
        default="",
        help='Optional raw-image polygon vertices, e.g. "283,226 356,226 356,304 283,304"',
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="",
        help="Optional explicit target type, e.g. coal_ore. Defaults to task-specific mine target.",
    )
    parser.add_argument("--projection-basis", type=str, default="minecraft")
    parser.add_argument("--reset-retries", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/gyulab/envgen2/Minestudio/outputs/evaluate_rocket/infer_target_from_pixel",
    )
    return parser.parse_args()


def _parse_polygon(value: str) -> List[Tuple[float, float]]:
    vertices: List[Tuple[float, float]] = []
    for piece in str(value or "").split():
        if "," not in piece:
            continue
        x_str, y_str = piece.split(",", 1)
        vertices.append((float(x_str), float(y_str)))
    return vertices


def _polygon_center(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    polygon = np.asarray(vertices, dtype=np.float32)
    if polygon.shape[0] < 3:
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        return (float(sum(xs) / max(1, len(xs))), float(sum(ys) / max(1, len(ys))))
    moments = cv2.moments(polygon)
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
    return (float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1])))


def _polygon_bbox(vertices: List[Tuple[float, float]]) -> Optional[Tuple[int, int, int, int]]:
    if not vertices:
        return None
    xs = [float(point[0]) for point in vertices]
    ys = [float(point[1]) for point in vertices]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def _bbox_iou(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = float((ix1 - ix0 + 1) * (iy1 - iy0 + 1))
    area_a = float((ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    area_b = float((bx1 - bx0 + 1) * (by1 - by0 + 1))
    denom = max(1.0, area_a + area_b - inter)
    return float(inter / denom)


def _ray_direction_for_pixel(
    player_pose: Dict[str, float],
    image_shape: Tuple[int, int, int],
    *,
    pixel_x: float,
    pixel_y: float,
    basis_name: str,
) -> np.ndarray:
    basis = _get_basis_by_name(player_pose, basis_name)
    if basis is None:
        raise KeyError(f"Unknown basis: {basis_name}")
    forward, right, up = basis
    height, width = int(image_shape[0]), int(image_shape[1])
    aspect = float(width) / max(1.0, float(height))
    fov_y = math.radians(float(DEFAULT_FOV_DEG))
    fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect)
    ndc_x = (2.0 * ((float(pixel_x) + 0.5) / float(width))) - 1.0
    ndc_y = 1.0 - (2.0 * ((float(pixel_y) + 0.5) / float(height)))
    cam_dir = (
        np.asarray(forward, dtype=np.float32)
        + float(ndc_x * math.tan(fov_x / 2.0)) * np.asarray(right, dtype=np.float32)
        + float(ndc_y * math.tan(fov_y / 2.0)) * np.asarray(up, dtype=np.float32)
    )
    norm = float(np.linalg.norm(cam_dir))
    if norm <= 1e-8:
        raise RuntimeError("Degenerate ray direction")
    return cam_dir / norm


def _ray_box_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    world_center: Tuple[float, float, float],
) -> Optional[Tuple[float, List[float]]]:
    bounds_min = np.asarray([float(world_center[0]) - 0.5, float(world_center[1]) - 0.5, float(world_center[2]) - 0.5], dtype=np.float32)
    bounds_max = np.asarray([float(world_center[0]) + 0.5, float(world_center[1]) + 0.5, float(world_center[2]) + 0.5], dtype=np.float32)
    t_min = -float("inf")
    t_max = float("inf")
    for axis in range(3):
        o = float(origin[axis])
        d = float(direction[axis])
        b0 = float(bounds_min[axis])
        b1 = float(bounds_max[axis])
        if abs(d) < 1e-8:
            if o < b0 or o > b1:
                return None
            continue
        inv_d = 1.0 / d
        t0 = (b0 - o) * inv_d
        t1 = (b1 - o) * inv_d
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return None
    if t_max < 0.0:
        return None
    t_hit = t_min if t_min >= 0.0 else t_max
    hit_point = (origin + float(t_hit) * direction).tolist()
    return float(t_hit), [float(value) for value in hit_point]


def _candidate_score(
    *,
    pixel_x: float,
    pixel_y: float,
    polygon_vertices: List[Tuple[float, float]],
    candidate_center: Optional[Dict[str, Any]],
    candidate_bbox: Optional[Tuple[int, int, int, int]],
    ray_hit_t: Optional[float],
) -> float:
    if candidate_center is None:
        return float("-inf")
    dx = float(candidate_center["x"]) - float(pixel_x)
    dy = float(candidate_center["y"]) - float(pixel_y)
    center_penalty = math.sqrt(dx * dx + dy * dy)
    polygon_bbox = _polygon_bbox(polygon_vertices)
    iou = _bbox_iou(candidate_bbox, polygon_bbox)
    hit_bonus = 250.0 if ray_hit_t is not None else 0.0
    ray_penalty = float(ray_hit_t) if ray_hit_t is not None else 1000.0
    return hit_bonus + 300.0 * float(iou) - 6.0 * float(center_penalty) - 2.0 * float(ray_penalty)


def _target_type_names(task_key: str, explicit_target_type: str) -> Tuple[str, ...]:
    if str(explicit_target_type or "").strip():
        normalized = _normalize_name(explicit_target_type)
        return (normalized,)
    if task_key in AUTO_GOAL_BLOCK_TYPES:
        return tuple(str(name) for name in AUTO_GOAL_BLOCK_TYPES[task_key])
    return ("target",)


def _collect_target_voxel_candidates(
    *,
    task_key: str,
    player_pose: Dict[str, float],
    info: Dict[str, Any],
    raw_image: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    polygon_vertices: List[Tuple[float, float]],
    basis_name: str,
    explicit_target_type: str,
) -> List[Dict[str, Any]]:
    target_names = _target_type_names(task_key, explicit_target_type)
    origin = np.asarray(
        [
            float(player_pose["x"]),
            float(player_pose["y"]) + float(DEFAULT_EYE_HEIGHT),
            float(player_pose["z"]),
        ],
        dtype=np.float32,
    )
    ray_dir = _ray_direction_for_pixel(
        player_pose,
        raw_image.shape,
        pixel_x=float(pixel_x),
        pixel_y=float(pixel_y),
        basis_name=basis_name,
    )
    candidates: List[Dict[str, Any]] = []
    voxels = info.get("voxels") or []
    if not isinstance(voxels, list):
        return candidates

    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if not voxel_type or not any(name in voxel_type for name in target_names):
            continue
        raw_offset = (
            float(voxel.get("x", 0.0)),
            float(voxel.get("y", 0.0)),
            float(voxel.get("z", 0.0)),
        )
        world_center = _voxel_world_center_from_player_pose(player_pose, raw_offset)
        ray_hit = _ray_box_intersection(origin, ray_dir, world_center)
        ray_hit_t = None if ray_hit is None else float(ray_hit[0])
        ray_hit_point = None if ray_hit is None else ray_hit[1]
        projected = _project_visible_target_with_basis(
            player_pose=player_pose,
            info=info,
            target_center=world_center,
            image_shape=raw_image.shape,
            basis_name=basis_name,
            voxel_key=(int(round(raw_offset[0])), int(round(raw_offset[1])), int(round(raw_offset[2]))),
        )
        if projected is None:
            continue
        center_raw = projected.get("target_center_raw")
        bbox_raw = projected.get("bbox_raw")
        if isinstance(bbox_raw, list):
            bbox_raw = tuple(int(v) for v in bbox_raw)
        elif isinstance(bbox_raw, tuple):
            bbox_raw = tuple(int(v) for v in bbox_raw)
        else:
            bbox_raw = None
        score = _candidate_score(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            polygon_vertices=polygon_vertices,
            candidate_center=center_raw if isinstance(center_raw, dict) else None,
            candidate_bbox=bbox_raw,
            ray_hit_t=ray_hit_t,
        )
        candidates.append(
            {
                "voxel_type": voxel_type,
                "raw_offset": [round(float(raw_offset[0]), 4), round(float(raw_offset[1]), 4), round(float(raw_offset[2]), 4)],
                "world_center": [round(float(world_center[0]), 4), round(float(world_center[1]), 4), round(float(world_center[2]), 4)],
                "ray_hit_t": None if ray_hit_t is None else round(float(ray_hit_t), 6),
                "ray_hit_point": _sanitize_debug_value(ray_hit_point),
                "score": round(float(score), 6),
                "projected": _sanitize_debug_value(projected),
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item.get("score", float("-inf"))),
            float(item.get("ray_hit_t") if item.get("ray_hit_t") is not None else 1e9),
        )
    )
    return candidates


def _overlay_inference(
    image: np.ndarray,
    *,
    pixel_x: float,
    pixel_y: float,
    polygon_vertices: List[Tuple[float, float]],
    best_candidate: Optional[Dict[str, Any]],
) -> np.ndarray:
    overlay = image.copy()
    if polygon_vertices:
        polygon = np.asarray([[int(round(x)), int(round(y))] for x, y in polygon_vertices], dtype=np.int32)
        cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.circle(overlay, (int(round(pixel_x)), int(round(pixel_y))), 4, (0, 255, 255), thickness=-1)
    if best_candidate is not None:
        projected = best_candidate.get("projected") or {}
        bbox = projected.get("bbox_raw")
        center = projected.get("target_center_raw")
        if isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
        if isinstance(center, dict):
            cv2.drawMarker(
                overlay,
                (int(round(float(center["x"]))), int(round(float(center["y"])))),
                (255, 64, 64),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )
    return overlay


def main():
    args = parse_args()
    protocol = resolve_interaction_protocol(args.protocol)
    warmup_noop_steps = protocol.warmup_noop_steps if args.warmup_noop_steps is None else int(args.warmup_noop_steps)
    polygon_vertices = _parse_polygon(args.polygon)
    pixel_x = float(args.pixel_x)
    pixel_y = float(args.pixel_y)
    if polygon_vertices:
        pixel_x, pixel_y = _polygon_center(polygon_vertices)

    task_specs = resolve_interaction_task_specs([args.task], env_source=args.env_source)
    task_specs = apply_protocol_to_task_specs(task_specs, protocol_name=args.protocol)
    task_spec = task_specs[0]
    task_key = task_spec.task_key or args.task

    refresh_task_configs = bool(Path(args.task_group_path).is_dir())
    file_list = prepare_task_configs(args.task_group, path=args.task_group_path, refresh=refresh_task_configs)
    name_file_mapping = {name: file for name, file in file_list.items()}
    if task_spec.task_config_name not in name_file_mapping:
        raise KeyError(f"Task config {task_spec.task_config_name} not found in prepared task configs.")
    task_path = Path(name_file_mapping[task_spec.task_config_name])
    with task_path.open("r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f) or {}
    task_config.pop("reference_video", None)

    callbacks = build_callbacks(task_key, task_config)
    env = None
    obs = None
    info = None
    last_reset_exc = None
    for reset_attempt in range(1, max(1, int(args.reset_retries)) + 1):
        env = MinecraftSim(
            seed=int(args.base_seed),
            preferred_spawn_biome="plains",
            callbacks=callbacks,
            obs_size=(224, 224),
            action_type="env",
        )
        try:
            obs, info = env.reset()
            break
        except Exception as exc:
            last_reset_exc = exc
            try:
                env.close()
            except Exception:
                pass
            env = None
            continue
    if env is None or obs is None or info is None:
        raise last_reset_exc or RuntimeError("Failed to reset Minecraft env")

    try:
        for _ in range(int(max(0, warmup_noop_steps))):
            obs, _, _, _, info = env.step(env.noop_action())
        raw_image = np.asarray(info["pov"], dtype=np.uint8).copy()
        player_pose = _player_pose(info)
        candidates = _collect_target_voxel_candidates(
            task_key=task_key,
            player_pose=player_pose,
            info=info,
            raw_image=raw_image,
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            polygon_vertices=polygon_vertices,
            basis_name=str(args.projection_basis),
            explicit_target_type=str(args.target_type),
        )
        best_candidate = candidates[0] if candidates else None

        run_dir = Path(args.out_dir) / task_key / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw_image).save(run_dir / "reference_pov.png")
        overlay = _overlay_inference(
            raw_image,
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            polygon_vertices=polygon_vertices,
            best_candidate=best_candidate,
        )
        Image.fromarray(overlay).save(run_dir / "reference_inference_overlay.png")

        result = {
            "task_key": task_key,
            "task_config_path": str(task_path.resolve()),
            "seed": int(args.base_seed),
            "projection_basis": str(args.projection_basis),
            "pixel_x": float(pixel_x),
            "pixel_y": float(pixel_y),
            "polygon_vertices": _sanitize_debug_value(polygon_vertices),
            "player_pose": _sanitize_debug_value(player_pose),
            "configured_target_world_center": _sanitize_debug_value(task_config.get("target_world_center")),
            "best_candidate": _sanitize_debug_value(best_candidate),
            "top_candidates": _sanitize_debug_value(candidates[:10]),
            "reference_pov_path": str((run_dir / "reference_pov.png").resolve()),
            "reference_inference_overlay_path": str((run_dir / "reference_inference_overlay.png").resolve()),
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
