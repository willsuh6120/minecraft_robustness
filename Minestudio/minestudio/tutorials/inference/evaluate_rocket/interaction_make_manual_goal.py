import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2_camera_predictor

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    resolve_interaction_task_specs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a fixed cross-view goal asset from a still image using point-prompted SAM2."
    )
    parser.add_argument("--image", type=str, required=True, help="Input goal image path.")
    parser.add_argument("--task", type=str, required=True, help="Interaction task key, e.g. mine_coal.")
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        action="append",
        required=True,
        metavar=("X", "Y"),
        help="Positive point prompt in image pixel coordinates. Repeatable.",
    )
    parser.add_argument(
        "--negative-point",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="Negative point prompt in image pixel coordinates. Repeatable.",
    )
    parser.add_argument(
        "--sam-path",
        type=str,
        default="",
        help="Directory containing SAM2 checkpoints. Defaults to the bundled Minestudio checkpoints.",
    )
    parser.add_argument(
        "--sam-choice",
        type=str,
        default="base",
        choices=["large", "base", "small", "tiny"],
        help="SAM2 backbone checkpoint to use.",
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=1.5,
        help="Stored in goal_spec for rollout compatibility.",
    )
    parser.add_argument(
        "--fallback-radius",
        type=int,
        default=18,
        help="Radius of circular fallback mask when SAM2 mask is too small.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=300,
        help="If the SAM2 mask area is below this threshold, union it with fallback circles.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Defaults to outputs/evaluate_rocket/manual_goal_specs/<task>/<timestamp>.",
    )
    return parser.parse_args()


def default_sam_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "minestudio" / "utils" / "realtime_sam" / "checkpoints"


def resolve_segment_type(task_key: str) -> str:
    task_spec = resolve_interaction_task_specs([task_key], env_source="rocket2_official")[0]
    if not task_spec.subtasks:
        raise ValueError(f"Task {task_key} has no subtasks.")
    return str(task_spec.subtasks[0].interaction_type)


def load_predictor(sam_path: Path, sam_choice: str):
    ckpt_mapping = {
        "large": [sam_path / "sam2_hiera_large.pt", "sam2_hiera_l.yaml"],
        "base": [sam_path / "sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"],
        "small": [sam_path / "sam2_hiera_small.pt", "sam2_hiera_s.yaml"],
        "tiny": [sam_path / "sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"],
    }
    sam_ckpt, model_cfg = ckpt_mapping[sam_choice]
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_ckpt}")
    return build_sam2_camera_predictor(model_cfg, str(sam_ckpt))


def build_mask(
    image: np.ndarray,
    predictor,
    positive_points: List[Tuple[float, float]],
    negative_points: List[Tuple[float, float]],
    min_mask_area: int,
    fallback_radius: int,
):
    predictor.load_first_frame(image)
    all_points = np.asarray(positive_points + negative_points, dtype=np.float32)
    all_labels = np.asarray([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
    _, _, out_mask_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=0,
        points=all_points,
        labels=all_labels,
    )
    raw_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
    raw_area = int(raw_mask.sum())
    used_fallback = False
    fallback_area = 0
    final_mask = raw_mask.copy()
    if raw_area < int(min_mask_area) and positive_points:
        fallback_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        for x, y in positive_points:
            cv2.circle(
                fallback_mask,
                (int(round(x)), int(round(y))),
                int(fallback_radius),
                1,
                -1,
            )
        fallback_area = int(fallback_mask.sum())
        final_mask = np.logical_or(raw_mask > 0, fallback_mask > 0).astype(np.uint8)
        used_fallback = True
    return final_mask, raw_area, fallback_area, used_fallback


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        0.65 * overlay[mask_bool].astype(np.float32)
        + 0.35 * np.asarray([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def overlay_bbox(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return overlay
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return overlay


def write_goal_asset(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    task_key: str,
    segment_type: str,
    positive_points: List[Tuple[float, float]],
    negative_points: List[Tuple[float, float]],
    sam_path: Path,
    sam_choice: str,
    raw_area: int,
    fallback_area: int,
    used_fallback: bool,
    cfg_coef: float,
):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_image_path = out_dir / "goal_image.png"
    goal_mask_path = out_dir / "goal_mask.png"
    goal_mask_overlay_path = out_dir / "goal_mask_overlay.png"
    goal_bbox_overlay_path = out_dir / "goal_bbox_overlay.png"
    metadata_path = out_dir / "metadata.json"
    goal_spec_path = out_dir / "goal_spec.json"

    Image.fromarray(image).save(goal_image_path)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(goal_mask_path)
    Image.fromarray(overlay_mask(image, mask)).save(goal_mask_overlay_path)
    Image.fromarray(overlay_bbox(image, mask)).save(goal_bbox_overlay_path)

    metadata = {
        "task_key": task_key,
        "segment_type": segment_type,
        "goal_metadata": {
            "goal_mode": "manual_sam_point",
            "task_key": task_key,
            "segment_type": segment_type,
            "positive_points": [[float(x), float(y)] for x, y in positive_points],
            "negative_points": [[float(x), float(y)] for x, y in negative_points],
            "sam_path": str(sam_path),
            "sam_choice": sam_choice,
            "raw_mask_area": int(raw_area),
            "final_mask_area": int(mask.sum()),
            "fallback_area": int(fallback_area),
            "used_fallback": bool(used_fallback),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    goal_spec = {
        "goal_image_path": str(goal_image_path),
        "goal_mask_path": str(goal_mask_path),
        "goal_mask_overlay_path": str(goal_mask_overlay_path),
        "goal_bbox_overlay_path": str(goal_bbox_overlay_path),
        "segment_type": segment_type,
        "task_key": task_key,
        "goal_mode": "manual_sam_point",
        "cfg_coef": float(cfg_coef),
        "obs_size": [int(image.shape[1]), int(image.shape[0])],
    }
    goal_spec.update(metadata["goal_metadata"])
    goal_spec_path.write_text(json.dumps(goal_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "goal_image_path": str(goal_image_path),
        "goal_mask_path": str(goal_mask_path),
        "goal_mask_overlay_path": str(goal_mask_overlay_path),
        "goal_bbox_overlay_path": str(goal_bbox_overlay_path),
        "metadata_path": str(metadata_path),
        "goal_spec_path": str(goal_spec_path),
        "raw_mask_area": int(raw_area),
        "final_mask_area": int(mask.sum()),
        "used_fallback": bool(used_fallback),
        "segment_type": segment_type,
        "task_key": task_key,
    }


def ensure_out_dir(args, task_key: str) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "outputs"
        / "evaluate_rocket"
        / "manual_goal_specs"
        / task_key
        / time.strftime("%Y%m%d_%H%M%S")
    ).resolve()


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    sam_path = Path(args.sam_path).expanduser().resolve() if args.sam_path else default_sam_path()
    out_dir = ensure_out_dir(args, args.task)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    positive_points = [(float(x), float(y)) for x, y in args.point]
    negative_points = [(float(x), float(y)) for x, y in args.negative_point]
    predictor = load_predictor(sam_path, args.sam_choice)
    try:
        mask, raw_area, fallback_area, used_fallback = build_mask(
            image=image,
            predictor=predictor,
            positive_points=positive_points,
            negative_points=negative_points,
            min_mask_area=args.min_mask_area,
            fallback_radius=args.fallback_radius,
        )
    finally:
        del predictor

    segment_type = resolve_segment_type(args.task)
    result = write_goal_asset(
        image=image,
        mask=mask,
        out_dir=out_dir,
        task_key=args.task,
        segment_type=segment_type,
        positive_points=positive_points,
        negative_points=negative_points,
        sam_path=sam_path,
        sam_choice=args.sam_choice,
        raw_area=raw_area,
        fallback_area=fallback_area,
        used_fallback=used_fallback,
        cfg_coef=args.cfg_coef,
    )

    print(
        json.dumps(
            {"out_dir": str(out_dir), **result},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
