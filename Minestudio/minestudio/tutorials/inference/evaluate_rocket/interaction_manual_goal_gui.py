import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from minestudio.tutorials.inference.evaluate_rocket.interaction_make_manual_goal import (
    build_mask,
    default_sam_path,
    ensure_out_dir,
    load_predictor,
    overlay_bbox,
    overlay_mask,
    resolve_segment_type,
    write_goal_asset,
)


PREDICTOR_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for loading a goal image from disk, clicking SAM2 points, and saving a fixed goal spec."
    )
    parser.add_argument(
        "--sam-path",
        type=str,
        default="",
        help="Directory containing SAM2 checkpoints. Defaults to bundled Minestudio checkpoints.",
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
        help="Radius of circular fallback mask when SAM2 output is too small.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=300,
        help="If SAM2 mask area is below this threshold, union it with fallback circles.",
    )
    parser.add_argument(
        "--default-task",
        type=str,
        default="mine_coal",
        help="Prefill task key in the GUI.",
    )
    parser.add_argument(
        "--default-image",
        type=str,
        default="",
        help="Optional image path to preload in the GUI.",
    )
    parser.add_argument(
        "--default-out-dir",
        type=str,
        default="",
        help="Optional output directory to prefill in the GUI.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def empty_state() -> Dict:
    return {
        "image_path": "",
        "image": None,
        "positive_points": [],
        "negative_points": [],
        "mask": None,
        "raw_mask_area": 0,
        "fallback_area": 0,
        "used_fallback": False,
        "segment_type": "",
    }


def serialize_points(points: List[Tuple[float, float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def render_canvas(state: Dict, show_mask: bool = True) -> np.ndarray | None:
    image = state.get("image")
    if image is None:
        return None
    canvas = image.copy()
    if show_mask and state.get("mask") is not None:
        canvas = overlay_mask(canvas, state["mask"])
        canvas = overlay_bbox(canvas, state["mask"])
    pil_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_image)
    for x, y in state.get("positive_points", []):
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0), outline=(255, 255, 255))
    for x, y in state.get("negative_points", []):
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0), outline=(255, 255, 255))
    return np.asarray(pil_image)


def points_json(state: Dict) -> str:
    payload = {
        "positive_points": serialize_points(state.get("positive_points", [])),
        "negative_points": serialize_points(state.get("negative_points", [])),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def predictor_for(sam_path: Path, sam_choice: str):
    key = (str(sam_path), sam_choice)
    predictor = PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = load_predictor(sam_path, sam_choice)
        PREDICTOR_CACHE[key] = predictor
    return predictor


def load_image_file(image_path: str, task_key: str, state: Dict):
    image_path = str(image_path or "").strip()
    task_key = str(task_key or "").strip()
    if not image_path:
        raise gr.Error("image path is required")
    if not task_key:
        raise gr.Error("task key is required")
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise gr.Error(f"image not found: {path}")
    image = np.asarray(Image.open(path).convert("RGB"))
    segment_type = resolve_segment_type(task_key)
    new_state = empty_state()
    new_state["image_path"] = str(path)
    new_state["image"] = image
    new_state["segment_type"] = segment_type
    status = f"loaded {path.name} ({image.shape[1]}x{image.shape[0]}) task={task_key} segment_type={segment_type}"
    return render_canvas(new_state, show_mask=False), new_state, points_json(new_state), status


def add_point(
    mode: str,
    state: Dict,
    evt: gr.SelectData,
):
    image = state.get("image")
    if image is None:
        raise gr.Error("load an image first")
    x, y = float(evt.index[0]), float(evt.index[1])
    state["mask"] = None
    state["raw_mask_area"] = 0
    state["fallback_area"] = 0
    state["used_fallback"] = False
    if mode == "positive":
        state["positive_points"].append((x, y))
    else:
        state["negative_points"].append((x, y))
    status = f"added {mode} point at ({int(round(x))}, {int(round(y))})"
    return render_canvas(state, show_mask=False), state, points_json(state), status


def undo_point(state: Dict):
    if state.get("negative_points"):
        state["negative_points"].pop()
    elif state.get("positive_points"):
        state["positive_points"].pop()
    state["mask"] = None
    state["raw_mask_area"] = 0
    state["fallback_area"] = 0
    state["used_fallback"] = False
    return render_canvas(state, show_mask=False), state, points_json(state), "removed last point"


def clear_points(state: Dict):
    image = state.get("image")
    if image is None:
        return None, empty_state(), points_json(empty_state()), "cleared"
    image_path = state.get("image_path", "")
    segment_type = state.get("segment_type", "")
    new_state = empty_state()
    new_state["image"] = image
    new_state["image_path"] = image_path
    new_state["segment_type"] = segment_type
    return render_canvas(new_state, show_mask=False), new_state, points_json(new_state), "cleared points and mask"


def run_sam(task_key: str, state: Dict, sam_path_text: str, sam_choice: str, min_mask_area: int, fallback_radius: int):
    image = state.get("image")
    if image is None:
        raise gr.Error("load an image first")
    if not state.get("positive_points"):
        raise gr.Error("at least one positive point is required")
    task_key = str(task_key or "").strip()
    if not task_key:
        raise gr.Error("task key is required")

    sam_path = Path(sam_path_text).expanduser().resolve() if sam_path_text else default_sam_path()
    predictor = predictor_for(sam_path, sam_choice)
    mask, raw_area, fallback_area, used_fallback = build_mask(
        image=image,
        predictor=predictor,
        positive_points=state["positive_points"],
        negative_points=state["negative_points"],
        min_mask_area=int(min_mask_area),
        fallback_radius=int(fallback_radius),
    )
    state["mask"] = mask
    state["raw_mask_area"] = int(raw_area)
    state["fallback_area"] = int(fallback_area)
    state["used_fallback"] = bool(used_fallback)
    state["segment_type"] = resolve_segment_type(task_key)
    status = (
        f"mask ready raw_area={int(raw_area)} final_area={int(mask.sum())} "
        f"fallback={'yes' if used_fallback else 'no'}"
    )
    return render_canvas(state, show_mask=True), state, points_json(state), status


def save_goal(task_key: str, out_dir_text: str, cfg_coef: float, sam_path_text: str, sam_choice: str, state: Dict):
    image = state.get("image")
    mask = state.get("mask")
    if image is None:
        raise gr.Error("load an image first")
    if mask is None:
        raise gr.Error("generate a mask first")
    task_key = str(task_key or "").strip()
    if not task_key:
        raise gr.Error("task key is required")
    sam_path = Path(sam_path_text).expanduser().resolve() if sam_path_text else default_sam_path()

    if out_dir_text.strip():
        out_dir = Path(out_dir_text).expanduser().resolve()
    else:
        class Args:
            out_dir = ""
        out_dir = ensure_out_dir(Args(), task_key)

    result = write_goal_asset(
        image=image,
        mask=mask,
        out_dir=out_dir,
        task_key=task_key,
        segment_type=state.get("segment_type") or resolve_segment_type(task_key),
        positive_points=state.get("positive_points", []),
        negative_points=state.get("negative_points", []),
        sam_path=sam_path,
        sam_choice=sam_choice,
        raw_area=int(state.get("raw_mask_area", int(mask.sum()))),
        fallback_area=int(state.get("fallback_area", 0)),
        used_fallback=bool(state.get("used_fallback", False)),
        cfg_coef=float(cfg_coef),
    )
    status = f"saved goal spec to {result['goal_spec_path']}"
    return out_dir.as_posix(), status, json.dumps(result, ensure_ascii=False, indent=2)


def build_demo(args):
    initial_state = empty_state()
    default_sam = str(Path(args.sam_path).expanduser().resolve()) if args.sam_path else str(default_sam_path())
    with gr.Blocks(title="Manual Goal Mask GUI") as demo:
        gr.Markdown(
            "## Manual Goal Mask GUI\n"
            "Load a goal image from disk, click positive/negative points, run SAM2, and save a fixed goal spec."
        )
        state = gr.State(initial_state)

        with gr.Row():
            image_path = gr.Textbox(label="Goal Image Path", value=args.default_image)
            task_key = gr.Textbox(label="Task Key", value=args.default_task)
        with gr.Row():
            out_dir = gr.Textbox(label="Output Directory", value=args.default_out_dir)
            sam_path = gr.Textbox(label="SAM Checkpoint Directory", value=default_sam)
        with gr.Row():
            sam_choice = gr.Dropdown(
                label="SAM2 Backbone",
                choices=["large", "base", "small", "tiny"],
                value=args.sam_choice,
            )
            point_mode = gr.Radio(label="Click Mode", choices=["positive", "negative"], value="positive")
            min_mask_area = gr.Number(label="Min Mask Area", value=args.min_mask_area, precision=0)
            fallback_radius = gr.Number(label="Fallback Radius", value=args.fallback_radius, precision=0)
            cfg_coef = gr.Number(label="CFG Coef", value=args.cfg_coef)

        with gr.Row():
            load_button = gr.Button("Load Image", variant="primary")
            sam_button = gr.Button("Run SAM2")
            undo_button = gr.Button("Undo Point")
            clear_button = gr.Button("Clear Points")
            save_button = gr.Button("Save Goal Spec")

        canvas = gr.Image(label="Goal Image / Mask Preview", type="numpy", interactive=True)
        status = gr.Textbox(label="Status", interactive=False)
        point_dump = gr.Code(label="Points", language="json", interactive=False)
        save_dump = gr.Code(label="Saved Artifact Info", language="json", interactive=False)

        load_button.click(
            fn=load_image_file,
            inputs=[image_path, task_key, state],
            outputs=[canvas, state, point_dump, status],
        )
        canvas.select(
            fn=add_point,
            inputs=[point_mode, state],
            outputs=[canvas, state, point_dump, status],
        )
        undo_button.click(
            fn=undo_point,
            inputs=[state],
            outputs=[canvas, state, point_dump, status],
        )
        clear_button.click(
            fn=clear_points,
            inputs=[state],
            outputs=[canvas, state, point_dump, status],
        )
        sam_button.click(
            fn=run_sam,
            inputs=[task_key, state, sam_path, sam_choice, min_mask_area, fallback_radius],
            outputs=[canvas, state, point_dump, status],
        )
        save_button.click(
            fn=save_goal,
            inputs=[task_key, out_dir, cfg_coef, sam_path, sam_choice, state],
            outputs=[out_dir, status, save_dump],
        )

        if args.default_image:
            demo.load(
                fn=load_image_file,
                inputs=[image_path, task_key, state],
                outputs=[canvas, state, point_dump, status],
            )
    return demo


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share))


if __name__ == "__main__":
    main()
