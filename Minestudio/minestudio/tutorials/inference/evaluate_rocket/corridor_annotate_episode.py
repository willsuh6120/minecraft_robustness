#!/usr/bin/env python3
"""
Post-process a P-maze episode directory: overlay corridor-commit phase labels
on the recorded video, and write per-step debug CSV/JSON.

Usage:
    python corridor_annotate_episode.py <episode_dir> [options]
    python corridor_annotate_episode.py <collect_root>  --recurse   # batch mode

Produces per episode_dir:
    corridor_annotated.mp4   — video with lane/causal/approach overlays
    corridor_debug.csv       — per-step CSV table
    corridor_debug.json      — episode meta + per-step records

Color code:
    GREEN border  = causal segment (commit → approach)
    YELLOW border = approach zone
    lane bar      = orange (left) / blue (right) / gray (center)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

# ── Import corridor_signal from the same package ─────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from corridor_signal import compute_corridor_phases, load_episode_phases, load_arena_geometry  # noqa: E402

# ── Visual constants ──────────────────────────────────────────────────────────
LANE_COLOR = {
    "left":   (255, 180,  60),   # orange
    "right":  ( 60, 180, 255),   # sky blue
    "center": (190, 190, 190),   # gray
    "unknown":(100, 100, 100),
}
BORDER_CAUSAL   = ( 50, 220,  50)   # green
BORDER_APPROACH = (220, 220,  50)   # yellow
PANEL_ALPHA = 0.6
VIDEO_FPS = 20.0


# ── Frame annotation ─────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, record: dict) -> np.ndarray:
    """Overlay corridor phase info on a single RGB frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    lane      = record["lane_id"]
    causal    = record["in_causal_segment"]
    approach  = record["in_approach"]
    lat       = record["lateral_offset"]
    dist      = record["dist_to_target"]
    step      = record["step"]

    # ── Coloured border ───────────────────────────────────────────────────────
    if causal:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), BORDER_CAUSAL, 5)
    elif approach:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), BORDER_APPROACH, 3)

    # ── Info panel (bottom-left) ──────────────────────────────────────────────
    lane_col = LANE_COLOR.get(lane, (150, 150, 150))
    tag = "CAUSAL" if causal else ("APPROACH" if approach else "")
    lx = record.get("local_x", float("nan"))
    lz = record.get("local_z", float("nan"))
    import math
    if not math.isnan(lx) and not math.isnan(lz):
        pos_str = f"lx{lx:+5.2f} lz{lz:4.2f}"
    else:
        pos_str = f"lat{lat:+5.2f} d{dist:4.1f}"
    lines = [
        f"step {step:3d}  {lane:<6}  {tag}",
        pos_str,
    ]
    pw, ph = 200, len(lines) * 22 + 10
    px0, py0 = 4, h - ph - 4
    roi = canvas[py0:py0 + ph, px0:px0 + pw]
    bg = np.zeros_like(roi)
    canvas[py0:py0 + ph, px0:px0 + pw] = cv2.addWeighted(roi, 1 - PANEL_ALPHA, bg, PANEL_ALPHA, 0)
    cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), lane_col, 2)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (px0 + 6, py0 + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Lateral indicator bar (right edge) ───────────────────────────────────
    bx = w - 16
    bt, bb = 16, h - 16
    bh = bb - bt
    mid_y = bt + bh // 2
    norm = max(-1.0, min(1.0, lat / 6.0))
    dot_y = int(mid_y - norm * (bh // 2))
    cv2.line(canvas, (bx, bt), (bx, bb), (80, 80, 80), 2)
    cv2.line(canvas, (bx - 4, mid_y), (bx + 4, mid_y), (150, 150, 150), 1)
    cv2.circle(canvas, (bx, dot_y), 6, lane_col, -1)

    return canvas


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_frames_from_video(video_path: Path) -> List[np.ndarray]:
    if not _CV2_OK:
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _load_frames_from_fragment(fragment_path: Path) -> List[np.ndarray]:
    try:
        frag = torch.load(str(fragment_path), map_location="cpu", weights_only=False)
        imgs = frag.get("image")
        if imgs is None:
            return []
        frames = []
        for t in range(imgs.shape[0]):
            img = imgs[t]
            if img.ndim == 3 and img.shape[0] in (3, 4):   # CHW
                img = img.permute(1, 2, 0)
            frames.append(img.numpy().astype(np.uint8))
        return frames
    except Exception:
        return []


def _write_annotated_video(out_path: Path, frames: List[np.ndarray], records: list, fps: float):
    if not frames or not _CV2_OK:
        return False
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = min(len(frames), len(records))
    for t in range(n):
        annotated = _draw_overlay(frames[t], records[t])
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    writer.release()
    return True


# ── Main annotation function ─────────────────────────────────────────────────

def annotate_episode(
    episode_dir: Path,
    focus_weight: float = 4.0,
    fps: float = VIDEO_FPS,
    overwrite: bool = False,
) -> bool:
    """
    Annotate a single episode directory.  Returns True on success.
    """
    episode_dir = Path(episode_dir)
    traj_path  = episode_dir / "trajectory.jsonl"
    goal_path  = episode_dir / "goal_spec.json"

    out_video  = episode_dir / "corridor_annotated.mp4"
    out_csv    = episode_dir / "corridor_debug.csv"
    out_json   = episode_dir / "corridor_debug.json"

    if out_json.exists() and not overwrite:
        print(f"[corridor-annotate] skip (already done): {episode_dir.name}")
        return True

    if not traj_path.exists() or not goal_path.exists():
        return False

    # ── Load positions and target ─────────────────────────────────────────────
    goal = json.loads(goal_path.read_text(encoding="utf-8"))
    traw = goal.get("target_world_center")
    if not (isinstance(traw, (list, tuple)) and len(traw) >= 3):
        return False
    target = (float(traw[0]), float(traw[2]))

    positions = []
    with traj_path.open(encoding="utf-8") as fh:
        for line in fh:
            pos = json.loads(line).get("player_pos") or {}
            positions.append((float(pos.get("x", target[0])), float(pos.get("z", target[1]))))

    if len(positions) < 2:
        return False

    # ── Compute phases ────────────────────────────────────────────────────────
    geom = load_arena_geometry(episode_dir, target)
    phases = compute_corridor_phases(positions, target, geom=geom)
    records = phases.to_per_step_records()

    # Add loss_weight column to records
    for rec in records:
        rec["loss_weight"] = float(focus_weight) if rec["in_causal_segment"] else 1.0

    meta = phases.summary()
    meta["target"] = list(target)
    meta["start_pos"] = list(positions[0])
    meta["focus_weight"] = focus_weight

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_json.write_text(
        json.dumps({"meta": meta, "steps": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    geom_tag = "geom" if phases.geometry_loaded else "heuristic"
    print(
        f"[corridor-annotate] {episode_dir.name}  "
        f"win={phases.winning_corridor}  "
        f"commit={phases.commit_step}→{phases.approach_start}  "
        f"fallback={phases.fallback_used}  [{geom_tag}]"
    )

    # ── Load video frames ─────────────────────────────────────────────────────
    frames: List[np.ndarray] = []
    if _CV2_OK:
        # prefer existing annotated video, then raw, then fragment images
        candidates = (
            sorted(episode_dir.glob("*_annotated.mp4"))
            + sorted(episode_dir.glob("*.mp4"))
        )
        for cand in candidates:
            if cand.name == out_video.name:
                continue
            frames = _load_frames_from_video(cand)
            if frames:
                break
        if not frames:
            frag = episode_dir / "episode_fragment.pt"
            if frag.exists():
                frames = _load_frames_from_fragment(frag)

    if frames:
        ok = _write_annotated_video(out_video, frames, records, fps)
        if ok:
            print(f"[corridor-annotate] wrote video: {out_video}")
    else:
        print(f"[corridor-annotate] no video frames found; wrote CSV/JSON only")

    return True


def annotate_tree(root: Path, focus_weight: float = 4.0, fps: float = VIDEO_FPS, overwrite: bool = False):
    """Recursively annotate all episode directories under root."""
    episode_dirs = sorted(
        d for d in root.rglob("trajectory.jsonl") if (d.parent / "goal_spec.json").exists()
    )
    total = len(episode_dirs)
    done = 0
    for traj in episode_dirs:
        ep_dir = traj.parent
        if annotate_episode(ep_dir, focus_weight=focus_weight, fps=fps, overwrite=overwrite):
            done += 1
    print(f"[corridor-annotate] annotated {done}/{total} episodes under {root}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add corridor-commit overlays to P-maze episode videos."
    )
    parser.add_argument("path", type=str,
                        help="Episode directory (single) or tree root (with --recurse)")
    parser.add_argument("--recurse", action="store_true",
                        help="Annotate all episodes found recursively under path")
    parser.add_argument("--focus-weight", type=float, default=4.0,
                        help="Loss weight to show for causal segment steps (default 4.0)")
    parser.add_argument("--fps", type=float, default=VIDEO_FPS,
                        help="Output video FPS (default 20.0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-annotate even if corridor_debug.json already exists")
    args = parser.parse_args()

    p = Path(args.path)
    if args.recurse:
        annotate_tree(p, focus_weight=args.focus_weight, fps=args.fps, overwrite=args.overwrite)
    else:
        ok = annotate_episode(p, focus_weight=args.focus_weight, fps=args.fps, overwrite=args.overwrite)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
