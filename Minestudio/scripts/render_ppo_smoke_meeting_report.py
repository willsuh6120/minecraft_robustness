#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPT-ready report assets for PPO smoke runs.")
    parser.add_argument("--pilot-root", required=True, help="Pilot run root, e.g. .../pilot/20260421_020345")
    parser.add_argument("--out-dir", default="", help="Optional output directory; defaults to <pilot-root>/meeting_report")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


@dataclass
class PhasePoint:
    phase: str
    kind: str
    iteration: int
    count: int
    successes: float
    success_rate: float
    reward_mean: float
    mean_steps: float


@dataclass
class UpdatePoint:
    iteration: int
    fragments: int
    advantage_mean: float
    advantage_std: float
    mean_total_loss: float
    mean_policy_loss: float
    mean_value_loss: float
    mean_kl_divergence: float
    mean_approx_kl: float
    mean_clip_fraction: float


def detect_config(pilot_root: Path) -> Dict[str, object]:
    run_metadata_paths = sorted((pilot_root / "baseline").glob("worker_*/*/run_metadata.json"))
    if not run_metadata_paths:
        return {}
    meta = load_json(run_metadata_paths[0])
    config = {
        "task_group_path": meta.get("task_group_path", ""),
        "episodes_per_task": meta.get("episodes_per_task", 0),
        "warmup_noop_steps": meta.get("warmup_noop_steps", 0),
        "step_budget_override": meta.get("step_budget_override", 0),
        "model_path": meta.get("model_path", ""),
        "protocol_name": (meta.get("protocol") or {}).get("name", ""),
    }
    return config


def collect_phase_points(pilot_root: Path) -> List[PhasePoint]:
    points: List[PhasePoint] = []

    baseline_files = sorted((pilot_root / "baseline").glob("worker_*/*/summary.json"))
    if baseline_files:
        rows = [load_json(path)[0] for path in baseline_files]
        count = len(rows)
        successes = sum(float(row.get("auto_success_rate", 0.0) or 0.0) for row in rows)
        reward_mean = sum(float(row.get("mean_reward", 0.0) or 0.0) for row in rows) / float(count)
        mean_steps = sum(float(row.get("mean_steps", 0.0) or 0.0) for row in rows) / float(count)
        points.append(
            PhasePoint(
                phase="baseline",
                kind="baseline",
                iteration=0,
                count=count,
                successes=successes,
                success_rate=successes / float(count),
                reward_mean=reward_mean,
                mean_steps=mean_steps,
            )
        )

    for iter_dir in sorted((pilot_root / "iterations").glob("iter_*")):
        try:
            iteration = int(iter_dir.name.split("_")[1])
        except Exception:
            continue
        for kind in ("collect", "eval"):
            summary_files = sorted((iter_dir / kind).glob("worker_*/*/summary.json"))
            if not summary_files:
                continue
            rows = [load_json(path)[0] for path in summary_files]
            count = len(rows)
            successes = sum(float(row.get("auto_success_rate", 0.0) or 0.0) for row in rows)
            reward_mean = sum(float(row.get("mean_reward", 0.0) or 0.0) for row in rows) / float(count)
            mean_steps = sum(float(row.get("mean_steps", 0.0) or 0.0) for row in rows) / float(count)
            points.append(
                PhasePoint(
                    phase=f"iter_{iteration:03d}_{kind}",
                    kind=kind,
                    iteration=iteration,
                    count=count,
                    successes=successes,
                    success_rate=successes / float(count),
                    reward_mean=reward_mean,
                    mean_steps=mean_steps,
                )
            )
    return points


def collect_update_points(pilot_root: Path) -> List[UpdatePoint]:
    points: List[UpdatePoint] = []
    for iter_dir in sorted((pilot_root / "iterations").glob("iter_*")):
        try:
            iteration = int(iter_dir.name.split("_")[1])
        except Exception:
            continue
        update_dirs = sorted((iter_dir / "update").glob("*"))
        update_dirs = [path for path in update_dirs if path.is_dir()]
        if not update_dirs:
            continue
        latest = update_dirs[-1]
        meta_path = latest / "train_metadata.json"
        hist_path = latest / "train_history.json"
        if not meta_path.exists() or not hist_path.exists():
            continue
        meta = load_json(meta_path)
        history = load_json(hist_path)
        if not history:
            continue
        last = history[-1]
        points.append(
            UpdatePoint(
                iteration=iteration,
                fragments=int(meta.get("fragments", 0) or 0),
                advantage_mean=float(meta.get("advantage_mean", 0.0) or 0.0),
                advantage_std=float(meta.get("advantage_std", 0.0) or 0.0),
                mean_total_loss=float(last.get("mean_total_loss", 0.0) or 0.0),
                mean_policy_loss=float(last.get("mean_policy_loss", 0.0) or 0.0),
                mean_value_loss=float(last.get("mean_value_loss", 0.0) or 0.0),
                mean_kl_divergence=float(last.get("mean_kl_divergence", 0.0) or 0.0),
                mean_approx_kl=float(last.get("mean_approx_kl", 0.0) or 0.0),
                mean_clip_fraction=float(last.get("mean_clip_fraction", 0.0) or 0.0),
            )
        )
    return points


def write_csv(points: Sequence[PhasePoint], updates: Sequence[UpdatePoint], out_dir: Path) -> None:
    with (out_dir / "phase_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "kind", "iteration", "count", "successes", "success_rate", "reward_mean", "mean_steps"])
        for point in points:
            writer.writerow(
                [
                    point.phase,
                    point.kind,
                    point.iteration,
                    point.count,
                    point.successes,
                    point.success_rate,
                    point.reward_mean,
                    point.mean_steps,
                ]
            )
    with (out_dir / "update_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "fragments",
                "advantage_mean",
                "advantage_std",
                "mean_total_loss",
                "mean_policy_loss",
                "mean_value_loss",
                "mean_kl_divergence",
                "mean_approx_kl",
                "mean_clip_fraction",
            ]
        )
        for point in updates:
            writer.writerow(
                [
                    point.iteration,
                    point.fragments,
                    point.advantage_mean,
                    point.advantage_std,
                    point.mean_total_loss,
                    point.mean_policy_loss,
                    point.mean_value_loss,
                    point.mean_kl_divergence,
                    point.mean_approx_kl,
                    point.mean_clip_fraction,
                ]
            )


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    max_width: int,
    line_spacing: int = 6,
) -> int:
    x, y = xy
    words = text.split()
    line = ""
    total_height = 0
    for word in words:
        candidate = word if not line else f"{line} {word}"
        w, h = text_size(draw, candidate, font)
        if line and w > max_width:
            draw.text((x, y + total_height), line, font=font, fill=fill)
            total_height += h + line_spacing
            line = word
        else:
            line = candidate
    if line:
        w, h = text_size(draw, line, font)
        draw.text((x, y + total_height), line, font=font, fill=fill)
        total_height += h
    return total_height


def draw_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str, title_font, body_font) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=(250, 250, 252), outline=(220, 224, 230), width=2)
    draw.text((x0 + 24, y0 + 18), title, font=title_font, fill=(22, 26, 31))
    return (x0 + 24, y0 + 60, x1 - 24, y1 - 24)


def map_point(value: float, lo: float, hi: float, px0: int, px1: int) -> int:
    if hi <= lo:
        return px1
    ratio = (value - lo) / float(hi - lo)
    ratio = max(0.0, min(1.0, ratio))
    return int(round(px0 + ratio * (px1 - px0)))


def draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    y_ticks: Sequence[Tuple[float, str]],
    x_labels: Sequence[Tuple[int, str]],
    font: ImageFont.ImageFont,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.line((x0, y1, x1, y1), fill=(120, 128, 140), width=2)
    draw.line((x0, y0, x0, y1), fill=(120, 128, 140), width=2)
    for y_val, label in y_ticks:
        yy = map_point(y_val, 0.0, 1.0, y1, y0)
        draw.line((x0, yy, x1, yy), fill=(230, 233, 238), width=1)
        draw.text((x0 - 40, yy - 8), label, font=font, fill=(90, 97, 108))
    for xx, label in x_labels:
        draw.line((xx, y1, xx, y1 + 6), fill=(120, 128, 140), width=2)
        draw.text((xx - 18, y1 + 12), label, font=font, fill=(90, 97, 108))
    return rect


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    xs: Sequence[int],
    ys: Sequence[float],
    color: Tuple[int, int, int],
    y_lo: float,
    y_hi: float,
    point_radius: int = 5,
    width: int = 4,
) -> None:
    if not xs or not ys:
        return
    x0, y0, x1, y1 = rect
    coords = []
    for x_val, y_val in zip(xs, ys):
        xx = map_point(float(x_val), min(xs), max(xs) if len(set(xs)) > 1 else min(xs) + 1, x0, x1)
        yy = map_point(float(y_val), y_lo, y_hi, y1, y0)
        coords.append((xx, yy))
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=width, joint="curve")
    for xx, yy in coords:
        draw.ellipse((xx - point_radius, yy - point_radius, xx + point_radius, yy + point_radius), fill=color, outline=(255, 255, 255), width=2)


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    xs: Sequence[int],
    ys: Sequence[float],
    color: Tuple[int, int, int],
    y_lo: float,
    y_hi: float,
) -> None:
    if not xs or not ys:
        return
    x0, y0, x1, y1 = rect
    span = max(1, len(xs))
    usable = x1 - x0
    bar_w = max(10, usable // (span * 2))
    for idx, (x_val, y_val) in enumerate(zip(xs, ys)):
        xx = map_point(float(x_val), min(xs), max(xs) if len(set(xs)) > 1 else min(xs) + 1, x0, x1)
        yy = map_point(float(y_val), y_lo, y_hi, y1, y0)
        draw.rounded_rectangle((xx - bar_w // 2, yy, xx + bar_w // 2, y1), radius=6, fill=color)


def phase_map(points: Sequence[PhasePoint]) -> Dict[str, List[PhasePoint]]:
    mapping: Dict[str, List[PhasePoint]] = {}
    for point in points:
        mapping.setdefault(point.kind, []).append(point)
    for key in mapping:
        mapping[key] = sorted(mapping[key], key=lambda item: item.iteration)
    return mapping


def render_overview_slide(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint], updates: Sequence[UpdatePoint], config: Dict[str, object]) -> Path:
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), (243, 246, 250))
    draw = ImageDraw.Draw(img)
    title_font = find_font(46, bold=True)
    panel_title_font = find_font(28, bold=True)
    body_font = find_font(22, bold=False)
    small_font = find_font(18, bold=False)

    draw.text((60, 36), "PPO Smoke Test: Straight + O2, Fixed Single World", font=title_font, fill=(18, 24, 32))
    draw.text((62, 92), f"Run: {pilot_root}", font=small_font, fill=(92, 100, 112))

    left = draw_panel(draw, (50, 140, 760, 1030), "Key Takeaways", panel_title_font, body_font)
    collect_eval = phase_map(points)
    eval_points = collect_eval.get("eval", [])
    baseline_points = collect_eval.get("baseline", [])
    collect_points = collect_eval.get("collect", [])
    peak_eval = max((p.success_rate for p in eval_points), default=0.0)
    peak_iter = max(eval_points, key=lambda p: p.success_rate).iteration if eval_points else 0
    latest_eval = eval_points[-1].success_rate if eval_points else 0.0
    latest_collect = collect_points[-1].success_rate if collect_points else 0.0
    bullets = [
        f"Pipeline is live: baseline success was {baseline_points[0].success_rate:.0%} and eval peaked at {peak_eval:.0%} on iter {peak_iter:03d}.",
        f"But training is unstable: latest completed eval is {latest_eval:.0%}, and latest completed collect is {latest_collect:.0%}.",
        "This is not an eval-set drift issue. The world is static and eval seeds are fixed inside the same smoke run.",
        "The update is driven by only 10 episode fragments per iteration, with very sparse reward (+1 only on success).",
        "Interpretation: PPO plumbing works, but the current update regime is too noisy to support a stable upward curve.",
    ]
    tx0, ty0, tx1, ty1 = left
    y_cursor = ty0
    for bullet in bullets:
        draw.text((tx0, y_cursor), "•", font=body_font, fill=(37, 99, 235))
        used_h = draw_wrapped_text(draw, (tx0 + 24, y_cursor), bullet, body_font, (34, 39, 46), tx1 - tx0 - 30)
        y_cursor += used_h + 22

    cfg_lines = [
        f"Config: stop_on_success=True, step_budget_override={config.get('step_budget_override', 'n/a')}",
        f"Protocol: {config.get('protocol_name', 'n/a')}, warmup_noop_steps={config.get('warmup_noop_steps', 'n/a')}",
        "World mode: collect/eval/final_eval all static, single baked world",
        "Workers: 10 parallel collect, 10 parallel eval",
        "Episodes: 10 collect and 10 eval per iteration",
    ]
    y_cursor += 16
    draw.text((tx0, y_cursor), "Experiment Setup", font=panel_title_font, fill=(22, 26, 31))
    y_cursor += 44
    for line in cfg_lines:
        used_h = draw_wrapped_text(draw, (tx0, y_cursor), line, small_font, (70, 77, 88), tx1 - tx0)
        y_cursor += used_h + 12

    top_chart = draw_panel(draw, (790, 140, 1860, 590), "Success Rate Across Iterations", panel_title_font, body_font)
    chart_rect = (top_chart[0] + 28, top_chart[1] + 18, top_chart[2] - 30, top_chart[3] - 70)
    x_labels = [(map_point(i, 0, max(1, max([p.iteration for p in eval_points], default=1)), chart_rect[0], chart_rect[2]), str(i)) for i in range(0, max([p.iteration for p in eval_points], default=0) + 1)]
    draw_axes(draw, chart_rect, [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")], x_labels, small_font)
    if baseline_points:
        draw_line_chart(draw, chart_rect, [p.iteration for p in baseline_points], [p.success_rate for p in baseline_points], (120, 124, 130), 0.0, 1.0, point_radius=6)
    if collect_points:
        draw_line_chart(draw, chart_rect, [p.iteration for p in collect_points], [p.success_rate for p in collect_points], (37, 99, 235), 0.0, 1.0)
    if eval_points:
        draw_line_chart(draw, chart_rect, [p.iteration for p in eval_points], [p.success_rate for p in eval_points], (220, 38, 38), 0.0, 1.0)
    legend_y = chart_rect[1] - 4
    legend_items = [("baseline", (120, 124, 130)), ("collect", (37, 99, 235)), ("eval", (220, 38, 38))]
    lx = chart_rect[0]
    for label, color in legend_items:
        draw.rounded_rectangle((lx, legend_y, lx + 24, legend_y + 12), radius=4, fill=color)
        draw.text((lx + 34, legend_y - 6), label, font=small_font, fill=(60, 66, 74))
        lx += 140

    bottom_chart = draw_panel(draw, (790, 620, 1860, 1030), "Update Diagnostics", panel_title_font, body_font)
    diag_left = (bottom_chart[0] + 28, bottom_chart[1] + 24, bottom_chart[0] + 500, bottom_chart[3] - 70)
    diag_right = (bottom_chart[0] + 560, bottom_chart[1] + 24, bottom_chart[2] - 30, bottom_chart[3] - 70)
    x_labels_upd = [(map_point(i, 1, max(1, max([u.iteration for u in updates], default=1)), diag_left[0], diag_left[2]), str(i)) for i in range(1, max([u.iteration for u in updates], default=0) + 1)]
    draw.text((diag_left[0], bottom_chart[1]), "KL divergence", font=small_font, fill=(55, 65, 81))
    draw_axes(draw, diag_left, [(0.0, "0"), (0.1, "0.1"), (0.2, "0.2"), (0.3, "0.3")], x_labels_upd, small_font)
    draw_line_chart(draw, diag_left, [u.iteration for u in updates], [u.mean_kl_divergence for u in updates], (124, 58, 237), 0.0, max(0.3, max((u.mean_kl_divergence for u in updates), default=0.3)))
    draw.text((diag_right[0], bottom_chart[1]), "Clip fraction", font=small_font, fill=(55, 65, 81))
    draw_axes(draw, diag_right, [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75")], x_labels_upd, small_font)
    draw_bar_chart(draw, diag_right, [u.iteration for u in updates], [u.mean_clip_fraction for u in updates], (245, 158, 11), 0.0, max(0.8, max((u.mean_clip_fraction for u in updates), default=0.8)))

    out_path = out_dir / "ppo_smoke_meeting_overview.png"
    img.save(out_path)
    return out_path


def render_diag_slide(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint], updates: Sequence[UpdatePoint]) -> Path:
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    title_font = find_font(42, bold=True)
    panel_title_font = find_font(26, bold=True)
    body_font = find_font(20)
    small_font = find_font(18)

    draw.text((60, 36), "PPO Smoke Diagnostics", font=title_font, fill=(18, 24, 32))
    draw.text((62, 88), f"Run: {pilot_root}", font=small_font, fill=(92, 100, 112))

    left = draw_panel(draw, (50, 140, 930, 1030), "Per-Iteration Table", panel_title_font, body_font)
    lx0, ly0, lx1, ly1 = left
    headers = ["iter", "collect", "eval", "fragments", "KL", "clip_frac"]
    col_x = [lx0, lx0 + 90, lx0 + 240, lx0 + 380, lx0 + 550, lx0 + 700]
    for cx, header in zip(col_x, headers):
        draw.text((cx, ly0), header, font=small_font, fill=(55, 65, 81))
    y = ly0 + 38
    point_map = {(p.iteration, p.kind): p for p in points if p.kind in {"collect", "eval"}}
    update_map = {u.iteration: u for u in updates}
    max_iter = max([u.iteration for u in updates], default=0)
    for iteration in range(1, max_iter + 1):
        collect = point_map.get((iteration, "collect"))
        evalp = point_map.get((iteration, "eval"))
        upd = update_map.get(iteration)
        if not upd:
            continue
        row_fill = (255, 255, 255) if iteration % 2 else (243, 245, 248)
        draw.rounded_rectangle((lx0 - 8, y - 6, lx1 - 8, y + 28), radius=8, fill=row_fill)
        values = [
            f"{iteration:03d}",
            f"{collect.success_rate:.0%}" if collect else "-",
            f"{evalp.success_rate:.0%}" if evalp else "-",
            str(upd.fragments),
            f"{upd.mean_kl_divergence:.3f}",
            f"{upd.mean_clip_fraction:.3f}",
        ]
        for cx, value in zip(col_x, values):
            draw.text((cx, y), value, font=body_font, fill=(33, 37, 41))
        y += 42

    right = draw_panel(draw, (970, 140, 1860, 1030), "Interpretation", panel_title_font, body_font)
    rx0, ry0, rx1, ry1 = right
    notes = [
        "Observed behavior is consistent with unstable on-policy fine-tuning, not a dead training loop.",
        "Evidence that PPO is active: success improves from 10% baseline to 70% at iter_002 eval on the same fixed world.",
        "Evidence that it is unstable: the curve collapses to 0% by iter_005 and iter_006 eval, still on the same fixed world and fixed eval seeds.",
        "Primary causes in the current setup: 10 fragments only, extremely sparse reward, and one optimizer step per whole episode fragment.",
        "This run supports the statement: 'the PPO update path is wired correctly, but its current regime is too noisy for reliable monotonic improvement.'",
        "Immediate next experiment: keep the same smoke world, increase collect size, and skip update when there are zero successful trajectories.",
    ]
    yy = ry0
    for note in notes:
        draw.text((rx0, yy), "•", font=body_font, fill=(2, 132, 199))
        used_h = draw_wrapped_text(draw, (rx0 + 24, yy), note, body_font, (34, 39, 46), rx1 - rx0 - 24)
        yy += used_h + 20

    out_path = out_dir / "ppo_smoke_meeting_diagnostics.png"
    img.save(out_path)
    return out_path


def render_graph_canvas(title: str, subtitle: str) -> Tuple[Image.Image, ImageDraw.ImageDraw, Tuple[int, int, int, int], ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    title_font = find_font(42, bold=True)
    subtitle_font = find_font(20, bold=False)
    axis_font = find_font(18, bold=False)
    draw.text((56, 34), title, font=title_font, fill=(18, 24, 32))
    draw.text((58, 88), subtitle, font=subtitle_font, fill=(92, 100, 112))
    plot_box = (92, 150, 1510, 790)
    draw.rounded_rectangle((46, 126, 1554, 844), radius=28, fill=(252, 252, 253), outline=(224, 228, 234), width=2)
    return img, draw, plot_box, title_font, subtitle_font, axis_font


def render_success_rate_all_graph(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint]) -> Path:
    img, draw, plot_box, _, _, axis_font = render_graph_canvas(
        "PPO Smoke: Success Rate Trend",
        "Baseline + collect + eval, fixed single world, static eval seeds",
    )
    mapping = phase_map(points)
    baseline = mapping.get("baseline", [])
    collect = mapping.get("collect", [])
    eval_points = mapping.get("eval", [])
    max_iter = max([p.iteration for p in points], default=1)
    x_labels = [(map_point(i, 0, max_iter, plot_box[0], plot_box[2]), str(i)) for i in range(0, max_iter + 1)]
    draw_axes(draw, plot_box, [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")], x_labels, axis_font)
    if baseline:
        draw_line_chart(draw, plot_box, [p.iteration for p in baseline], [p.success_rate for p in baseline], (120, 124, 130), 0.0, 1.0, point_radius=7, width=4)
    if collect:
        draw_line_chart(draw, plot_box, [p.iteration for p in collect], [p.success_rate for p in collect], (37, 99, 235), 0.0, 1.0, point_radius=6, width=4)
    if eval_points:
        draw_line_chart(draw, plot_box, [p.iteration for p in eval_points], [p.success_rate for p in eval_points], (220, 38, 38), 0.0, 1.0, point_radius=6, width=4)
    legend_items = [("baseline", (120, 124, 130)), ("collect", (37, 99, 235)), ("eval", (220, 38, 38))]
    lx = plot_box[0]
    ly = 810
    for label, color in legend_items:
        draw.rounded_rectangle((lx, ly, lx + 28, ly + 14), radius=4, fill=color)
        draw.text((lx + 38, ly - 4), label, font=axis_font, fill=(60, 66, 74))
        lx += 150
    out_path = out_dir / "ppo_smoke_success_rate_all.png"
    img.save(out_path)
    return out_path


def render_eval_success_rate_graph(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint]) -> Path:
    img, draw, plot_box, _, _, axis_font = render_graph_canvas(
        "PPO Smoke: Eval Success Rate Only",
        "Most important metric for same-world smoke validation",
    )
    baseline = [p for p in points if p.kind == "baseline"]
    eval_points = [p for p in points if p.kind == "eval"]
    max_iter = max([p.iteration for p in eval_points], default=1)
    x_labels = [(map_point(i, 0, max_iter, plot_box[0], plot_box[2]), str(i)) for i in range(0, max_iter + 1)]
    draw_axes(draw, plot_box, [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")], x_labels, axis_font)
    if baseline:
        draw_line_chart(draw, plot_box, [p.iteration for p in baseline], [p.success_rate for p in baseline], (156, 163, 175), 0.0, 1.0, point_radius=7, width=4)
    if eval_points:
        draw_line_chart(draw, plot_box, [p.iteration for p in eval_points], [p.success_rate for p in eval_points], (190, 24, 93), 0.0, 1.0, point_radius=7, width=5)
        peak = max(eval_points, key=lambda item: item.success_rate)
        px = map_point(float(peak.iteration), 0.0, float(max_iter), plot_box[0], plot_box[2])
        py = map_point(float(peak.success_rate), 0.0, 1.0, plot_box[3], plot_box[1])
        label = f"peak {peak.success_rate:.0%} @ iter {peak.iteration:03d}"
        draw.rounded_rectangle((px + 18, py - 42, px + 300, py - 6), radius=10, fill=(255, 255, 255), outline=(229, 231, 235), width=2)
        draw.text((px + 30, py - 36), label, font=axis_font, fill=(17, 24, 39))
    out_path = out_dir / "ppo_smoke_eval_success_rate.png"
    img.save(out_path)
    return out_path


def render_mean_steps_graph(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint], config: Dict[str, object]) -> Path:
    img, draw, plot_box, _, _, axis_font = render_graph_canvas(
        "PPO Smoke: Mean Episode Steps",
        "Higher values near the max budget indicate failure or inefficient trajectories",
    )
    mapping = phase_map(points)
    baseline = mapping.get("baseline", [])
    collect = mapping.get("collect", [])
    eval_points = mapping.get("eval", [])
    max_iter = max([p.iteration for p in points], default=1)
    budget = float(config.get("step_budget_override", 150) or 150)
    x_labels = [(map_point(i, 0, max_iter, plot_box[0], plot_box[2]), str(i)) for i in range(0, max_iter + 1)]
    y_ticks = [(0.0, "0"), (budget * 0.33, f"{int(round(budget * 0.33))}"), (budget * 0.66, f"{int(round(budget * 0.66))}"), (budget, str(int(budget)))]
    draw.line((plot_box[0], plot_box[3], plot_box[2], plot_box[3]), fill=(120, 128, 140), width=2)
    draw.line((plot_box[0], plot_box[1], plot_box[0], plot_box[3]), fill=(120, 128, 140), width=2)
    for y_val, label in y_ticks:
        yy = map_point(y_val, 0.0, budget, plot_box[3], plot_box[1])
        draw.line((plot_box[0], yy, plot_box[2], yy), fill=(230, 233, 238), width=1)
        draw.text((plot_box[0] - 48, yy - 8), label, font=axis_font, fill=(90, 97, 108))
    for xx, label in x_labels:
        draw.line((xx, plot_box[3], xx, plot_box[3] + 6), fill=(120, 128, 140), width=2)
        draw.text((xx - 18, plot_box[3] + 12), label, font=axis_font, fill=(90, 97, 108))
    if baseline:
        draw_line_chart(draw, plot_box, [p.iteration for p in baseline], [p.mean_steps for p in baseline], (120, 124, 130), 0.0, budget, point_radius=6, width=4)
    if collect:
        draw_line_chart(draw, plot_box, [p.iteration for p in collect], [p.mean_steps for p in collect], (16, 185, 129), 0.0, budget, point_radius=6, width=4)
    if eval_points:
        draw_line_chart(draw, plot_box, [p.iteration for p in eval_points], [p.mean_steps for p in eval_points], (245, 158, 11), 0.0, budget, point_radius=6, width=4)
    draw.line((plot_box[0], map_point(budget, 0.0, budget, plot_box[3], plot_box[1]), plot_box[2], map_point(budget, 0.0, budget, plot_box[3], plot_box[1])), fill=(239, 68, 68), width=2)
    draw.text((plot_box[2] - 170, plot_box[1] + 8), f"max budget = {int(budget)}", font=axis_font, fill=(239, 68, 68))
    out_path = out_dir / "ppo_smoke_mean_steps.png"
    img.save(out_path)
    return out_path


def render_update_kl_graph(pilot_root: Path, out_dir: Path, updates: Sequence[UpdatePoint]) -> Path:
    img, draw, plot_box, _, _, axis_font = render_graph_canvas(
        "PPO Smoke: Update KL Divergence",
        "Large KL spikes indicate aggressive policy movement between iterations",
    )
    max_iter = max([u.iteration for u in updates], default=1)
    max_kl = max(0.05, max([u.mean_kl_divergence for u in updates], default=0.05) * 1.15)
    x_labels = [(map_point(i, 1, max_iter, plot_box[0], plot_box[2]), str(i)) for i in range(1, max_iter + 1)]
    y_ticks = [(0.0, "0"), (max_kl / 3.0, f"{max_kl/3.0:.2f}"), (2.0 * max_kl / 3.0, f"{2.0*max_kl/3.0:.2f}"), (max_kl, f"{max_kl:.2f}")]
    draw.line((plot_box[0], plot_box[3], plot_box[2], plot_box[3]), fill=(120, 128, 140), width=2)
    draw.line((plot_box[0], plot_box[1], plot_box[0], plot_box[3]), fill=(120, 128, 140), width=2)
    for y_val, label in y_ticks:
        yy = map_point(y_val, 0.0, max_kl, plot_box[3], plot_box[1])
        draw.line((plot_box[0], yy, plot_box[2], yy), fill=(230, 233, 238), width=1)
        draw.text((plot_box[0] - 58, yy - 8), label, font=axis_font, fill=(90, 97, 108))
    for xx, label in x_labels:
        draw.line((xx, plot_box[3], xx, plot_box[3] + 6), fill=(120, 128, 140), width=2)
        draw.text((xx - 8, plot_box[3] + 12), label, font=axis_font, fill=(90, 97, 108))
    draw_line_chart(draw, plot_box, [u.iteration for u in updates], [u.mean_kl_divergence for u in updates], (124, 58, 237), 0.0, max_kl, point_radius=7, width=5)
    out_path = out_dir / "ppo_smoke_update_kl.png"
    img.save(out_path)
    return out_path


def render_update_clip_fraction_graph(pilot_root: Path, out_dir: Path, updates: Sequence[UpdatePoint]) -> Path:
    img, draw, plot_box, _, _, axis_font = render_graph_canvas(
        "PPO Smoke: Update Clip Fraction",
        "High clip fraction means many policy ratios are hitting the PPO clipping boundary",
    )
    max_iter = max([u.iteration for u in updates], default=1)
    max_val = max(0.5, max([u.mean_clip_fraction for u in updates], default=0.5) * 1.15)
    x_labels = [(map_point(i, 1, max_iter, plot_box[0], plot_box[2]), str(i)) for i in range(1, max_iter + 1)]
    y_ticks = [(0.0, "0"), (max_val / 3.0, f"{max_val/3.0:.2f}"), (2.0 * max_val / 3.0, f"{2.0*max_val/3.0:.2f}"), (max_val, f"{max_val:.2f}")]
    draw.line((plot_box[0], plot_box[3], plot_box[2], plot_box[3]), fill=(120, 128, 140), width=2)
    draw.line((plot_box[0], plot_box[1], plot_box[0], plot_box[3]), fill=(120, 128, 140), width=2)
    for y_val, label in y_ticks:
        yy = map_point(y_val, 0.0, max_val, plot_box[3], plot_box[1])
        draw.line((plot_box[0], yy, plot_box[2], yy), fill=(230, 233, 238), width=1)
        draw.text((plot_box[0] - 58, yy - 8), label, font=axis_font, fill=(90, 97, 108))
    for xx, label in x_labels:
        draw.line((xx, plot_box[3], xx, plot_box[3] + 6), fill=(120, 128, 140), width=2)
        draw.text((xx - 8, plot_box[3] + 12), label, font=axis_font, fill=(90, 97, 108))
    draw_bar_chart(draw, plot_box, [u.iteration for u in updates], [u.mean_clip_fraction for u in updates], (245, 158, 11), 0.0, max_val)
    out_path = out_dir / "ppo_smoke_update_clip_fraction.png"
    img.save(out_path)
    return out_path


def write_summary_md(pilot_root: Path, out_dir: Path, points: Sequence[PhasePoint], updates: Sequence[UpdatePoint], config: Dict[str, object]) -> Path:
    baseline = next((p for p in points if p.kind == "baseline"), None)
    eval_points = [p for p in points if p.kind == "eval"]
    collect_points = [p for p in points if p.kind == "collect"]
    peak_eval = max(eval_points, key=lambda item: item.success_rate) if eval_points else None
    latest_eval = eval_points[-1] if eval_points else None
    latest_collect = collect_points[-1] if collect_points else None
    latest_update = updates[-1] if updates else None
    lines = [
        "# PPO Smoke Meeting Summary",
        "",
        f"- Pilot root: `{pilot_root}`",
        f"- Current completed eval iteration: `{latest_eval.iteration:03d}`" if latest_eval else "- Current completed eval iteration: n/a",
        f"- Current run status: iter_007 collect is running" if (pilot_root / "iterations" / "iter_007" / "collect").exists() else "- Current run status: no active iter_007 collect detected",
        "",
        "## Run Config",
        "",
        f"- `stop_on_success`: `True`",
        f"- `step_budget_override`: `{config.get('step_budget_override', 'n/a')}`",
        f"- `warmup_noop_steps`: `{config.get('warmup_noop_steps', 'n/a')}`",
        "- `collect/eval/final_eval` world mode: `static`",
        "- `collect/eval/final_eval` world instances: `1`",
        "- `collect_parallel_workers`: `10`",
        "- `eval_parallel_workers`: `10`",
        "- `episodes_per_task` per phase worker aggregate: `10` collect / `10` eval",
        "",
        "## Headline Numbers",
        "",
    ]
    if baseline:
        lines.append(f"- Baseline eval success: `{baseline.success_rate:.0%}` ({int(round(baseline.successes))}/{baseline.count})")
    if peak_eval:
        lines.append(f"- Peak eval success: `{peak_eval.success_rate:.0%}` at `iter_{peak_eval.iteration:03d}`")
    if latest_eval:
        lines.append(f"- Latest completed eval success: `{latest_eval.success_rate:.0%}` at `iter_{latest_eval.iteration:03d}`")
    if latest_collect:
        lines.append(f"- Latest completed collect success: `{latest_collect.success_rate:.0%}` at `iter_{latest_collect.iteration:03d}`")
    if latest_update:
        lines.append(f"- Latest completed update KL: `{latest_update.mean_kl_divergence:.4f}`, clip fraction: `{latest_update.mean_clip_fraction:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PPO update is active, because eval success moved from 10% baseline to 70% at iter_002 on the same static world.",
            "- PPO is unstable under the current setting, because the success curve later collapsed to 0%.",
            "- This is not explained by changing evaluation worlds or changing evaluation seeds inside the smoke run.",
            "- The current training signal is high-variance: only 10 episode fragments per iteration, sparse success-only reward, and per-fragment optimizer stepping.",
            "- Conclusion for the meeting: the PPO plumbing works, but the current regime is too noisy to claim stable improvement.",
        ]
    )
    out_path = out_dir / "ppo_smoke_meeting_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (pilot_root / "meeting_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    points = collect_phase_points(pilot_root)
    updates = collect_update_points(pilot_root)
    config = detect_config(pilot_root)
    write_csv(points, updates, out_dir)
    overview = render_overview_slide(pilot_root, out_dir, points, updates, config)
    diagnostics = render_diag_slide(pilot_root, out_dir, points, updates)
    success_rate_all = render_success_rate_all_graph(pilot_root, out_dir, points)
    eval_success_rate = render_eval_success_rate_graph(pilot_root, out_dir, points)
    mean_steps = render_mean_steps_graph(pilot_root, out_dir, points, config)
    update_kl = render_update_kl_graph(pilot_root, out_dir, updates)
    update_clip_fraction = render_update_clip_fraction_graph(pilot_root, out_dir, updates)
    summary = write_summary_md(pilot_root, out_dir, points, updates, config)

    manifest = {
        "pilot_root": str(pilot_root),
        "out_dir": str(out_dir),
        "overview_png": str(overview),
        "diagnostics_png": str(diagnostics),
        "success_rate_all_png": str(success_rate_all),
        "eval_success_rate_png": str(eval_success_rate),
        "mean_steps_png": str(mean_steps),
        "update_kl_png": str(update_kl),
        "update_clip_fraction_png": str(update_clip_fraction),
        "summary_md": str(summary),
        "phase_metrics_csv": str(out_dir / "phase_metrics.csv"),
        "update_metrics_csv": str(out_dir / "update_metrics.csv"),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
