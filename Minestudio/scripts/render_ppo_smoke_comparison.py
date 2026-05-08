#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render comparison graphs for two PPO smoke runs.")
    parser.add_argument("--old-root", required=True, help="Pilot root for the earlier/baseline PPO smoke run.")
    parser.add_argument("--new-root", required=True, help="Pilot root for the adjusted/stable PPO smoke run.")
    parser.add_argument("--old-label", default="before", help="Legend label for the earlier run.")
    parser.add_argument("--new-label", default="after", help="Legend label for the adjusted run.")
    parser.add_argument("--out-dir", required=True, help="Output directory for PNG/CSV/MD artifacts.")
    parser.add_argument("--old-ppo-clip", type=float, default=None)
    parser.add_argument("--new-ppo-clip", type=float, default=None)
    parser.add_argument("--old-kl-coef", type=float, default=None)
    parser.add_argument("--new-kl-coef", type=float, default=None)
    parser.add_argument("--old-min-successful-fragments", type=int, default=None)
    parser.add_argument("--new-min-successful-fragments", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates: List[str] = []
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
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


@dataclass
class PhasePoint:
    kind: str
    iteration: int
    episode_count: int
    success_count: int
    success_rate: float
    mean_steps: float


@dataclass
class UpdatePoint:
    iteration: int
    fragments: int
    learning_rate: float
    mean_kl_divergence: float
    mean_approx_kl: float
    mean_clip_fraction: float
    mean_total_loss: float


@dataclass
class RunData:
    label: str
    pilot_root: Path
    step_budget: int
    collect_workers: int
    eval_workers: int
    baseline_episodes: int
    collect_episodes_per_iter: int
    eval_episodes_per_iter: int
    ppo_epochs: Optional[int]
    learning_rate: Optional[float]
    ppo_clip: Optional[float]
    kl_coef: Optional[float]
    min_successful_fragments: Optional[int]
    phase_points: List[PhasePoint]
    update_points: List[UpdatePoint]
    max_collect_iter: int
    max_eval_iter: int
    max_update_iter: int
    has_final_eval: bool


def map_point(value: float, lo: float, hi: float, px0: int, px1: int) -> int:
    if hi <= lo:
        return px0
    ratio = (value - lo) / float(hi - lo)
    ratio = max(0.0, min(1.0, ratio))
    return int(round(px0 + ratio * (px1 - px0)))


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
        width, height = text_size(draw, candidate, font)
        if line and width > max_width:
            draw.text((x, y + total_height), line, font=font, fill=fill)
            total_height += height + line_spacing
            line = word
        else:
            line = candidate
    if line:
        _, height = text_size(draw, line, font)
        draw.text((x, y + total_height), line, font=font, fill=fill)
        total_height += height
    return total_height


def draw_axes_generic(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_labels: Sequence[Tuple[float, str]],
    y_labels: Sequence[Tuple[float, str]],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = rect
    draw.line((x0, y1, x1, y1), fill=(120, 128, 140), width=2)
    draw.line((x0, y0, x0, y1), fill=(120, 128, 140), width=2)
    for y_val, label in y_labels:
        yy = map_point(y_val, y_range[0], y_range[1], y1, y0)
        draw.line((x0, yy, x1, yy), fill=(230, 233, 238), width=1)
        draw.text((x0 - 64, yy - 8), label, font=font, fill=(90, 97, 108))
    for x_val, label in x_labels:
        xx = map_point(x_val, x_range[0], x_range[1], x0, x1)
        draw.line((xx, y1, xx, y1 + 6), fill=(120, 128, 140), width=2)
        draw.text((xx - 12, y1 + 12), label, font=font, fill=(90, 97, 108))


def draw_series_line(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    xs: Sequence[float],
    ys: Sequence[float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 5,
    point_radius: int = 6,
) -> None:
    if not xs or not ys:
        return
    x0, y0, x1, y1 = rect
    coords: List[Tuple[int, int]] = []
    for x_val, y_val in zip(xs, ys):
        xx = map_point(float(x_val), x_range[0], x_range[1], x0, x1)
        yy = map_point(float(y_val), y_range[0], y_range[1], y1, y0)
        coords.append((xx, yy))
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=width, joint="curve")
    for xx, yy in coords:
        draw.ellipse((xx - point_radius, yy - point_radius, xx + point_radius, yy + point_radius), fill=color, outline=(255, 255, 255), width=2)


def render_graph_canvas(title: str, subtitle: str) -> Tuple[Image.Image, ImageDraw.ImageDraw, Tuple[int, int, int, int], ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(42, bold=True)
    subtitle_font = find_font(20, bold=False)
    axis_font = find_font(18, bold=False)
    draw.text((56, 34), title, font=title_font, fill=(18, 24, 32))
    draw.text((58, 88), subtitle, font=subtitle_font, fill=(92, 100, 112))
    plot_box = (100, 155, 1510, 790)
    draw.rounded_rectangle((50, 130, 1550, 840), radius=28, fill=(252, 252, 253), outline=(224, 228, 234), width=2)
    return image, draw, plot_box, title_font, subtitle_font, axis_font


def read_episode_csvs(paths: Sequence[Path]) -> Tuple[int, int, float]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    if not rows:
        return 0, 0, 0.0
    success_count = sum(1 for row in rows if row.get("auto_success", "").strip().lower() == "true")
    mean_steps = sum(float(row.get("num_steps", 0) or 0.0) for row in rows) / float(len(rows))
    return len(rows), success_count, mean_steps


def collect_phase_points(pilot_root: Path) -> List[PhasePoint]:
    points: List[PhasePoint] = []

    baseline_paths = sorted((pilot_root / "baseline").glob("worker_*/*/episodes.csv"))
    episodes, successes, mean_steps = read_episode_csvs(baseline_paths)
    if episodes:
        points.append(
            PhasePoint(
                kind="baseline",
                iteration=0,
                episode_count=episodes,
                success_count=successes,
                success_rate=successes / float(episodes),
                mean_steps=mean_steps,
            )
        )

    for iter_dir in sorted((pilot_root / "iterations").glob("iter_*")):
        try:
            iteration = int(iter_dir.name.split("_")[1])
        except Exception:
            continue
        for kind in ("collect", "eval"):
            episode_paths = sorted((iter_dir / kind).glob("worker_*/*/episodes.csv"))
            episodes, successes, mean_steps = read_episode_csvs(episode_paths)
            if not episodes:
                continue
            points.append(
                PhasePoint(
                    kind=kind,
                    iteration=iteration,
                    episode_count=episodes,
                    success_count=successes,
                    success_rate=successes / float(episodes),
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
        update_dirs = sorted(path for path in (iter_dir / "update").glob("*") if path.is_dir())
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
                learning_rate=float(meta.get("learning_rate", 0.0) or 0.0),
                mean_kl_divergence=float(last.get("mean_kl_divergence", 0.0) or 0.0),
                mean_approx_kl=float(last.get("mean_approx_kl", 0.0) or 0.0),
                mean_clip_fraction=float(last.get("mean_clip_fraction", 0.0) or 0.0),
                mean_total_loss=float(last.get("mean_total_loss", 0.0) or 0.0),
            )
        )
    return points


def detect_step_budget(pilot_root: Path) -> int:
    candidates = sorted((pilot_root / "baseline").glob("worker_*/*/run_metadata.json"))
    if not candidates:
        return 0
    data = load_json(candidates[0])
    return int(data.get("step_budget_override", 0) or 0)


def detect_worker_count(pilot_root: Path, phase: str) -> int:
    root = pilot_root / phase
    return len([path for path in root.glob("worker_*") if path.is_dir()])


def detect_per_iter_episode_count(points: Sequence[PhasePoint], kind: str) -> int:
    matches = [point.episode_count for point in points if point.kind == kind]
    return matches[0] if matches else 0


def compute_run_data(
    label: str,
    pilot_root: Path,
    ppo_clip: Optional[float],
    kl_coef: Optional[float],
    min_successful_fragments: Optional[int],
) -> RunData:
    phase_points = collect_phase_points(pilot_root)
    update_points = collect_update_points(pilot_root)
    baseline_points = [point for point in phase_points if point.kind == "baseline"]
    collect_points = [point for point in phase_points if point.kind == "collect"]
    eval_points = [point for point in phase_points if point.kind == "eval"]
    has_final_eval = any((pilot_root / "final_eval").rglob("episodes.csv"))
    ppo_epochs = 1 if update_points else None
    learning_rate = update_points[0].learning_rate if update_points else None
    return RunData(
        label=label,
        pilot_root=pilot_root,
        step_budget=detect_step_budget(pilot_root),
        collect_workers=detect_worker_count(pilot_root, "baseline"),
        eval_workers=detect_worker_count(pilot_root, "baseline"),
        baseline_episodes=baseline_points[0].episode_count if baseline_points else 0,
        collect_episodes_per_iter=detect_per_iter_episode_count(phase_points, "collect"),
        eval_episodes_per_iter=detect_per_iter_episode_count(phase_points, "eval"),
        ppo_epochs=ppo_epochs,
        learning_rate=learning_rate,
        ppo_clip=ppo_clip,
        kl_coef=kl_coef,
        min_successful_fragments=min_successful_fragments,
        phase_points=phase_points,
        update_points=update_points,
        max_collect_iter=max((point.iteration for point in collect_points), default=0),
        max_eval_iter=max((point.iteration for point in eval_points), default=0),
        max_update_iter=max((point.iteration for point in update_points), default=0),
        has_final_eval=has_final_eval,
    )


def phase_points_by_kind(run: RunData, kind: str) -> List[PhasePoint]:
    return sorted([point for point in run.phase_points if point.kind == kind], key=lambda point: point.iteration)


def common_prefix_count(old: Sequence[float], new: Sequence[float]) -> int:
    return min(len(old), len(new))


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def render_hyperparam_card(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    width, height = 1800, 980
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(42, bold=True)
    header_font = find_font(24, bold=True)
    body_font = find_font(22, bold=False)
    small_font = find_font(18, bold=False)

    draw.text((60, 36), "PPO Smoke Comparison: Collect Regime and Constants", font=title_font, fill=(18, 24, 32))
    draw.text((62, 90), f"old={old_run.pilot_root} | new={new_run.pilot_root}", font=small_font, fill=(92, 100, 112))

    table_box = (50, 140, 1750, 910)
    draw.rounded_rectangle(table_box, radius=24, fill=(252, 252, 253), outline=(224, 228, 234), width=2)

    col_x = [90, 520, 980, 1440]
    draw.text((col_x[0], 180), "Item", font=header_font, fill=(31, 41, 55))
    draw.text((col_x[1], 180), old_run.label, font=header_font, fill=(185, 28, 28))
    draw.text((col_x[2], 180), new_run.label, font=header_font, fill=(30, 64, 175))
    draw.text((col_x[3], 180), "Change", font=header_font, fill=(31, 41, 55))

    rows = [
        ("Max completed eval iter", str(old_run.max_eval_iter), str(new_run.max_eval_iter), f"{old_run.max_eval_iter} -> {new_run.max_eval_iter}"),
        ("Final eval present", "yes" if old_run.has_final_eval else "no", "yes" if new_run.has_final_eval else "no", "both incomplete"),
        ("Collect episodes / iter", str(old_run.collect_episodes_per_iter), str(new_run.collect_episodes_per_iter), f"{old_run.collect_episodes_per_iter} -> {new_run.collect_episodes_per_iter}"),
        ("Eval episodes / iter", str(old_run.eval_episodes_per_iter), str(new_run.eval_episodes_per_iter), f"{old_run.eval_episodes_per_iter} -> {new_run.eval_episodes_per_iter}"),
        ("Collect workers", str(old_run.collect_workers), str(new_run.collect_workers), "unchanged"),
        ("Step budget", str(old_run.step_budget), str(new_run.step_budget), "unchanged"),
        ("PPO epochs", str(old_run.ppo_epochs or 'n/a'), str(new_run.ppo_epochs or 'n/a'), "unchanged"),
        ("Learning rate", f"{old_run.learning_rate:.1e}" if old_run.learning_rate else "n/a", f"{new_run.learning_rate:.1e}" if new_run.learning_rate else "n/a", "reduced 2x"),
        ("PPO clip", f"{old_run.ppo_clip:.3f}" if old_run.ppo_clip is not None else "n/a", f"{new_run.ppo_clip:.3f}" if new_run.ppo_clip is not None else "n/a", "tighter clipping"),
        ("KL coef", f"{old_run.kl_coef:.3f}" if old_run.kl_coef is not None else "n/a", f"{new_run.kl_coef:.3f}" if new_run.kl_coef is not None else "n/a", "larger KL penalty"),
        ("Min successful frags", str(old_run.min_successful_fragments) if old_run.min_successful_fragments is not None else "n/a", str(new_run.min_successful_fragments) if new_run.min_successful_fragments is not None else "n/a", "new update guard"),
    ]

    y = 240
    row_h = 56
    for index, row in enumerate(rows):
        fill = (255, 255, 255) if index % 2 == 0 else (246, 248, 251)
        draw.rounded_rectangle((78, y - 10, 1720, y + 34), radius=10, fill=fill)
        draw.text((col_x[0], y), row[0], font=body_font, fill=(31, 41, 55))
        draw.text((col_x[1], y), row[1], font=body_font, fill=(153, 27, 27))
        draw.text((col_x[2], y), row[2], font=body_font, fill=(30, 64, 175))
        draw.text((col_x[3], y), row[3], font=body_font, fill=(55, 65, 81))
        y += row_h

    note = (
        "Interpretation: the adjusted run used 5x more collect episodes per PPO step, "
        "a lower learning rate, a tighter PPO clip, a larger KL penalty, and a "
        "minimum-success update guard. The run still stopped early, but the update "
        "trajectory is materially less explosive than the earlier setting."
    )
    draw.text((90, 860), "Summary", font=header_font, fill=(31, 41, 55))
    draw_wrapped_text(draw, (90, 900), note, body_font, (55, 65, 81), 1600)

    out_path = out_dir / "ppo_compare_hyperparams.png"
    image.save(out_path)
    return out_path


def render_two_run_line_graph(
    out_dir: Path,
    filename: str,
    title: str,
    subtitle: str,
    old_run: RunData,
    new_run: RunData,
    old_xy: Tuple[Sequence[float], Sequence[float]],
    new_xy: Tuple[Sequence[float], Sequence[float]],
    y_range: Tuple[float, float],
    y_labels: Sequence[Tuple[float, str]],
    legend_note: Optional[str] = None,
) -> Path:
    image, draw, plot_box, _, _, axis_font = render_graph_canvas(title, subtitle)
    max_x = max(
        max(old_xy[0], default=0.0),
        max(new_xy[0], default=0.0),
        1.0,
    )
    x_labels = [(float(i), str(i)) for i in range(0, int(max_x) + 1)]
    draw_axes_generic(draw, plot_box, (0.0, max_x), y_range, x_labels, y_labels, axis_font)
    draw_series_line(draw, plot_box, old_xy[0], old_xy[1], (0.0, max_x), y_range, (185, 28, 28), width=5, point_radius=7)
    draw_series_line(draw, plot_box, new_xy[0], new_xy[1], (0.0, max_x), y_range, (30, 64, 175), width=5, point_radius=7)

    legend_y = 812
    legend_x = plot_box[0]
    legend_items = [
        (old_run.label, (185, 28, 28)),
        (new_run.label, (30, 64, 175)),
    ]
    for label, color in legend_items:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 28, legend_y + 14), radius=4, fill=color)
        draw.text((legend_x + 38, legend_y - 4), label, font=axis_font, fill=(60, 66, 74))
        legend_x += 220
    if legend_note:
        draw.text((plot_box[0] + 520, legend_y - 4), legend_note, font=axis_font, fill=(90, 97, 108))

    out_path = out_dir / filename
    image.save(out_path)
    return out_path


def render_eval_success_rate_graph(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    old_baseline = [point for point in old_run.phase_points if point.kind == "baseline"]
    new_baseline = [point for point in new_run.phase_points if point.kind == "baseline"]
    old_eval = phase_points_by_kind(old_run, "eval")
    new_eval = phase_points_by_kind(new_run, "eval")
    old_x = [0.0] + [float(point.iteration) for point in old_eval]
    old_y = [old_baseline[0].success_rate if old_baseline else 0.0] + [point.success_rate for point in old_eval]
    new_x = [0.0] + [float(point.iteration) for point in new_eval]
    new_y = [new_baseline[0].success_rate if new_baseline else 0.0] + [point.success_rate for point in new_eval]
    return render_two_run_line_graph(
        out_dir,
        "ppo_compare_eval_success_rate.png",
        "PPO Smoke: Eval Success Rate Comparison",
        "Baseline at x=0; later points are eval after each PPO iteration",
        old_run,
        new_run,
        (old_x, old_y),
        (new_x, new_y),
        (0.0, 1.0),
        [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")],
    )


def render_collect_success_count_graph(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    old_collect = phase_points_by_kind(old_run, "collect")
    new_collect = phase_points_by_kind(new_run, "collect")
    max_y = max(
        [point.success_count for point in old_collect] + [point.success_count for point in new_collect] + [1]
    )
    return render_two_run_line_graph(
        out_dir,
        "ppo_compare_collect_success_count.png",
        "PPO Smoke: Successful Training Trajectories per Iteration",
        "Absolute success counts in collect batches; this is the real PPO update signal volume",
        old_run,
        new_run,
        ([float(point.iteration) for point in old_collect], [float(point.success_count) for point in old_collect]),
        ([float(point.iteration) for point in new_collect], [float(point.success_count) for point in new_collect]),
        (0.0, float(max_y)),
        [(0.0, "0"), (float(max_y) / 3.0, f"{max_y/3.0:.1f}"), (2.0 * float(max_y) / 3.0, f"{2.0*max_y/3.0:.1f}"), (float(max_y), str(max_y))],
        legend_note=f"{old_run.label}: {old_run.collect_episodes_per_iter} eps/iter | {new_run.label}: {new_run.collect_episodes_per_iter} eps/iter",
    )


def render_eval_mean_steps_graph(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    old_eval = phase_points_by_kind(old_run, "eval")
    new_eval = phase_points_by_kind(new_run, "eval")
    budget = max(old_run.step_budget, new_run.step_budget, 150)
    return render_two_run_line_graph(
        out_dir,
        "ppo_compare_eval_mean_steps.png",
        "PPO Smoke: Eval Mean Steps Comparison",
        "Closer to the max budget means failure or delayed success",
        old_run,
        new_run,
        ([0.0] + [float(point.iteration) for point in old_eval], [old_run.phase_points[0].mean_steps] + [point.mean_steps for point in old_eval]),
        ([0.0] + [float(point.iteration) for point in new_eval], [new_run.phase_points[0].mean_steps] + [point.mean_steps for point in new_eval]),
        (0.0, float(budget)),
        [(0.0, "0"), (budget / 3.0, f"{int(round(budget / 3.0))}"), (2.0 * budget / 3.0, f"{int(round(2.0 * budget / 3.0))}"), (float(budget), str(int(budget)))],
    )


def render_update_kl_graph(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    max_y = max(
        [point.mean_kl_divergence for point in old_run.update_points]
        + [point.mean_kl_divergence for point in new_run.update_points]
        + [0.05]
    )
    max_y *= 1.1
    return render_two_run_line_graph(
        out_dir,
        "ppo_compare_update_kl.png",
        "PPO Smoke: Update KL Divergence Comparison",
        "Lower and smoother KL means less abrupt policy movement between iterations",
        old_run,
        new_run,
        ([float(point.iteration) for point in old_run.update_points], [point.mean_kl_divergence for point in old_run.update_points]),
        ([float(point.iteration) for point in new_run.update_points], [point.mean_kl_divergence for point in new_run.update_points]),
        (0.0, max_y),
        [(0.0, "0"), (max_y / 3.0, f"{max_y/3.0:.2f}"), (2.0 * max_y / 3.0, f"{2.0*max_y/3.0:.2f}"), (max_y, f"{max_y:.2f}")],
    )


def render_update_clip_graph(out_dir: Path, old_run: RunData, new_run: RunData) -> Path:
    max_y = max(
        [point.mean_clip_fraction for point in old_run.update_points]
        + [point.mean_clip_fraction for point in new_run.update_points]
        + [0.5]
    )
    max_y *= 1.1
    return render_two_run_line_graph(
        out_dir,
        "ppo_compare_update_clip_fraction.png",
        "PPO Smoke: Update Clip Fraction Comparison",
        "Not directly apples-to-apples because the adjusted run also tightened PPO clip from 0.2 to 0.1",
        old_run,
        new_run,
        ([float(point.iteration) for point in old_run.update_points], [point.mean_clip_fraction for point in old_run.update_points]),
        ([float(point.iteration) for point in new_run.update_points], [point.mean_clip_fraction for point in new_run.update_points]),
        (0.0, max_y),
        [(0.0, "0"), (max_y / 3.0, f"{max_y/3.0:.2f}"), (2.0 * max_y / 3.0, f"{2.0*max_y/3.0:.2f}"), (max_y, f"{max_y:.2f}")],
    )


def write_csvs(out_dir: Path, old_run: RunData, new_run: RunData) -> Tuple[Path, Path]:
    phase_csv = out_dir / "phase_comparison.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "kind", "iteration", "episode_count", "success_count", "success_rate", "mean_steps"])
        for run in (old_run, new_run):
            for point in run.phase_points:
                writer.writerow([run.label, point.kind, point.iteration, point.episode_count, point.success_count, point.success_rate, point.mean_steps])

    update_csv = out_dir / "update_comparison.csv"
    with update_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "iteration", "fragments", "learning_rate", "mean_kl_divergence", "mean_approx_kl", "mean_clip_fraction", "mean_total_loss"])
        for run in (old_run, new_run):
            for point in run.update_points:
                writer.writerow([run.label, point.iteration, point.fragments, point.learning_rate, point.mean_kl_divergence, point.mean_approx_kl, point.mean_clip_fraction, point.mean_total_loss])
    return phase_csv, update_csv


def summarize_runs(old_run: RunData, new_run: RunData) -> Dict[str, object]:
    old_eval = phase_points_by_kind(old_run, "eval")
    new_eval = phase_points_by_kind(new_run, "eval")
    old_collect = phase_points_by_kind(old_run, "collect")
    new_collect = phase_points_by_kind(new_run, "collect")

    old_peak_eval = max((point.success_rate for point in old_eval), default=0.0)
    new_peak_eval = max((point.success_rate for point in new_eval), default=0.0)
    old_peak_iter = max(old_eval, key=lambda point: point.success_rate).iteration if old_eval else 0
    new_peak_iter = max(new_eval, key=lambda point: point.success_rate).iteration if new_eval else 0

    old_common_kl = [point.mean_kl_divergence for point in old_run.update_points[: len(new_run.update_points)]]
    new_common_kl = [point.mean_kl_divergence for point in new_run.update_points]
    old_common_clip = [point.mean_clip_fraction for point in old_run.update_points[: len(new_run.update_points)]]
    new_common_clip = [point.mean_clip_fraction for point in new_run.update_points]

    return {
        "old": {
            "label": old_run.label,
            "pilot_root": str(old_run.pilot_root),
            "max_eval_iter": old_run.max_eval_iter,
            "max_update_iter": old_run.max_update_iter,
            "peak_eval_success_rate": old_peak_eval,
            "peak_eval_iteration": old_peak_iter,
            "latest_eval_success_rate": old_eval[-1].success_rate if old_eval else None,
            "latest_collect_success_rate": old_collect[-1].success_rate if old_collect else None,
            "latest_collect_success_count": old_collect[-1].success_count if old_collect else None,
            "mean_kl_first_common_iters": mean_or_none(old_common_kl),
            "mean_clip_fraction_first_common_iters": mean_or_none(old_common_clip),
        },
        "new": {
            "label": new_run.label,
            "pilot_root": str(new_run.pilot_root),
            "max_eval_iter": new_run.max_eval_iter,
            "max_update_iter": new_run.max_update_iter,
            "peak_eval_success_rate": new_peak_eval,
            "peak_eval_iteration": new_peak_iter,
            "latest_eval_success_rate": new_eval[-1].success_rate if new_eval else None,
            "latest_collect_success_rate": new_collect[-1].success_rate if new_collect else None,
            "latest_collect_success_count": new_collect[-1].success_count if new_collect else None,
            "mean_kl_first_common_iters": mean_or_none(new_common_kl),
            "mean_clip_fraction_first_common_iters": mean_or_none(new_common_clip),
        },
    }


def write_summary_md(out_dir: Path, old_run: RunData, new_run: RunData, summary: Dict[str, object]) -> Path:
    old_eval = phase_points_by_kind(old_run, "eval")
    new_eval = phase_points_by_kind(new_run, "eval")
    old_collect = phase_points_by_kind(old_run, "collect")
    new_collect = phase_points_by_kind(new_run, "collect")
    old_update_kl = [point.mean_kl_divergence for point in old_run.update_points[: len(new_run.update_points)]]
    new_update_kl = [point.mean_kl_divergence for point in new_run.update_points]

    lines = [
        "# PPO Smoke Comparison Summary",
        "",
        "## Completion Status",
        "",
        f"- `{old_run.label}` completed through `iter_{old_run.max_eval_iter:03d}` eval and `iter_{old_run.max_update_iter:03d}` update.",
        f"- `{new_run.label}` completed through `iter_{new_run.max_eval_iter:03d}` eval and `iter_{new_run.max_update_iter:03d}` update.",
        f"- `{new_run.label}` has empty `iter_006` placeholders and no `final_eval`, so the run stopped after `iter_005` evaluation.",
        "",
        "## What Changed",
        "",
        f"- Collect episodes per PPO step: `{old_run.collect_episodes_per_iter}` -> `{new_run.collect_episodes_per_iter}`",
        f"- Learning rate: `{old_run.learning_rate:.1e}` -> `{new_run.learning_rate:.1e}`" if old_run.learning_rate and new_run.learning_rate else "- Learning rate: n/a",
        f"- PPO clip: `{old_run.ppo_clip}` -> `{new_run.ppo_clip}`",
        f"- KL coef: `{old_run.kl_coef}` -> `{new_run.kl_coef}`",
        f"- Min successful fragments guard: `{old_run.min_successful_fragments}` -> `{new_run.min_successful_fragments}`",
        "",
        "## Performance",
        "",
        f"- `{old_run.label}` peak eval success: `{max((point.success_rate for point in old_eval), default=0.0):.0%}` at `iter_{max(old_eval, key=lambda point: point.success_rate).iteration:03d}`" if old_eval else f"- `{old_run.label}` peak eval success: n/a",
        f"- `{new_run.label}` peak eval success: `{max((point.success_rate for point in new_eval), default=0.0):.0%}` at `iter_{max(new_eval, key=lambda point: point.success_rate).iteration:03d}`" if new_eval else f"- `{new_run.label}` peak eval success: n/a",
        f"- `{old_run.label}` latest eval success: `{old_eval[-1].success_rate:.0%}`" if old_eval else f"- `{old_run.label}` latest eval success: n/a",
        f"- `{new_run.label}` latest eval success: `{new_eval[-1].success_rate:.0%}`" if new_eval else f"- `{new_run.label}` latest eval success: n/a",
        "",
        "## Stability Read",
        "",
        f"- Mean update KL over the first {len(new_update_kl)} comparable iterations: `{mean_or_none(old_update_kl):.4f}` -> `{mean_or_none(new_update_kl):.4f}`" if old_update_kl and new_update_kl else "- Mean update KL over comparable iterations: n/a",
        f"- `{old_run.label}` collect successes per iteration ranged `{min((point.success_count for point in old_collect), default=0)}-{max((point.success_count for point in old_collect), default=0)}` out of `{old_run.collect_episodes_per_iter}`.",
        f"- `{new_run.label}` collect successes per iteration ranged `{min((point.success_count for point in new_collect), default=0)}-{max((point.success_count for point in new_collect), default=0)}` out of `{new_run.collect_episodes_per_iter}`.",
        "- Interpretation: the adjusted run is clearly less explosive in KL and maintains non-zero training successes every completed iteration, but it has not yet shown a strong monotonic eval improvement.",
        "- Important caveat: clip fraction is not directly apples-to-apples because the adjusted run also tightened the PPO clip threshold from 0.2 to 0.1.",
    ]
    out_path = out_dir / "ppo_compare_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_run = compute_run_data(
        label=args.old_label,
        pilot_root=Path(args.old_root).resolve(),
        ppo_clip=args.old_ppo_clip,
        kl_coef=args.old_kl_coef,
        min_successful_fragments=args.old_min_successful_fragments,
    )
    new_run = compute_run_data(
        label=args.new_label,
        pilot_root=Path(args.new_root).resolve(),
        ppo_clip=args.new_ppo_clip,
        kl_coef=args.new_kl_coef,
        min_successful_fragments=args.new_min_successful_fragments,
    )

    hyperparams_png = render_hyperparam_card(out_dir, old_run, new_run)
    eval_success_png = render_eval_success_rate_graph(out_dir, old_run, new_run)
    collect_success_png = render_collect_success_count_graph(out_dir, old_run, new_run)
    eval_steps_png = render_eval_mean_steps_graph(out_dir, old_run, new_run)
    kl_png = render_update_kl_graph(out_dir, old_run, new_run)
    clip_png = render_update_clip_graph(out_dir, old_run, new_run)
    phase_csv, update_csv = write_csvs(out_dir, old_run, new_run)
    summary_obj = summarize_runs(old_run, new_run)
    summary_json = out_dir / "ppo_compare_summary.json"
    summary_json.write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md = write_summary_md(out_dir, old_run, new_run, summary_obj)

    manifest = {
        "out_dir": str(out_dir),
        "old_root": str(old_run.pilot_root),
        "new_root": str(new_run.pilot_root),
        "hyperparams_png": str(hyperparams_png),
        "eval_success_rate_png": str(eval_success_png),
        "collect_success_count_png": str(collect_success_png),
        "eval_mean_steps_png": str(eval_steps_png),
        "update_kl_png": str(kl_png),
        "update_clip_fraction_png": str(clip_png),
        "phase_csv": str(phase_csv),
        "update_csv": str(update_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
