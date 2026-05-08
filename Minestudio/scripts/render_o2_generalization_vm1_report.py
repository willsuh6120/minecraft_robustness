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
    parser = argparse.ArgumentParser(description="Render PPT-ready PNGs and markdown for VM1 O2 generalization run.")
    parser.add_argument("--import-root", required=True, help="Imported VM1 run root.")
    parser.add_argument("--out-dir", default="", help="Optional output directory.")
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
        _, h = text_size(draw, line, font)
        draw.text((x, y + total_height), line, font=font, fill=fill)
        total_height += h
    return total_height


def pattern_code(row: Sequence[str]) -> str:
    out = []
    for cell in row:
        label = str(cell).lower()
        out.append("G" if "glass" in label else "L")
    return "".join(out)


def safe_rate(successes: int, episodes: int) -> float:
    return float(successes) / float(episodes) if episodes else 0.0


def fmt_pct(rate: float) -> str:
    return f"{rate * 100.0:.1f}%"


def blend(base: Tuple[int, int, int], top: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return tuple(int(round(base[i] * (1.0 - alpha) + top[i] * alpha)) for i in range(3))


@dataclass
class PhaseEvalCell:
    success: bool
    steps: int
    seed: int
    sampling_seed: int


@dataclass
class IterationSummary:
    iteration: int
    collect_successes: int
    collect_episodes: int
    collect_rate: float
    eval_rate: float
    eval_root: Path
    eval_results: Dict[int, PhaseEvalCell]
    approx_kl: float
    clip_fraction: float
    value_loss: float


@dataclass
class BlockSummary:
    block_idx: int
    block_tag: str
    run_dir: Path
    collect_pattern: str
    collect_pattern_variant: str
    baseline_rate: float
    baseline_root: Path
    baseline_results: Dict[int, PhaseEvalCell]
    iterations: List[IterationSummary]
    best_iteration: int
    best_eval_rate: float
    final_eval_rate: Optional[float]
    final_eval_root: Optional[Path]
    final_eval_results: Dict[int, PhaseEvalCell]


def latest_subdir(path: Path) -> Optional[Path]:
    candidates = sorted([child for child in path.iterdir() if child.is_dir()])
    return candidates[-1] if candidates else None


def load_phase_results(phase_root: Path) -> Dict[int, PhaseEvalCell]:
    results: Dict[int, PhaseEvalCell] = {}
    if not phase_root.exists():
        return results
    for instance_dir in sorted(phase_root.glob("instance_*")):
        try:
            instance_idx = int(instance_dir.name.split("_")[1])
        except Exception:
            continue
        run_dir = latest_subdir(instance_dir)
        if run_dir is None:
            continue
        episodes_path = run_dir / "episodes.csv"
        if not episodes_path.exists():
            continue
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        row = rows[0]
        results[instance_idx] = PhaseEvalCell(
            success=str(row.get("auto_success", "")).lower() == "true",
            steps=int(float(row.get("num_steps") or 0)),
            seed=int(float(row.get("seed") or 0)),
            sampling_seed=int(float(row.get("sampling_seed") or 0)),
        )
    return results


def discover_blocks(import_root: Path) -> List[BlockSummary]:
    blocks: List[BlockSummary] = []
    blocks_root = import_root / "blocks"
    for block_dir in sorted(blocks_root.glob("block_*")):
        run_dir = latest_subdir(block_dir)
        if run_dir is None:
            continue
        pilot_state = load_json(run_dir / "pilot_state.json")
        block_idx = int(block_dir.name.split("_")[1])
        collect_world = pilot_state["iteration_summaries"][0]["collect_world"]
        collect_plan = collect_world["instance_worlds"][0]["plan_rows"][0]["world_generation_suggestions"]
        collect_pattern = f"{pattern_code(collect_plan['mine_occluder_top_row'])}/{pattern_code(collect_plan['mine_occluder_bottom_row'])}"
        collect_variant = str(collect_plan.get("mine_occluder_variant_id", ""))

        baseline_rate = float(next(iter(pilot_state["baseline_summary"].values()))["auto_success_rate"])
        baseline_root = run_dir / "baseline"
        baseline_results = load_phase_results(baseline_root)

        iterations: List[IterationSummary] = []
        for it in pilot_state["iteration_summaries"]:
            iteration = int(it["iteration"])
            collect_stats = it["collect_episode_stats"]
            collect_successes = int(collect_stats["successful_episodes"])
            collect_episodes = int(collect_stats["episodes"])
            eval_rate = float(next(iter(it["eval_summary"].values()))["auto_success_rate"])
            eval_root = run_dir / "iterations" / f"iter_{iteration:03d}" / "eval"
            eval_results = load_phase_results(eval_root)

            update_root = run_dir / "iterations" / f"iter_{iteration:03d}" / "update"
            update_run_dir = latest_subdir(update_root)
            if update_run_dir is None:
                raise FileNotFoundError(f"Missing local update dir under {update_root}")
            train_history = load_json(update_run_dir / "train_history.json")
            last = train_history[-1]
            iterations.append(
                IterationSummary(
                    iteration=iteration,
                    collect_successes=collect_successes,
                    collect_episodes=collect_episodes,
                    collect_rate=safe_rate(collect_successes, collect_episodes),
                    eval_rate=eval_rate,
                    eval_root=eval_root,
                    eval_results=eval_results,
                    approx_kl=float(last.get("mean_approx_kl", 0.0) or 0.0),
                    clip_fraction=float(last.get("mean_clip_fraction", 0.0) or 0.0),
                    value_loss=float(last.get("mean_value_loss", 0.0) or 0.0),
                )
            )

        best_iteration = int(pilot_state["best_eval_selection"]["iteration"])
        best_eval_rate = float(pilot_state["best_eval_selection"]["auto_success_rate"])
        final_eval_summary = pilot_state.get("final_eval_summary") or {}
        final_eval_rate: Optional[float] = None
        final_eval_root: Optional[Path] = None
        final_eval_results: Dict[int, PhaseEvalCell] = {}
        if final_eval_summary:
            final_eval_rate = float(next(iter(final_eval_summary.values()))["auto_success_rate"])
            final_eval_root = run_dir / "final_eval"
            final_eval_results = load_phase_results(final_eval_root)

        blocks.append(
            BlockSummary(
                block_idx=block_idx,
                block_tag=block_dir.name,
                run_dir=run_dir,
                collect_pattern=collect_pattern,
                collect_pattern_variant=collect_variant,
                baseline_rate=baseline_rate,
                baseline_root=baseline_root,
                baseline_results=baseline_results,
                iterations=iterations,
                best_iteration=best_iteration,
                best_eval_rate=best_eval_rate,
                final_eval_rate=final_eval_rate,
                final_eval_root=final_eval_root,
                final_eval_results=final_eval_results,
            )
        )
    return blocks


def extract_eval_bank_patterns(blocks: Sequence[BlockSummary]) -> List[Tuple[int, str, str]]:
    pilot_state = load_json(blocks[0].run_dir / "pilot_state.json")
    rows = []
    for world in pilot_state["baseline_world"]["instance_worlds"]:
        idx = int(world["instance_idx"])
        sugg = world["plan_rows"][0]["world_generation_suggestions"]
        pattern = f"{pattern_code(sugg['mine_occluder_top_row'])}/{pattern_code(sugg['mine_occluder_bottom_row'])}"
        variant = str(sugg.get("mine_occluder_variant_id", ""))
        rows.append((idx, variant, pattern))
    return rows


def extract_final_eval_patterns(blocks: Sequence[BlockSummary]) -> List[Tuple[int, str, str]]:
    pilot_state = load_json(blocks[-1].run_dir / "pilot_state.json")
    rows = []
    for world in pilot_state["final_eval_world"]["instance_worlds"]:
        idx = int(world["instance_idx"])
        sugg = world["plan_rows"][0]["world_generation_suggestions"]
        pattern = f"{pattern_code(sugg['mine_occluder_top_row'])}/{pattern_code(sugg['mine_occluder_bottom_row'])}"
        variant = str(sugg.get("mine_occluder_variant_id", ""))
        rows.append((idx, variant, pattern))
    return rows


def render_block_eval_summary(blocks: Sequence[BlockSummary], out_path: Path) -> None:
    width, height = 2200, 1100
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)

    title_font = find_font(42, bold=True)
    subtitle_font = find_font(22)
    axis_font = find_font(20)
    label_font = find_font(18)
    small_font = find_font(16)

    draw.text((70, 46), "VM1 O2 Generalization: Baseline vs Best Eval by Collect Block", fill=(17, 24, 39), font=title_font)
    draw.text((70, 102), "Bars show eval_bank success rates. Red star shows final_eval_bank for block 5.", fill=(75, 85, 99), font=subtitle_font)

    plot = (140, 220, 2060, 860)
    left, top, right, bottom = plot
    y_max = 0.5
    for tick_idx in range(6):
        rate = y_max * tick_idx / 5.0
        y = bottom - (bottom - top) * (rate / y_max)
        draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
        label = f"{rate:.1f}"
        tw, th = text_size(draw, label, axis_font)
        draw.text((left - tw - 16, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
    draw.line((left, top, left, bottom), fill=(55, 65, 81), width=3)
    draw.line((left, bottom, right, bottom), fill=(55, 65, 81), width=3)

    baseline_color = (156, 163, 175)
    best_color = (29, 78, 216)
    final_color = (185, 28, 28)
    group_w = (right - left) / max(1, len(blocks))
    bar_w = group_w * 0.22

    def y_for(rate: float) -> float:
        return bottom - (bottom - top) * (rate / y_max)

    # Legend
    legend_x = left + 20
    legend_y = top - 78
    legend_items = [("Baseline Eval", baseline_color), ("Best Eval", best_color), ("Final Eval Bank", final_color)]
    cursor_x = legend_x
    for text, color in legend_items:
        draw.rectangle((cursor_x, legend_y, cursor_x + 24, legend_y + 24), fill=color, outline=color)
        draw.text((cursor_x + 34, legend_y - 1), text, fill=(31, 41, 55), font=label_font)
        cursor_x += 34 + text_size(draw, text, label_font)[0] + 34

    for idx, block in enumerate(blocks):
        center_x = left + group_w * (idx + 0.5)
        bx0 = center_x - bar_w - 12
        bx1 = center_x - 12
        ex0 = center_x + 12
        ex1 = center_x + bar_w + 12

        by = y_for(block.baseline_rate)
        ey = y_for(block.best_eval_rate)
        draw.rectangle((bx0, by, bx1, bottom), fill=baseline_color)
        draw.rectangle((ex0, ey, ex1, bottom), fill=best_color)
        draw.text((bx0, by - 28), fmt_pct(block.baseline_rate), fill=(55, 65, 81), font=small_font)
        draw.text((ex0, ey - 28), fmt_pct(block.best_eval_rate), fill=(29, 78, 216), font=small_font)

        label = f"B{block.block_idx}"
        tw, th = text_size(draw, label, axis_font)
        draw.text((center_x - tw / 2, bottom + 16), label, fill=(17, 24, 39), font=axis_font)

        pw, ph = text_size(draw, block.collect_pattern, small_font)
        draw.text((center_x - pw / 2, bottom + 48), block.collect_pattern, fill=(75, 85, 99), font=small_font)

    if blocks[-1].final_eval_rate is not None:
        center_x = left + group_w * (len(blocks) - 0.5)
        fx = center_x + bar_w + 84
        fy = y_for(blocks[-1].final_eval_rate)
        radius = 14
        draw.ellipse((fx - radius, fy - radius, fx + radius, fy + radius), fill=final_color, outline=final_color)
        draw.line((fx - radius - 10, fy, fx + radius + 10, fy), fill=final_color, width=3)
        draw.line((fx, fy - radius - 10, fx, fy + radius + 10), fill=final_color, width=3)
        draw.text((fx + 22, fy - 12), fmt_pct(blocks[-1].final_eval_rate), fill=final_color, font=label_font)

    draw.text((70, 980), "Collect pattern is shown below each block label. Block 5 is the final winning block.", fill=(107, 114, 128), font=small_font)
    image.save(out_path)


def render_collect_eval_timeline(blocks: Sequence[BlockSummary], out_path: Path) -> None:
    labels: List[str] = []
    collect_rates: List[float] = []
    eval_rates: List[float] = []
    colors: List[str] = []
    palette = [
        (37, 99, 235),
        (15, 118, 110),
        (124, 58, 237),
        (180, 83, 9),
        (190, 18, 60),
    ]
    for block in blocks:
        for it in block.iterations:
            labels.append(f"B{block.block_idx}-I{it.iteration}")
            collect_rates.append(it.collect_rate)
            eval_rates.append(it.eval_rate)
            colors.append(palette[(block.block_idx - 1) % len(palette)])

    width, height = 2400, 1100
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)

    title_font = find_font(42, bold=True)
    subtitle_font = find_font(22)
    axis_font = find_font(20)
    label_font = find_font(16)
    small_font = find_font(15)

    draw.text((70, 46), "Per-Iteration Collect vs Eval Success Rates", fill=(17, 24, 39), font=title_font)
    draw.text((70, 102), "Collect bars are colored by block. Eval is the black line.", fill=(75, 85, 99), font=subtitle_font)

    plot = (140, 220, 2260, 860)
    left, top, right, bottom = plot
    y_max = 0.55
    for tick_idx in range(6):
        rate = y_max * tick_idx / 5.0
        y = bottom - (bottom - top) * (rate / y_max)
        draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
        label = f"{rate:.1f}"
        tw, th = text_size(draw, label, axis_font)
        draw.text((left - tw - 16, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
    draw.line((left, top, left, bottom), fill=(55, 65, 81), width=3)
    draw.line((left, bottom, right, bottom), fill=(55, 65, 81), width=3)

    def y_for(rate: float) -> float:
        return bottom - (bottom - top) * (rate / y_max)

    slot_w = (right - left) / max(1, len(labels))
    bar_w = slot_w * 0.56
    points: List[Tuple[float, float]] = []

    for idx, label in enumerate(labels):
        center_x = left + slot_w * (idx + 0.5)
        cx0 = center_x - bar_w / 2
        cx1 = center_x + bar_w / 2
        cy = y_for(collect_rates[idx])
        draw.rectangle((cx0, cy, cx1, bottom), fill=colors[idx], outline=blend(colors[idx], (0, 0, 0), 0.2))
        draw.text((cx0, cy - 24), fmt_pct(collect_rates[idx]), fill=(55, 65, 81), font=small_font)

        px = center_x
        py = y_for(eval_rates[idx])
        points.append((px, py))
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(17, 24, 39))
        draw.text((px - 18, py - 30), fmt_pct(eval_rates[idx]), fill=(17, 24, 39), font=small_font)

        tw, th = text_size(draw, label, label_font)
        draw.text((center_x - tw / 2, bottom + 18), label, fill=(17, 24, 39), font=label_font)

    if len(points) >= 2:
        draw.line(points, fill=(17, 24, 39), width=4)

    legend_x = left + 10
    legend_y = top - 84
    for block_idx, color in enumerate(palette[: len(blocks)]):
        x0 = legend_x + block_idx * 170
        draw.rectangle((x0, legend_y, x0 + 22, legend_y + 22), fill=color)
        draw.text((x0 + 32, legend_y - 1), f"B{block_idx + 1}", fill=(31, 41, 55), font=label_font)
    lx = legend_x + 170 * len(blocks) + 20
    draw.rectangle((lx, legend_y, lx + 22, legend_y + 22), fill=(156, 163, 175))
    draw.text((lx + 32, legend_y - 1), "Collect bars", fill=(31, 41, 55), font=label_font)
    draw.line((lx + 210, legend_y + 11, lx + 250, legend_y + 11), fill=(17, 24, 39), width=4)
    draw.ellipse((lx + 228, legend_y + 4, lx + 242, legend_y + 18), fill=(17, 24, 39))
    draw.text((lx + 260, legend_y - 1), "Eval line", fill=(31, 41, 55), font=label_font)

    image.save(out_path)


def render_update_metrics(blocks: Sequence[BlockSummary], out_path: Path) -> None:
    labels: List[str] = []
    approx_kls: List[float] = []
    clip_fracs: List[float] = []
    value_losses: List[float] = []
    for block in blocks:
        for it in block.iterations:
            labels.append(f"B{block.block_idx}-I{it.iteration}")
            approx_kls.append(it.approx_kl)
            clip_fracs.append(it.clip_fraction)
            value_losses.append(it.value_loss)

    width, height = 2400, 1240
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)

    title_font = find_font(42, bold=True)
    subtitle_font = find_font(22)
    axis_font = find_font(19)
    label_font = find_font(16)
    small_font = find_font(14)

    draw.text((70, 46), "Update Metrics Across the Block Chain", fill=(17, 24, 39), font=title_font)
    draw.text((70, 102), "Top: approx KL and clip fraction. Bottom: value loss.", fill=(75, 85, 99), font=subtitle_font)

    upper = (140, 220, 2260, 610)
    lower = (140, 720, 2260, 1080)

    def draw_series_panel(
        rect: Tuple[int, int, int, int],
        series_a: Sequence[float],
        series_b: Optional[Sequence[float]],
        y_max: float,
        title: str,
        color_a: Tuple[int, int, int],
        color_b: Optional[Tuple[int, int, int]],
        bars: bool = False,
    ) -> None:
        left, top, right, bottom = rect
        for tick_idx in range(6):
            rate = y_max * tick_idx / 5.0
            y = bottom - (bottom - top) * (rate / y_max if y_max > 0 else 0.0)
            draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
            label = f"{rate:.3f}" if y_max < 0.1 else f"{rate:.1f}"
            tw, th = text_size(draw, label, axis_font)
            draw.text((left - tw - 16, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
        draw.line((left, top, left, bottom), fill=(55, 65, 81), width=3)
        draw.line((left, bottom, right, bottom), fill=(55, 65, 81), width=3)
        draw.text((left, top - 40), title, fill=(17, 24, 39), font=axis_font)

        def y_for(value: float) -> float:
            return bottom - (bottom - top) * (value / y_max if y_max > 0 else 0.0)

        slot_w = (right - left) / max(1, len(labels))
        if bars:
            for idx, value in enumerate(series_a):
                center_x = left + slot_w * (idx + 0.5)
                bar_w = slot_w * 0.58
                x0 = center_x - bar_w / 2
                x1 = center_x + bar_w / 2
                y0 = y_for(value)
                draw.rectangle((x0, y0, x1, bottom), fill=color_a)
                draw.text((x0, y0 - 20), f"{value:.1f}", fill=color_a, font=small_font)
                tw, th = text_size(draw, labels[idx], label_font)
                draw.text((center_x - tw / 2, bottom + 14), labels[idx], fill=(17, 24, 39), font=label_font)
            return

        points_a: List[Tuple[float, float]] = []
        points_b: List[Tuple[float, float]] = []
        for idx, value in enumerate(series_a):
            center_x = left + slot_w * (idx + 0.5)
            y0 = y_for(value)
            points_a.append((center_x, y0))
            draw.ellipse((center_x - 6, y0 - 6, center_x + 6, y0 + 6), fill=color_a)
            draw.text((center_x - 18, y0 - 24), f"{value:.4f}", fill=color_a, font=small_font)
            tw, th = text_size(draw, labels[idx], label_font)
            draw.text((center_x - tw / 2, bottom + 14), labels[idx], fill=(17, 24, 39), font=label_font)
        if len(points_a) >= 2:
            draw.line(points_a, fill=color_a, width=4)

        if series_b is not None and color_b is not None:
            for idx, value in enumerate(series_b):
                center_x = left + slot_w * (idx + 0.5)
                y0 = y_for(value)
                points_b.append((center_x, y0))
                draw.rectangle((center_x - 6, y0 - 6, center_x + 6, y0 + 6), fill=color_b)
                draw.text((center_x + 8, y0 - 20), f"{value:.4f}", fill=color_b, font=small_font)
            if len(points_b) >= 2:
                draw.line(points_b, fill=color_b, width=4)

    top_ymax = max(max(approx_kls), max(clip_fracs)) * 1.35
    draw_series_panel(upper, approx_kls, clip_fracs, top_ymax, "Approx KL (blue) and Clip Fraction (red)", (29, 78, 216), (220, 38, 38))
    draw_series_panel(lower, value_losses, None, max(value_losses) * 1.25, "Value Loss", (156, 163, 175), None, bars=True)
    image.save(out_path)


def render_table_image(
    title: str,
    subtitle: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cell_text: Sequence[Sequence[str]],
    cell_colors: Sequence[Sequence[Tuple[int, int, int]]],
    out_path: Path,
    row_label_width: int = 280,
    col_width: int = 140,
    cell_height: int = 58,
) -> None:
    title_font = find_font(34, bold=True)
    subtitle_font = find_font(18)
    header_font = find_font(18, bold=True)
    body_font = find_font(16)

    margin = 36
    title_h = 54
    subtitle_h = 42 if subtitle else 0
    width = margin * 2 + row_label_width + col_width * len(col_labels)
    height = margin * 2 + title_h + subtitle_h + cell_height * (len(row_labels) + 1) + 16

    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)

    draw.text((margin, margin), title, fill=(17, 24, 39), font=title_font)
    if subtitle:
        draw.text((margin, margin + title_h), subtitle, fill=(75, 85, 99), font=subtitle_font)

    table_top = margin + title_h + subtitle_h + 14
    table_left = margin

    # Header row
    draw.rectangle(
        [table_left, table_top, table_left + row_label_width, table_top + cell_height],
        fill=(229, 231, 235),
        outline=(156, 163, 175),
        width=1,
    )
    draw.text((table_left + 12, table_top + 16), "Pattern / Phase", fill=(17, 24, 39), font=header_font)

    for col_idx, col_label in enumerate(col_labels):
        x0 = table_left + row_label_width + col_idx * col_width
        x1 = x0 + col_width
        draw.rectangle([x0, table_top, x1, table_top + cell_height], fill=(229, 231, 235), outline=(156, 163, 175), width=1)
        w, h = text_size(draw, col_label, header_font)
        draw.text((x0 + (col_width - w) / 2, table_top + (cell_height - h) / 2), col_label, fill=(17, 24, 39), font=header_font)

    # Body
    for row_idx, row_label in enumerate(row_labels):
        y0 = table_top + cell_height * (row_idx + 1)
        y1 = y0 + cell_height
        label_fill = (249, 250, 251) if row_idx % 2 == 0 else (243, 244, 246)
        draw.rectangle([table_left, y0, table_left + row_label_width, y1], fill=label_fill, outline=(209, 213, 219), width=1)
        draw.text((table_left + 12, y0 + 12), row_label, fill=(17, 24, 39), font=body_font)

        for col_idx, value in enumerate(cell_text[row_idx]):
            x0 = table_left + row_label_width + col_idx * col_width
            x1 = x0 + col_width
            draw.rectangle([x0, y0, x1, y1], fill=cell_colors[row_idx][col_idx], outline=(209, 213, 219), width=1)
            lines = value.split("\n")
            total_h = 0
            dims = []
            for line in lines:
                w, h = text_size(draw, line, body_font)
                dims.append((w, h))
                total_h += h
            total_h += max(0, len(lines) - 1) * 2
            current_y = y0 + (cell_height - total_h) / 2
            for line, (w, h) in zip(lines, dims):
                draw.text((x0 + (col_width - w) / 2, current_y), line, fill=(17, 24, 39), font=body_font)
                current_y += h + 2

    image.save(out_path)


def render_block_summary_table(blocks: Sequence[BlockSummary], out_path: Path) -> None:
    row_labels = [f"Block {block.block_idx}" for block in blocks]
    col_labels = ["Collect", "Baseline", "Iter1 C", "Iter1 E", "Iter2 C", "Iter2 E", "Best", "Final"]
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[int, int, int]]] = []

    for block in blocks:
        row = [
            block.collect_pattern,
            fmt_pct(block.baseline_rate),
            f"{block.iterations[0].collect_successes}/{block.iterations[0].collect_episodes}\n{fmt_pct(block.iterations[0].collect_rate)}",
            fmt_pct(block.iterations[0].eval_rate),
            f"{block.iterations[1].collect_successes}/{block.iterations[1].collect_episodes}\n{fmt_pct(block.iterations[1].collect_rate)}",
            fmt_pct(block.iterations[1].eval_rate),
            fmt_pct(block.best_eval_rate),
            fmt_pct(block.final_eval_rate) if block.final_eval_rate is not None else "-",
        ]
        colors = [
            (229, 231, 235),
            (224, 242, 254),
            (255, 247, 237),
            (239, 246, 255),
            (255, 247, 237),
            (239, 246, 255),
            (219, 234, 254),
            (254, 226, 226) if block.final_eval_rate is not None else (243, 244, 246),
        ]
        cell_text.append(row)
        cell_colors.append(colors)

    render_table_image(
        title="Block-Level Summary",
        subtitle="Collect patterns, collect success, eval success, and final eval outcomes.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=160,
        col_width=125,
        cell_height=58,
    )


def render_eval_bank_best_matrix(blocks: Sequence[BlockSummary], patterns: Sequence[Tuple[int, str, str]], out_path: Path) -> None:
    col_labels: List[str] = []
    phase_results: List[Dict[int, PhaseEvalCell]] = []
    for block in blocks:
        col_labels.extend([f"B{block.block_idx}\nbase", f"B{block.block_idx}\nbest"])
        phase_results.append(block.baseline_results)
        best_iter = next(it for it in block.iterations if it.iteration == block.best_iteration)
        phase_results.append(best_iter.eval_results)

    row_labels = [f"{idx:02d} {pattern}" for idx, _, pattern in patterns]
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[int, int, int]]] = []
    for idx, _, _pattern in patterns:
        row_values: List[str] = []
        row_colors: List[Tuple[int, int, int]] = []
        for results in phase_results:
            cell = results.get(idx)
            if cell is None:
                row_values.append("NA")
                row_colors.append((243, 244, 246))
            elif cell.success:
                row_values.append(f"S\n{cell.steps}")
                row_colors.append((220, 252, 231))
            else:
                row_values.append(f"F\n{cell.steps}")
                row_colors.append((254, 226, 226))
        cell_text.append(row_values)
        cell_colors.append(row_colors)

    render_table_image(
        title="Eval Bank Success Matrix: Baseline vs Best-Per-Block",
        subtitle="Green = success, red = failure. Lower line is episode steps.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=210,
        col_width=96,
        cell_height=54,
    )


def render_eval_bank_full_matrix(blocks: Sequence[BlockSummary], patterns: Sequence[Tuple[int, str, str]], out_path: Path) -> None:
    col_labels: List[str] = []
    phase_results: List[Dict[int, PhaseEvalCell]] = []
    for block in blocks:
        col_labels.append(f"B{block.block_idx}\nbase")
        phase_results.append(block.baseline_results)
        for iteration in block.iterations:
            col_labels.append(f"B{block.block_idx}\nI{iteration.iteration}")
            phase_results.append(iteration.eval_results)

    row_labels = [f"{idx:02d} {pattern}" for idx, _, pattern in patterns]
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[int, int, int]]] = []
    for idx, _, _pattern in patterns:
        row_values: List[str] = []
        row_colors: List[Tuple[int, int, int]] = []
        for results in phase_results:
            cell = results.get(idx)
            if cell is None:
                row_values.append("NA")
                row_colors.append((243, 244, 246))
            elif cell.success:
                row_values.append(f"S\n{cell.steps}")
                row_colors.append((220, 252, 231))
            else:
                row_values.append(f"F\n{cell.steps}")
                row_colors.append((254, 226, 226))
        cell_text.append(row_values)
        cell_colors.append(row_colors)

    render_table_image(
        title="Eval Bank Success Matrix: All Baseline and Iteration Phases",
        subtitle="This matrix shows which fixed-bank patterns flipped as the chain moved block to block.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=210,
        col_width=78,
        cell_height=54,
    )


def render_final_eval_matrix(blocks: Sequence[BlockSummary], patterns: Sequence[Tuple[int, str, str]], out_path: Path) -> None:
    final_block = blocks[-1]
    row_labels = [f"{idx:02d} {pattern}" for idx, _, pattern in patterns]
    col_labels = ["Final"]
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[int, int, int]]] = []
    for idx, _, _pattern in patterns:
        cell = final_block.final_eval_results.get(idx)
        if cell is None:
            cell_text.append(["NA"])
            cell_colors.append([(243, 244, 246)])
        elif cell.success:
            cell_text.append([[f"S\n{cell.steps}"][0]])
            cell_colors.append([(220, 252, 231)])
        else:
            cell_text.append([[f"F\n{cell.steps}"][0]])
            cell_colors.append([(254, 226, 226)])

    render_table_image(
        title="Final Eval Bank Success Matrix",
        subtitle="Final eval uses the final bank and azalea leaves material. Green = success, red = failure.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=210,
        col_width=120,
        cell_height=54,
    )


def render_summary_slide(blocks: Sequence[BlockSummary], chart_paths: Dict[str, Path], out_path: Path) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(image)

    title_font = find_font(44, bold=True)
    subtitle_font = find_font(22)
    body_font = find_font(24)
    small_font = find_font(18)

    draw.text((64, 48), "VM1 O2 Generalization Chain Report", fill=(17, 24, 39), font=title_font)
    draw.text(
        (64, 104),
        "Imported run: ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330_progress_vm1",
        fill=(75, 85, 99),
        font=subtitle_font,
    )

    best_block = blocks[-1]
    bullets = [
        f"Best eval checkpoint: Block 5 Iter 2 = {fmt_pct(best_block.best_eval_rate)} on eval_bank.",
        f"Final eval result: {fmt_pct(best_block.final_eval_rate or 0.0)} on final_eval_bank.",
        "Strong collect patterns: GLG/LGL and LGL/GGL. Harmful collect pattern: LLG/GGL.",
        "Collect success is not a sufficient proxy: Block 3 eval improved despite weaker collect, while Block 4 collect rose as eval fell.",
    ]
    y = 154
    for bullet in bullets:
        draw.text((76, y), f"- {bullet}", fill=(31, 41, 55), font=body_font)
        y += 40

    panel_specs = [
        ("Block Summary", chart_paths["block_summary"], (64, 320), (840, 330)),
        ("Eval Timeline", chart_paths["timeline"], (980, 220), (860, 310)),
        ("Best-Phase Matrix", chart_paths["best_matrix"], (980, 560), (860, 440)),
    ]

    for title, path, xy, size in panel_specs:
        px, py = xy
        pw, ph = size
        draw.rounded_rectangle([px, py, px + pw, py + ph], radius=18, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
        draw.text((px + 18, py + 14), title, fill=(17, 24, 39), font=subtitle_font)
        panel = Image.open(path).convert("RGB")
        panel.thumbnail((pw - 32, ph - 54))
        panel_x = px + (pw - panel.width) // 2
        panel_y = py + 44 + (ph - 54 - panel.height) // 2
        image.paste(panel, (panel_x, panel_y))

    footer = "Generated PNG set includes charts, tables, and this slide. See analysis.md for interpretation and file inventory."
    draw.text((64, 1028), footer, fill=(107, 114, 128), font=small_font)
    image.save(out_path)


def write_markdown(blocks: Sequence[BlockSummary], out_dir: Path, asset_paths: Dict[str, Path]) -> None:
    best_block = blocks[-1]
    lines = [
        "# VM1 O2 Generalization Run Report",
        "",
        "## Scope",
        "",
        "- Imported run root: `ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330_progress_vm1`",
        "- Source: local imported metrics and referenced eval/final-eval bank metadata.",
        "",
        "## Key Results",
        "",
        f"- Best eval checkpoint: `block_005 / iter_002` with `{fmt_pct(best_block.best_eval_rate)}` on `eval_bank`.",
        f"- Final eval result: `{fmt_pct(best_block.final_eval_rate or 0.0)}` on `final_eval_bank`.",
        "- Strong collect patterns: `GLG/LGL` and `LGL/GGL`.",
        "- Weak collect pattern: `LLG/GGL`.",
        "- Collect success was not a sufficient proxy for generalization: `block_003` improved on eval while collect weakened, and `block_004` lost eval performance even as collect rose.",
        "",
        "## Block Summary",
        "",
        "| Block | Collect Pattern | Baseline Eval | Iter1 Collect | Iter1 Eval | Iter2 Collect | Iter2 Eval | Best Eval | Final Eval |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for block in blocks:
        lines.append(
            "| "
            + f"{block.block_idx} | {block.collect_pattern} | {fmt_pct(block.baseline_rate)} | "
            + f"{block.iterations[0].collect_successes}/{block.iterations[0].collect_episodes} ({fmt_pct(block.iterations[0].collect_rate)}) | "
            + f"{fmt_pct(block.iterations[0].eval_rate)} | "
            + f"{block.iterations[1].collect_successes}/{block.iterations[1].collect_episodes} ({fmt_pct(block.iterations[1].collect_rate)}) | "
            + f"{fmt_pct(block.iterations[1].eval_rate)} | "
            + f"{fmt_pct(block.best_eval_rate)} | "
            + (fmt_pct(block.final_eval_rate) if block.final_eval_rate is not None else "-")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `block_001` started from an all-glass collect world and improved cleanly, but did not by itself explain the final best result.",
            "- `block_002` was the first strong transfer block. It had both high collect success and higher eval success.",
            "- `block_003` is the counterexample that proves collect is not enough: collect dropped to `7/64` while eval still reached `31.25%`.",
            "- `block_004` was a harmful block. Collect rose to `17/64`, but eval fell from a `37.5%` baseline to `18.75%`.",
            "- `block_005` was the decisive recovery block. It lifted eval from `25.0%` baseline to `43.75%` and still held `37.5%` on the harder `final_eval_bank`.",
            "",
            "## Generated Assets",
            "",
        ]
    )

    for key, path in asset_paths.items():
        lines.append(f"- `{path.name}`: {key}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `eval_bank` uses the fixed O2 eval bank.",
            "- `final_eval_bank` is a separate fixed bank and uses azalea leaves material.",
            "- All visual assets were exported as standalone PNG files for direct slide use.",
        ]
    )

    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (import_root / "analysis_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = discover_blocks(import_root)
    if not blocks:
        raise SystemExit(f"No blocks found under {import_root}")

    eval_bank_patterns = extract_eval_bank_patterns(blocks)
    final_eval_patterns = extract_final_eval_patterns(blocks)

    asset_paths = {
        "summary_slide": out_dir / "00_summary_slide.png",
        "block_eval_summary": out_dir / "01_block_eval_summary.png",
        "collect_eval_timeline": out_dir / "02_collect_eval_timeline.png",
        "update_metrics": out_dir / "03_update_metrics.png",
        "block_summary": out_dir / "04_block_summary_table.png",
        "eval_best_matrix": out_dir / "05_eval_bank_best_matrix.png",
        "eval_full_matrix": out_dir / "06_eval_bank_full_matrix.png",
        "final_eval_matrix": out_dir / "07_final_eval_bank_matrix.png",
    }

    render_block_eval_summary(blocks, asset_paths["block_eval_summary"])
    render_collect_eval_timeline(blocks, asset_paths["collect_eval_timeline"])
    render_update_metrics(blocks, asset_paths["update_metrics"])
    render_block_summary_table(blocks, asset_paths["block_summary"])
    render_eval_bank_best_matrix(blocks, eval_bank_patterns, asset_paths["eval_best_matrix"])
    render_eval_bank_full_matrix(blocks, eval_bank_patterns, asset_paths["eval_full_matrix"])
    render_final_eval_matrix(blocks, final_eval_patterns, asset_paths["final_eval_matrix"])
    render_summary_slide(
        blocks,
        {
            "block_summary": asset_paths["block_summary"],
            "timeline": asset_paths["collect_eval_timeline"],
            "best_matrix": asset_paths["eval_best_matrix"],
        },
        asset_paths["summary_slide"],
    )
    write_markdown(blocks, out_dir, asset_paths)

    manifest = {key: str(path) for key, path in asset_paths.items()}
    manifest["analysis_md"] = str(out_dir / "analysis.md")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
