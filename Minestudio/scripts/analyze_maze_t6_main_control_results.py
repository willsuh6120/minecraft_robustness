#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


CHECKPOINT_SPECS = [
    ("baseline", "baseline/full16_x4/split_report.json", "Baseline x4"),
    ("iter_004", "evals/iter_004_full16_x2/split_report.json", "Iter 4 x2"),
    ("iter_008", "evals/iter_008_full16_x4/split_report.json", "Iter 8 x4"),
    ("iter_012", "evals/iter_012_full16_x2/split_report.json", "Iter 12 x2"),
    ("iter_016", "evals/iter_016_full16_x4/split_report.json", "Iter 16 x4"),
    ("final", "final_probe/full16_x16/split_report.json", "Final x16"),
]
GROUP_ORDER = ["train8", "heldout8", "full16"]
GROUP_LABELS = {"train8": "Train 8", "heldout8": "Held-out 8", "full16": "Full 16"}

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRID = (220, 220, 220)
GRAY = (154, 160, 166)
GREEN = (27, 158, 119)
ORANGE = (217, 95, 2)
BLUE = (66, 133, 244)
RED = (219, 68, 55)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.1f}%"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z * z) / (4 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


def weighted_mean(rows: Iterable[Tuple[float, int]]) -> float | None:
    numer = 0.0
    denom = 0
    for value, weight in rows:
        if value is None or weight <= 0:
            continue
        numer += float(value) * int(weight)
        denom += int(weight)
    if denom <= 0:
        return None
    return numer / denom


def font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


FONT_SM = font(12)
FONT_MD = font(14)
FONT_LG = font(18)
FONT_XL = font(22)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def new_canvas(width: int = 1200, height: int = 700) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), WHITE)
    return img, ImageDraw.Draw(img)


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None, width: int = 1200) -> int:
    y = 18
    tw, th = text_size(draw, title, FONT_XL)
    draw.text(((width - tw) / 2, y), title, fill=BLACK, font=FONT_XL)
    y += th + 6
    if subtitle:
        sw, sh = text_size(draw, subtitle, FONT_MD)
        draw.text(((width - sw) / 2, y), subtitle, fill=(90, 90, 90), font=FONT_MD)
        y += sh + 10
    return y


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_checkpoints(root: Path, experiment_dir: str | None) -> List[dict]:
    out: List[dict] = []
    for ckpt_key, rel_path, display in CHECKPOINT_SPECS:
        if ckpt_key == "baseline":
            path = root / rel_path
        else:
            if experiment_dir is None:
                continue
            path = root / experiment_dir / rel_path
        if not path.is_file():
            continue
        payload = load_json(path)
        out.append(
            {
                "checkpoint": ckpt_key,
                "display": display,
                "path": str(path),
                "label": payload.get("label", display),
                "groups": payload["groups"],
                "per_variant": payload.get("per_variant", []),
                "baseline_groups": payload.get("baseline_groups", {}),
                "deltas": payload.get("deltas", {}),
                "train_variants": payload.get("train_variants", []),
            }
        )
    return out


def read_collect_summaries(training_root: Path) -> List[dict]:
    rows: List[dict] = []
    for iter_dir in sorted(training_root.glob("iter_*")):
        summaries = sorted(iter_dir.glob("*/pilot_summary.json"))
        if not summaries:
            continue
        payload = load_json(summaries[0])
        iteration = (payload.get("iteration_summaries") or [{}])[0]
        collect_stats = iteration.get("collect_episode_stats", {}) or {}
        collect_world = iteration.get("collect_world", {}) or {}
        split_label = str(collect_world.get("split_label") or "")
        variant = split_label.split("singleton:", 1)[-1] if split_label.startswith("singleton:") else split_label
        rows.append(
            {
                "iter_idx": int(iter_dir.name.split("_")[-1]),
                "variant": variant,
                "successful_episodes": int(collect_stats.get("successful_episodes") or 0),
                "reward_positive_episodes": int(collect_stats.get("reward_positive_episodes") or 0),
                "episodes": int(collect_stats.get("episodes") or 0),
                "success_rate": float(collect_stats.get("success_rate") or 0.0),
                "wall_time_sec": float(iteration.get("iteration_wall_time_sec") or 0.0),
            }
        )
    return rows


def pool_baseline_variants(*per_variant_lists: List[dict]) -> Dict[str, dict]:
    pooled: Dict[str, dict] = {}
    for rows in per_variant_lists:
        for row in rows:
            variant = row["variant_id"]
            item = pooled.setdefault(
                variant,
                {
                    "variant_id": variant,
                    "episodes": 0,
                    "successful_episodes": 0,
                    "instance_idx": row.get("instance_idx"),
                    "mean_steps_terms": [],
                },
            )
            episodes = int(row.get("episodes") or 0)
            item["episodes"] += episodes
            item["successful_episodes"] += int(row.get("successful_episodes") or 0)
            if row.get("mean_steps") is not None:
                item["mean_steps_terms"].append((float(row["mean_steps"]), episodes))
    for item in pooled.values():
        item["success_rate"] = item["successful_episodes"] / item["episodes"] if item["episodes"] else 0.0
        item["mean_steps"] = weighted_mean(item["mean_steps_terms"])
    return pooled


def pool_baseline_groups(*group_maps: dict) -> Dict[str, dict]:
    pooled: Dict[str, dict] = {}
    for group_name in GROUP_ORDER:
        episodes = 0
        successes = 0
        mean_terms: List[Tuple[float, int]] = []
        variants: List[str] = []
        for gm in group_maps:
            g = gm[group_name]
            e = int(g.get("episodes") or 0)
            s = int(g.get("successful_episodes") or 0)
            episodes += e
            successes += s
            variants.extend(list(g.get("variants") or []))
            if g.get("mean_steps") is not None:
                mean_terms.append((float(g["mean_steps"]), e))
        pooled[group_name] = {
            "episodes": episodes,
            "successful_episodes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "mean_steps": weighted_mean(mean_terms),
            "variant_count": len(dict.fromkeys(variants)),
            "variants": list(dict.fromkeys(variants)),
        }
    return pooled


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_axes(draw, left, top, right, bottom, y_ticks: Sequence[float], y_label_fmt="{:.0f}%") -> None:
    draw.line((left, top, left, bottom), fill=BLACK, width=2)
    draw.line((left, bottom, right, bottom), fill=BLACK, width=2)
    for tick in y_ticks:
        y = bottom - tick
        draw.line((left, y, right, y), fill=GRID, width=1)


def line_chart(
    out_path: Path,
    title: str,
    subtitle: str,
    x_labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
    y_max: float = 100.0,
) -> None:
    width, height = 1200, 700
    img, draw = new_canvas(width, height)
    chart_top = draw_title(draw, title, subtitle, width) + 20
    left, right, bottom = 110, width - 50, height - 90
    top = chart_top
    ticks = [0, 25, 50, 75, 100]
    for t in ticks:
        y = bottom - (bottom - top) * (t / y_max)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{t:.0f}%"
        tw, th = text_size(draw, label, FONT_SM)
        draw.text((left - tw - 8, y - th / 2), label, fill=BLACK, font=FONT_SM)
    draw.line((left, top, left, bottom), fill=BLACK, width=2)
    draw.line((left, bottom, right, bottom), fill=BLACK, width=2)

    n = len(x_labels)
    xs = []
    for i, lab in enumerate(x_labels):
        x = left + (right - left) * (i / (n - 1)) if n > 1 else (left + right) / 2
        xs.append(x)
        draw.line((x, bottom, x, bottom + 6), fill=BLACK, width=1)
        tw, th = text_size(draw, lab, FONT_SM)
        draw.text((x - tw / 2, bottom + 12), lab, fill=BLACK, font=FONT_SM)

    for name, ys, color in series:
        pts = []
        for x, yv in zip(xs, ys):
            y = bottom - (bottom - top) * (yv / y_max)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color)

    leg_x = right - 250
    leg_y = top + 10
    for idx, (name, _, color) in enumerate(series):
        y = leg_y + idx * 24
        draw.rectangle((leg_x, y + 4, leg_x + 16, y + 16), fill=color)
        draw.text((leg_x + 24, y), name, fill=BLACK, font=FONT_MD)

    img.save(out_path)


def grouped_bar_chart(
    out_path: Path,
    title: str,
    subtitle: str,
    group_labels: Sequence[str],
    bars: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
    y_max: float = 100.0,
    y_min: float = 0.0,
) -> None:
    width, height = 1200, 700
    img, draw = new_canvas(width, height)
    chart_top = draw_title(draw, title, subtitle, width) + 20
    left, right, bottom = 110, width - 50, height - 90
    top = chart_top
    if y_min < 0:
        ticks = [y_min + (y_max - y_min) * i / 8 for i in range(9)]
    else:
        ticks = [0, 25, 50, 75, 100]
    for t in ticks:
        y = bottom - (bottom - top) * ((t - y_min) / (y_max - y_min))
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{t:.0f}%"
        tw, th = text_size(draw, label, FONT_SM)
        draw.text((left - tw - 8, y - th / 2), label, fill=BLACK, font=FONT_SM)
    draw.line((left, top, left, bottom), fill=BLACK, width=2)
    draw.line((left, bottom, right, bottom), fill=BLACK, width=2)
    if y_min < 0 < y_max:
        zero_y = bottom - (bottom - top) * ((0.0 - y_min) / (y_max - y_min))
        draw.line((left, zero_y, right, zero_y), fill=BLACK, width=2)

    group_width = (right - left) / len(group_labels)
    bar_width = group_width / (len(bars) + 1)
    for gi, g in enumerate(group_labels):
        gx0 = left + gi * group_width
        gx_center = gx0 + group_width / 2
        tw, th = text_size(draw, g, FONT_SM)
        draw.text((gx_center - tw / 2, bottom + 12), g, fill=BLACK, font=FONT_SM)
        for bi, (name, values, color) in enumerate(bars):
            x0 = gx0 + (bi + 0.5) * bar_width
            x1 = x0 + bar_width * 0.8
            value = values[gi]
            zero_y = bottom - (bottom - top) * ((0.0 - y_min) / (y_max - y_min)) if y_min < 0 < y_max else bottom
            val_y = bottom - (bottom - top) * ((value - y_min) / (y_max - y_min))
            y0, y1 = sorted((zero_y, val_y))
            draw.rectangle((x0, y0, x1, y1), fill=color)
            label = f"{value:.1f}"
            tw2, th2 = text_size(draw, label, FONT_SM)
            label_y = y0 - th2 - 2 if value >= 0 else y1 + 2
            draw.text((x0 + (x1 - x0) / 2 - tw2 / 2, label_y), label, fill=BLACK, font=FONT_SM)

    leg_x = right - 260
    leg_y = top + 10
    for idx, (name, _, color) in enumerate(bars):
        y = leg_y + idx * 24
        draw.rectangle((leg_x, y + 4, leg_x + 16, y + 16), fill=color)
        draw.text((leg_x + 24, y), name, fill=BLACK, font=FONT_MD)

    img.save(out_path)


def horizontal_grouped_bar_chart(
    out_path: Path,
    title: str,
    subtitle: str,
    variants: Sequence[str],
    bars: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
    x_min: float,
    x_max: float,
) -> None:
    width = 1500
    row_h = 28
    top_pad = 90
    bottom_pad = 40
    height = top_pad + bottom_pad + row_h * len(variants) + 40
    img, draw = new_canvas(width, height)
    chart_top = draw_title(draw, title, subtitle, width) + 10
    left, right = 280, width - 60
    top = chart_top + 20
    bottom = height - 60
    ticks = [x_min + (x_max - x_min) * i / 8 for i in range(9)]
    zero_x = left + (0 - x_min) / (x_max - x_min) * (right - left) if x_min < 0 < x_max else None
    for t in ticks:
        x = left + (float(t) - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = f"{t:.0f}"
        tw, th = text_size(draw, label, FONT_SM)
        draw.text((x - tw / 2, bottom + 6), label, fill=BLACK, font=FONT_SM)
    if zero_x is not None:
        draw.line((zero_x, top, zero_x, bottom), fill=BLACK, width=2)

    row_gap = row_h
    for i, variant in enumerate(variants):
        cy = top + i * row_gap + row_h / 2
        tw, th = text_size(draw, variant, FONT_SM)
        draw.text((left - tw - 10, cy - th / 2), variant, fill=BLACK, font=FONT_SM)
        n_bars = len(bars)
        for bi, (name, values, color) in enumerate(bars):
            value = values[i]
            bar_top = cy - 8 + (bi - (n_bars - 1) / 2) * 8
            bar_bot = bar_top + 6
            x0 = left + (min(0.0, value) - x_min) / (x_max - x_min) * (right - left)
            x1 = left + (max(0.0, value) - x_min) / (x_max - x_min) * (right - left)
            draw.rectangle((x0, bar_top, x1, bar_bot), fill=color)

    leg_x = right - 330
    leg_y = chart_top - 6
    for idx, (name, _, color) in enumerate(bars):
        y = leg_y + idx * 22
        draw.rectangle((leg_x, y + 4, leg_x + 16, y + 16), fill=color)
        draw.text((leg_x + 24, y), name, fill=BLACK, font=FONT_MD)

    img.save(out_path)


def plot_checkpoint_line(out_path: Path, main_ckpts: List[dict], control_ckpts: List[dict], group_name: str) -> None:
    main_map = {r["checkpoint"]: r for r in main_ckpts}
    ctrl_map = {r["checkpoint"]: r for r in control_ckpts}
    line_chart(
        out_path,
        f"{GROUP_LABELS[group_name]} Success Over Training",
        "Main (waypoint shaping) vs Control (sparse only)",
        [spec[2] for spec in CHECKPOINT_SPECS],
        [
            ("Main (waypoint)", [100.0 * main_map[k]["groups"][group_name]["success_rate"] for k, _, _ in CHECKPOINT_SPECS], GREEN),
            ("Control (sparse)", [100.0 * ctrl_map[k]["groups"][group_name]["success_rate"] for k, _, _ in CHECKPOINT_SPECS], ORANGE),
        ],
    )


def plot_collect_series(out_path: Path, main_rows: List[dict], ctrl_rows: List[dict], key: str, ylabel: str, title: str) -> None:
    line_chart(
        out_path,
        title,
        ylabel,
        [str(i) for i in range(1, 17)],
        [
            ("Main (waypoint)", [100.0 * r[key] if key == "success_rate" else float(r[key]) for r in main_rows], GREEN),
            ("Control (sparse)", [100.0 * r[key] if key == "success_rate" else float(r[key]) for r in ctrl_rows], ORANGE),
        ],
        y_max=100.0 if key == "success_rate" else 64.0,
    )


def plot_final_split_comparison(out_path: Path, pooled_baseline_groups: dict, main_final: dict, ctrl_final: dict) -> None:
    grouped_bar_chart(
        out_path,
        "Final Success vs Pooled Baseline",
        "Each bar uses split-level success rate; baseline pools local+VM 128 episodes",
        [GROUP_LABELS[g] for g in GROUP_ORDER],
        [
            ("Pooled baseline", [100.0 * pooled_baseline_groups[g]["success_rate"] for g in GROUP_ORDER], GRAY),
            ("Main final", [100.0 * main_final["groups"][g]["success_rate"] for g in GROUP_ORDER], GREEN),
            ("Control final", [100.0 * ctrl_final["groups"][g]["success_rate"] for g in GROUP_ORDER], ORANGE),
        ],
    )


def plot_variant_grouped(out_path: Path, pooled_variant: Dict[str, dict], main_final: dict, ctrl_final: dict) -> None:
    main_map = {r["variant_id"]: r for r in main_final["per_variant"]}
    ctrl_map = {r["variant_id"]: r for r in ctrl_final["per_variant"]}
    variants = sorted(pooled_variant.keys(), key=lambda v: (pooled_variant[v]["success_rate"], v))
    horizontal_grouped_bar_chart(
        out_path,
        "Per-Variant Final Success Comparison",
        "Sorted by pooled baseline difficulty",
        variants,
        [
            ("Pooled baseline", [100.0 * pooled_variant[v]["success_rate"] for v in variants], GRAY),
            ("Main final", [100.0 * main_map[v]["success_rate"] for v in variants], GREEN),
            ("Control final", [100.0 * ctrl_map[v]["success_rate"] for v in variants], ORANGE),
        ],
        0.0,
        100.0,
    )


def plot_variant_delta(out_path: Path, pooled_variant: Dict[str, dict], main_final: dict, ctrl_final: dict) -> None:
    main_map = {r["variant_id"]: r for r in main_final["per_variant"]}
    ctrl_map = {r["variant_id"]: r for r in ctrl_final["per_variant"]}
    variants = sorted(
        pooled_variant.keys(),
        key=lambda v: ((main_map[v]["success_rate"] - pooled_variant[v]["success_rate"]) - (ctrl_map[v]["success_rate"] - pooled_variant[v]["success_rate"])),
    )
    horizontal_grouped_bar_chart(
        out_path,
        "Per-Variant Change From Pooled Baseline",
        "Positive values mean better than pooled baseline; negative means regression",
        variants,
        [
            ("Main delta (pp)", [100.0 * (main_map[v]["success_rate"] - pooled_variant[v]["success_rate"]) for v in variants], GREEN),
            ("Control delta (pp)", [100.0 * (ctrl_map[v]["success_rate"] - pooled_variant[v]["success_rate"]) for v in variants], ORANGE),
        ],
        -50.0,
        50.0,
    )


def plot_train_vs_heldout_delta(out_path: Path, pooled_groups: dict, main_final: dict, ctrl_final: dict) -> None:
    grouped_bar_chart(
        out_path,
        "Final Improvement/Regression by Split",
        "Delta relative to pooled baseline",
        [GROUP_LABELS[g] for g in GROUP_ORDER],
        [
            ("Main delta (pp)", [100.0 * (main_final["groups"][g]["success_rate"] - pooled_groups[g]["success_rate"]) for g in GROUP_ORDER], GREEN),
            ("Control delta (pp)", [100.0 * (ctrl_final["groups"][g]["success_rate"] - pooled_groups[g]["success_rate"]) for g in GROUP_ORDER], ORANGE),
        ],
        y_max=25.0,
        y_min=-25.0,
    )


def summarise_key_findings(main_final: dict, ctrl_final: dict, pooled_groups: dict) -> List[str]:
    findings: List[str] = []
    mf = main_final["groups"]["full16"]["success_rate"]
    cf = ctrl_final["groups"]["full16"]["success_rate"]
    pf = pooled_groups["full16"]["success_rate"]
    findings.append(f"Final full16 success was {pct(mf)} for main and {pct(cf)} for control, versus pooled baseline {pct(pf)}.")
    findings.append(
        f"Train8 improved from pooled baseline {pct(pooled_groups['train8']['success_rate'])} to {pct(main_final['groups']['train8']['success_rate'])} in main and {pct(ctrl_final['groups']['train8']['success_rate'])} in control."
    )
    findings.append(
        f"Held-out8 fell from pooled baseline {pct(pooled_groups['heldout8']['success_rate'])} to {pct(main_final['groups']['heldout8']['success_rate'])} in main and {pct(ctrl_final['groups']['heldout8']['success_rate'])} in control."
    )
    if cf > mf:
        findings.append("Control finished slightly above main on aggregate final probe, so waypoint shaping did not beat sparse control in the final result.")
    else:
        findings.append("Main finished above control on aggregate final probe.")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    main_root = Path(args.main_root)
    ctrl_root = Path(args.control_root)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    main_ckpts = read_checkpoints(main_root, "main_waypoint_uniform")
    ctrl_ckpts = read_checkpoints(ctrl_root, "control_sparse_uniform")
    main_collect = read_collect_summaries(main_root / "main_waypoint_uniform" / "training")
    ctrl_collect = read_collect_summaries(ctrl_root / "control_sparse_uniform" / "training")

    main_baseline = next(r for r in main_ckpts if r["checkpoint"] == "baseline")
    ctrl_baseline = next(r for r in ctrl_ckpts if r["checkpoint"] == "baseline")
    main_final = next(r for r in main_ckpts if r["checkpoint"] == "final")
    ctrl_final = next(r for r in ctrl_ckpts if r["checkpoint"] == "final")

    pooled_baseline_groups = pool_baseline_groups(main_baseline["groups"], ctrl_baseline["groups"])
    pooled_baseline_variants = pool_baseline_variants(main_baseline["per_variant"], ctrl_baseline["per_variant"])

    checkpoint_rows: List[dict] = []
    for arm_name, rows in [("main", main_ckpts), ("control", ctrl_ckpts)]:
        for row in rows:
            for group_name in GROUP_ORDER:
                g = row["groups"][group_name]
                checkpoint_rows.append(
                    {
                        "arm": arm_name,
                        "checkpoint": row["checkpoint"],
                        "display": row["display"],
                        "group": group_name,
                        "success_rate": g["success_rate"],
                        "successful_episodes": g["successful_episodes"],
                        "episodes": g["episodes"],
                        "mean_steps": g["mean_steps"],
                    }
                )
    write_csv(
        data_dir / "checkpoint_groups.csv",
        checkpoint_rows,
        ["arm", "checkpoint", "display", "group", "success_rate", "successful_episodes", "episodes", "mean_steps"],
    )
    write_csv(
        data_dir / "collect_main.csv",
        main_collect,
        ["iter_idx", "variant", "successful_episodes", "reward_positive_episodes", "episodes", "success_rate", "wall_time_sec"],
    )
    write_csv(
        data_dir / "collect_control.csv",
        ctrl_collect,
        ["iter_idx", "variant", "successful_episodes", "reward_positive_episodes", "episodes", "success_rate", "wall_time_sec"],
    )

    per_variant_rows: List[dict] = []
    main_final_map = {r["variant_id"]: r for r in main_final["per_variant"]}
    ctrl_final_map = {r["variant_id"]: r for r in ctrl_final["per_variant"]}
    train_variants = set(main_final["train_variants"])
    for variant, pooled in sorted(pooled_baseline_variants.items()):
        main_row = main_final_map[variant]
        ctrl_row = ctrl_final_map[variant]
        per_variant_rows.append(
            {
                "variant_id": variant,
                "split": "train8" if variant in train_variants else "heldout8",
                "baseline_pooled_success_rate": pooled["success_rate"],
                "main_final_success_rate": main_row["success_rate"],
                "control_final_success_rate": ctrl_row["success_rate"],
                "main_delta_pp": 100.0 * (main_row["success_rate"] - pooled["success_rate"]),
                "control_delta_pp": 100.0 * (ctrl_row["success_rate"] - pooled["success_rate"]),
            }
        )
    write_csv(
        data_dir / "per_variant_final_comparison.csv",
        per_variant_rows,
        [
            "variant_id",
            "split",
            "baseline_pooled_success_rate",
            "main_final_success_rate",
            "control_final_success_rate",
            "main_delta_pp",
            "control_delta_pp",
        ],
    )

    plot_checkpoint_line(fig_dir / "01_checkpoint_train8.png", main_ckpts, ctrl_ckpts, "train8")
    plot_checkpoint_line(fig_dir / "02_checkpoint_heldout8.png", main_ckpts, ctrl_ckpts, "heldout8")
    plot_checkpoint_line(fig_dir / "03_checkpoint_full16.png", main_ckpts, ctrl_ckpts, "full16")
    plot_collect_series(fig_dir / "04_collect_success_by_iter.png", main_collect, ctrl_collect, "success_rate", "Collect success", "Collect Success By Iteration")
    plot_collect_series(fig_dir / "05_collect_reward_positive_by_iter.png", main_collect, ctrl_collect, "reward_positive_episodes", "Reward-positive / 64", "Reward-Positive Episodes By Iteration")
    plot_final_split_comparison(fig_dir / "06_final_split_comparison.png", pooled_baseline_groups, main_final, ctrl_final)
    plot_variant_grouped(fig_dir / "07_per_variant_final_comparison.png", pooled_baseline_variants, main_final, ctrl_final)
    plot_variant_delta(fig_dir / "08_per_variant_delta_from_pooled_baseline.png", pooled_baseline_variants, main_final, ctrl_final)
    plot_train_vs_heldout_delta(fig_dir / "09_final_delta_by_split.png", pooled_baseline_groups, main_final, ctrl_final)

    key_findings = summarise_key_findings(main_final, ctrl_final, pooled_baseline_groups)
    summary = {
        "main_root": str(main_root),
        "control_root": str(ctrl_root),
        "pooled_baseline_groups": pooled_baseline_groups,
        "main_final_groups": main_final["groups"],
        "control_final_groups": ctrl_final["groups"],
        "key_findings": key_findings,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Maze T6 Main vs Control Analysis",
        "",
        f"- Main source: `{main_root}`",
        f"- Control source: `{ctrl_root}`",
        "",
        "## Headline",
        "",
    ]
    for item in key_findings:
        report_lines.append(f"- {item}")
    report_lines += [
        "",
        "## Final Split Comparison",
        "",
        "| metric | pooled baseline | main final | control final |",
        "|---|---:|---:|---:|",
    ]
    for group_name in GROUP_ORDER:
        report_lines.append(
            f"| {GROUP_LABELS[group_name]} | {pct(pooled_baseline_groups[group_name]['success_rate'])} | "
            f"{pct(main_final['groups'][group_name]['success_rate'])} | {pct(ctrl_final['groups'][group_name]['success_rate'])} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- Main had an earlier mid-training bump, but that did not survive to final probe.",
        "- Control finished slightly above main on final full16 and held-out8.",
        "- Both arms improved the harder train split relative to pooled baseline, but both lost substantial success on held-out mirror worlds.",
        "- On aggregate full16, neither arm produced a convincing improvement over the pooled baseline.",
        "",
        "## Figures",
        "",
        "1. `figures/01_checkpoint_train8.png`",
        "2. `figures/02_checkpoint_heldout8.png`",
        "3. `figures/03_checkpoint_full16.png`",
        "4. `figures/04_collect_success_by_iter.png`",
        "5. `figures/05_collect_reward_positive_by_iter.png`",
        "6. `figures/06_final_split_comparison.png`",
        "7. `figures/07_per_variant_final_comparison.png`",
        "8. `figures/08_per_variant_delta_from_pooled_baseline.png`",
        "9. `figures/09_final_delta_by_split.png`",
        "",
        "## Professor-Facing Summary",
        "",
        "1. The revised setup successfully lifted the experiment off the previous floor: both arms now produce non-zero baseline success on the 16-world bank.",
        "2. Waypoint shaping changed the training signal strongly, but it did not produce a better final probe than sparse control.",
        "3. The dominant pattern is specialization: train-world gains come with held-out mirror regressions.",
        "4. The next intervention should target stability/generalization, not just denser reward. Mixed-world replay or more revisits per hard world are better candidates than further shaping alone.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
