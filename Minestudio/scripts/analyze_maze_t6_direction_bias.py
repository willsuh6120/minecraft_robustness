#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


TRAIN_VARIANTS = [
    "left_lane_z1_t6",
    "right_lane_z2_t6",
    "left_chicane_t6",
    "right_s_curve_t6",
    "left_funnel_t6",
    "right_gate_entrance_t6",
    "left_outer_detour_t6",
    "right_narrow_door_t6",
]

HELDOUT_VARIANTS = [
    "right_lane_z1_t6",
    "left_lane_z2_t6",
    "right_chicane_t6",
    "left_s_curve_t6",
    "right_funnel_t6",
    "left_gate_entrance_t6",
    "right_outer_detour_t6",
    "left_narrow_door_t6",
]

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRID = (225, 225, 225)
BLUE = (66, 133, 244)
RED = (219, 68, 55)
GRAY = (120, 120, 120)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_geometry(variant_id: str) -> str:
    core = variant_id.removeprefix("left_").removeprefix("right_")
    return core.removesuffix("_t6")


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_rows(payload: dict) -> List[dict]:
    per_variant = {row["variant_id"]: row for row in payload["per_variant"]}
    rows = []
    for trained_variant, mirror_variant in zip(TRAIN_VARIANTS, HELDOUT_VARIANTS):
        train_rate = float(per_variant[trained_variant]["success_rate"])
        mirror_rate = float(per_variant[mirror_variant]["success_rate"])
        rows.append(
            {
                "geometry": short_geometry(trained_variant),
                "trained_variant": trained_variant,
                "mirror_variant": mirror_variant,
                "trained_success_rate": train_rate,
                "mirror_success_rate": mirror_rate,
                "mirror_minus_trained": mirror_rate - train_rate,
            }
        )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "geometry",
                "trained_variant",
                "mirror_variant",
                "trained_success_rate",
                "mirror_success_rate",
                "mirror_minus_trained",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_gap_chart(out_path: Path, title: str, rows: List[dict], label: str) -> None:
    width, height = 1200, 720
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)

    y = 18
    tw, th = text_size(draw, title, FONT_XL)
    draw.text(((width - tw) / 2, y), title, fill=BLACK, font=FONT_XL)
    y += th + 6
    subtitle = "Positive = mirror direction better, Negative = trained direction better"
    sw, sh = text_size(draw, subtitle, FONT_MD)
    draw.text(((width - sw) / 2, y), subtitle, fill=GRAY, font=FONT_MD)
    y += sh + 22

    left, right, bottom, top = 110, width - 50, height - 90, y
    zero_y = int((top + bottom) / 2)
    draw.line((left, zero_y, right, zero_y), fill=BLACK, width=2)
    for frac, txt in [(-0.5, "-50pp"), (-0.25, "-25pp"), (0.0, "0"), (0.25, "+25pp"), (0.5, "+50pp")]:
        yy = zero_y - int((bottom - top) * frac)
        draw.line((left, yy, right, yy), fill=GRID, width=1)
        tw, th = text_size(draw, txt, FONT_SM)
        draw.text((left - tw - 8, yy - th / 2), txt, fill=BLACK, font=FONT_SM)

    n = len(rows)
    slot = (right - left) / max(1, n)
    bar_w = int(slot * 0.55)
    for idx, row in enumerate(rows):
        x = int(left + slot * (idx + 0.5))
        gap = float(row["mirror_minus_trained"])
        bar_h = int((bottom - top) * min(abs(gap), 0.5))
        color = BLUE if gap >= 0 else RED
        if gap >= 0:
            y0, y1 = zero_y - bar_h, zero_y
        else:
            y0, y1 = zero_y, zero_y + bar_h
        draw.rectangle((x - bar_w // 2, y0, x + bar_w // 2, y1), fill=color)
        lab = row["geometry"].replace("_", "\n")
        lines = lab.split("\n")
        text_y = bottom + 10
        for line in lines:
            tw, th = text_size(draw, line, FONT_SM)
            draw.text((x - tw / 2, text_y), line, fill=BLACK, font=FONT_SM)
            text_y += th
        gap_txt = f"{gap * 100:+.1f}pp"
        tw, th = text_size(draw, gap_txt, FONT_SM)
        offset_y = y0 - th - 4 if gap >= 0 else y1 + 4
        draw.text((x - tw / 2, offset_y), gap_txt, fill=BLACK, font=FONT_SM)

    legend = f"{label}: mirror better {sum(r['mirror_minus_trained'] > 1e-9 for r in rows)}/8, trained better {sum(r['mirror_minus_trained'] < -1e-9 for r in rows)}/8"
    tw, th = text_size(draw, legend, FONT_MD)
    draw.text(((width - tw) / 2, height - th - 10), legend, fill=BLACK, font=FONT_MD)
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-final-json", required=True)
    parser.add_argument("--control-final-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    main_rows = build_rows(load_json(Path(args.main_final_json)))
    control_rows = build_rows(load_json(Path(args.control_final_json)))

    all_rows = []
    for label, rows in [("main", main_rows), ("control", control_rows)]:
        for row in rows:
            item = dict(row)
            item["model"] = label
            all_rows.append(item)
    write_csv(out_dir / "direction_bias_rows.csv", all_rows)

    draw_gap_chart(fig_dir / "main_final_mirror_gap.png", "Main Final: Trained Direction vs Mirror Direction", main_rows, "Main")
    draw_gap_chart(fig_dir / "control_final_mirror_gap.png", "Control Final: Trained Direction vs Mirror Direction", control_rows, "Control")

    lines = [
        "# Direction Bias Analysis",
        "",
        "Current split is not a clean left-vs-right generalization test.",
        "Each geometry family is split so that one direction is trained and its mirror is held out.",
        "That means the right question is not 'is there a global left bias?' but 'for each geometry pair, does the trained direction outperform its held-out mirror?'",
        "",
        "## Main Final",
        "",
        "| geometry | trained variant | mirror variant | trained | mirror | mirror - trained |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['geometry']} | {row['trained_variant']} | {row['mirror_variant']} | "
            f"{fmt_pct(row['trained_success_rate'])} | {fmt_pct(row['mirror_success_rate'])} | "
            f"{row['mirror_minus_trained'] * 100:+.1f}pp |"
        )
    lines.extend(
        [
            "",
            "## Control Final",
            "",
            "| geometry | trained variant | mirror variant | trained | mirror | mirror - trained |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in control_rows:
        lines.append(
            f"| {row['geometry']} | {row['trained_variant']} | {row['mirror_variant']} | "
            f"{fmt_pct(row['trained_success_rate'])} | {fmt_pct(row['mirror_success_rate'])} | "
            f"{row['mirror_minus_trained'] * 100:+.1f}pp |"
        )

    def counts(rows: List[dict]) -> Tuple[int, int, int]:
        mirror = sum(r["mirror_minus_trained"] > 1e-9 for r in rows)
        trained = sum(r["mirror_minus_trained"] < -1e-9 for r in rows)
        equal = len(rows) - mirror - trained
        return mirror, trained, equal

    main_counts = counts(main_rows)
    control_counts = counts(control_rows)
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            f"- Main final: mirror better in {main_counts[0]}/8 geometries, trained direction better in {main_counts[1]}/8, equal in {main_counts[2]}/8.",
            f"- Control final: mirror better in {control_counts[0]}/8 geometries, trained direction better in {control_counts[1]}/8, equal in {control_counts[2]}/8.",
            "- This is not consistent with a simple 'model learned only left turns' story.",
            "- The failure mode is closer to geometry-specific asymmetric transfer plus difficulty confounding from the split itself.",
            "- Better redesign: train with both left/right members of each geometry family together, and hold out whole geometry families instead of mirror direction.",
            "",
            "Saved figures:",
            f"- `{fig_dir / 'main_final_mirror_gap.png'}`",
            f"- `{fig_dir / 'control_final_mirror_gap.png'}`",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
