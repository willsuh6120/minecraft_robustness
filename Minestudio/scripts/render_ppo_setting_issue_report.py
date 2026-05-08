#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from minestudio.models import load_cross_view_rocket
except Exception:
    load_cross_view_rocket = None


@dataclass
class PhaseRow:
    run: str
    kind: str
    iteration: int
    episode_count: int
    success_count: int
    success_rate: float
    mean_steps: float


@dataclass
class UpdateRow:
    run: str
    iteration: int
    fragments: int
    learning_rate: float
    mean_kl_divergence: float
    mean_approx_kl: float
    mean_clip_fraction: float
    mean_total_loss: float


@dataclass
class ModuleRow:
    name: str
    trainable: int
    frozen: int

    @property
    def total(self) -> int:
        return self.trainable + self.frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPT-style assets explaining why the current PPO setting is wrong.")
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory from render_ppo_smoke_comparison.py containing phase/update CSVs and comparison PNGs.",
    )
    parser.add_argument(
        "--model-ckpt",
        required=True,
        help="Local model checkpoint to inspect for trainable/frozen parameter counts.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for slides, figures, tables, diagrams, and summary files.",
    )
    parser.add_argument("--old-ppo-clip", type=float, default=0.2)
    parser.add_argument("--new-ppo-clip", type=float, default=0.1)
    parser.add_argument("--old-kl-coef", type=float, default=0.01)
    parser.add_argument("--new-kl-coef", type=float, default=0.05)
    parser.add_argument("--old-min-successful-fragments", type=int, default=0)
    parser.add_argument("--new-min-successful-fragments", type=int, default=2)
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


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    title_font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (251, 252, 254),
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=26, fill=fill, outline=(220, 224, 230), width=2)
    draw.text((x0 + 24, y0 + 18), title, font=title_font, fill=(17, 24, 39))
    return (x0 + 24, y0 + 58, x1 - 24, y1 - 24)


def paste_contain(canvas: Image.Image, image_path: Path, box: Tuple[int, int, int, int], background: Tuple[int, int, int] = (255, 255, 255)) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.rounded_rectangle(box, radius=18, fill=background, outline=(226, 230, 236), width=2)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((w - 12, h - 12), Image.Resampling.LANCZOS)
    ix = x0 + (w - image.width) // 2
    iy = y0 + (h - image.height) // 2
    canvas.paste(image, (ix, iy))


def read_phase_rows(path: Path) -> List[PhaseRow]:
    rows: List[PhaseRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                PhaseRow(
                    run=row["run"],
                    kind=row["kind"],
                    iteration=int(row["iteration"]),
                    episode_count=int(row["episode_count"]),
                    success_count=int(row["success_count"]),
                    success_rate=float(row["success_rate"]),
                    mean_steps=float(row["mean_steps"]),
                )
            )
    return rows


def read_update_rows(path: Path) -> List[UpdateRow]:
    rows: List[UpdateRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                UpdateRow(
                    run=row["run"],
                    iteration=int(row["iteration"]),
                    fragments=int(row["fragments"]),
                    learning_rate=float(row["learning_rate"]),
                    mean_kl_divergence=float(row["mean_kl_divergence"]),
                    mean_approx_kl=float(row["mean_approx_kl"]),
                    mean_clip_fraction=float(row["mean_clip_fraction"]),
                    mean_total_loss=float(row["mean_total_loss"]),
                )
            )
    return rows


def rows_for_run(rows: Sequence[PhaseRow], run: str, kind: Optional[str] = None) -> List[PhaseRow]:
    selected = [row for row in rows if row.run == run]
    if kind is not None:
        selected = [row for row in selected if row.kind == kind]
    return sorted(selected, key=lambda item: (item.iteration, item.kind))


def updates_for_run(rows: Sequence[UpdateRow], run: str) -> List[UpdateRow]:
    return sorted([row for row in rows if row.run == run], key=lambda item: item.iteration)


def safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_count_range(values: Sequence[int]) -> str:
    if not values:
        return "-"
    return f"{min(values)}-{max(values)}"


def summarize_comparison(
    phase_rows: Sequence[PhaseRow],
    update_rows: Sequence[UpdateRow],
    old_label: str,
    new_label: str,
) -> Dict[str, Dict[str, float]]:
    old_eval = rows_for_run(phase_rows, old_label, "eval")
    new_eval = rows_for_run(phase_rows, new_label, "eval")
    old_collect = rows_for_run(phase_rows, old_label, "collect")
    new_collect = rows_for_run(phase_rows, new_label, "collect")
    old_base = rows_for_run(phase_rows, old_label, "baseline")
    new_base = rows_for_run(phase_rows, new_label, "baseline")
    old_updates = updates_for_run(update_rows, old_label)
    new_updates = updates_for_run(update_rows, new_label)

    common_update_iters = min(len(old_updates), len(new_updates))
    old_first = old_updates[:common_update_iters]
    new_first = new_updates[:common_update_iters]

    return {
        old_label: {
            "baseline_eval_success": old_base[0].success_rate if old_base else 0.0,
            "peak_eval_success": max((row.success_rate for row in old_eval), default=0.0),
            "peak_eval_iter": max(old_eval, key=lambda row: row.success_rate).iteration if old_eval else 0,
            "latest_eval_success": old_eval[-1].success_rate if old_eval else 0.0,
            "latest_collect_success": old_collect[-1].success_rate if old_collect else 0.0,
            "eval_success_std": safe_std([row.success_rate for row in old_eval]),
            "collect_success_min": min((row.success_count for row in old_collect), default=0),
            "collect_success_max": max((row.success_count for row in old_collect), default=0),
            "collect_episodes": old_collect[0].episode_count if old_collect else 0,
            "eval_episodes": old_eval[0].episode_count if old_eval else 0,
            "learning_rate": old_updates[0].learning_rate if old_updates else 0.0,
            "mean_kl_first_common": safe_mean([row.mean_kl_divergence for row in old_first]),
            "kl_std_first_common": safe_std([row.mean_kl_divergence for row in old_first]),
            "mean_clip_first_common": safe_mean([row.mean_clip_fraction for row in old_first]),
        },
        new_label: {
            "baseline_eval_success": new_base[0].success_rate if new_base else 0.0,
            "peak_eval_success": max((row.success_rate for row in new_eval), default=0.0),
            "peak_eval_iter": max(new_eval, key=lambda row: row.success_rate).iteration if new_eval else 0,
            "latest_eval_success": new_eval[-1].success_rate if new_eval else 0.0,
            "latest_collect_success": new_collect[-1].success_rate if new_collect else 0.0,
            "eval_success_std": safe_std([row.success_rate for row in new_eval]),
            "collect_success_min": min((row.success_count for row in new_collect), default=0),
            "collect_success_max": max((row.success_count for row in new_collect), default=0),
            "collect_episodes": new_collect[0].episode_count if new_collect else 0,
            "eval_episodes": new_eval[0].episode_count if new_eval else 0,
            "learning_rate": new_updates[0].learning_rate if new_updates else 0.0,
            "mean_kl_first_common": safe_mean([row.mean_kl_divergence for row in new_first]),
            "kl_std_first_common": safe_std([row.mean_kl_divergence for row in new_first]),
            "mean_clip_first_common": safe_mean([row.mean_clip_fraction for row in new_first]),
        },
    }


def count_model_modules(model_ckpt: Path) -> Tuple[int, int, List[ModuleRow]]:
    if load_cross_view_rocket is None:
        raise RuntimeError("minestudio.models.load_cross_view_rocket is unavailable in this Python environment.")

    model = load_cross_view_rocket(str(model_ckpt))
    module_map: Dict[str, ModuleRow] = {}
    total_trainable = 0
    total_frozen = 0

    for name, param in model.named_parameters():
        prefix = name.split(".", 1)[0]
        row = module_map.setdefault(prefix, ModuleRow(name=prefix, trainable=0, frozen=0))
        count = int(param.numel())
        if param.requires_grad:
            row.trainable += count
            total_trainable += count
        else:
            row.frozen += count
            total_frozen += count

    rows = sorted(module_map.values(), key=lambda item: (item.trainable + item.frozen), reverse=True)
    return total_trainable, total_frozen, rows


def render_table_png(
    title: str,
    subtitle: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: Path,
    col_widths: Optional[Sequence[int]] = None,
    row_height: int = 46,
) -> Path:
    title_font = find_font(36, bold=True)
    subtitle_font = find_font(18)
    header_font = find_font(20, bold=True)
    cell_font = find_font(20)
    padding_x = 24
    top = 42
    if col_widths is None:
        col_widths = [180 for _ in columns]
    table_width = sum(col_widths)
    width = table_width + padding_x * 2
    height = top + 72 + row_height * (len(rows) + 1) + 32
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    draw.text((padding_x, top), title, font=title_font, fill=(17, 24, 39))
    draw.text((padding_x + 2, top + 44), subtitle, font=subtitle_font, fill=(89, 96, 109))
    x0 = padding_x
    y0 = top + 88
    draw.rounded_rectangle((x0, y0, x0 + table_width, y0 + row_height), radius=14, fill=(229, 236, 255))
    cx = x0
    for idx, col in enumerate(columns):
        draw.text((cx + 12, y0 + 11), col, font=header_font, fill=(30, 41, 59))
        cx += col_widths[idx]
    for row_idx, row in enumerate(rows):
        yy = y0 + row_height * (row_idx + 1)
        fill = (255, 255, 255) if row_idx % 2 == 0 else (242, 245, 249)
        draw.rounded_rectangle((x0, yy, x0 + table_width, yy + row_height), radius=10, fill=fill)
        cx = x0
        for col_idx, value in enumerate(row):
            draw.text((cx + 12, yy + 11), str(value), font=cell_font, fill=(31, 41, 55))
            cx += col_widths[col_idx]
    image.save(out_path)
    return out_path


def render_trainable_ratio_figure(
    total_trainable: int,
    total_frozen: int,
    out_path: Path,
) -> Path:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(42, bold=True)
    body_font = find_font(26)
    small_font = find_font(20)
    draw.text((56, 34), "Current PPO Trainable Footprint", font=title_font, fill=(17, 24, 39))
    draw.text((58, 88), "This is the live trainable/frozen split under the current post-training code path.", font=small_font, fill=(89, 96, 109))
    draw.rounded_rectangle((48, 126, 1350, 840), radius=26, fill=(252, 252, 253), outline=(224, 228, 234), width=2)

    total = total_trainable + total_frozen
    train_ratio = total_trainable / float(total) if total else 0.0
    frozen_ratio = total_frozen / float(total) if total else 0.0

    center = (430, 500)
    radius = 210
    start = -90
    train_end = start + int(round(360 * train_ratio))
    draw.pieslice((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), start, train_end, fill=(37, 99, 235))
    draw.pieslice((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), train_end, start + 360, fill=(203, 213, 225))
    inner = radius - 72
    draw.ellipse((center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner), fill=(252, 252, 253))

    pct_font = find_font(56, bold=True)
    draw.text((center[0] - 112, center[1] - 46), f"{train_ratio * 100:.1f}%", font=pct_font, fill=(17, 24, 39))
    draw.text((center[0] - 88, center[1] + 20), "trainable", font=body_font, fill=(71, 85, 105))

    lx = 770
    ly = 250
    legend_font = find_font(28, bold=True)
    value_font = find_font(28)
    items = [
        ("Trainable", (37, 99, 235), total_trainable, train_ratio),
        ("Frozen", (203, 213, 225), total_frozen, frozen_ratio),
    ]
    for label, color, count, ratio in items:
        draw.rounded_rectangle((lx, ly, lx + 34, ly + 22), radius=6, fill=color)
        draw.text((lx + 50, ly - 8), label, font=legend_font, fill=(17, 24, 39))
        draw.text((lx + 50, ly + 34), f"{count:,} params  ({ratio * 100:.1f}%)", font=value_font, fill=(55, 65, 81))
        ly += 140

    note = (
        "Under the current PPO setup, more than half of the model remains trainable. "
        "That is much larger than a typical conservative RL post-training footprint."
    )
    draw_wrapped_text(draw, (770, 560), note, body_font, (31, 41, 55), 500, line_spacing=10)

    image.save(out_path)
    return out_path


def render_cfg_mismatch_diagram(out_path: Path) -> Path:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(40, bold=True)
    body_font = find_font(24)
    small_font = find_font(18)
    draw.text((56, 34), "CFG Behavior-Policy Mismatch", font=title_font, fill=(17, 24, 39))
    draw.text((58, 88), "CFG = Classifier-Free Guidance. The rollout action and stored old_logprob come from different policies.", font=small_font, fill=(89, 96, 109))

    def box(x0: int, y0: int, x1: int, y1: int, title: str, lines: Sequence[str], fill: Tuple[int, int, int]) -> None:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=fill, outline=(220, 224, 230), width=2)
        draw.text((x0 + 20, y0 + 18), title, font=find_font(28, bold=True), fill=(17, 24, 39))
        yy = y0 + 62
        for line in lines:
            draw.text((x0 + 22, yy), line, font=body_font, fill=(31, 41, 55))
            yy += 34

    box(60, 180, 470, 410, "1. Conditioned Branch", [
        "goal image/mask enabled",
        "produces cond_logits",
        "cache_latents stores this branch",
    ], (239, 246, 255))
    box(60, 510, 470, 740, "2. Base Branch", [
        "goal inputs blanked out",
        "produces base_logits",
        "used only for CFG mixing",
    ], (245, 243, 255))
    box(590, 285, 1020, 525, "3. Actual Behavior Policy", [
        "mixed_logits = (1+k)*cond - k*base",
        "rollout action is sampled here",
        "this is the true behavior policy",
    ], (240, 253, 244))
    box(1120, 180, 1540, 410, "4. Stored old_logprob", [
        "computed from cond_logits only",
        "does NOT use mixed_logits",
        "behavior != logged policy",
    ], (254, 242, 242))
    box(1120, 510, 1540, 740, "5. PPO Update Assumption", [
        "ratio = exp(new_logprob - old_logprob)",
        "assumes old_logprob came from behavior",
        "that assumption is broken here",
    ], (255, 247, 237))

    arrow_color = (71, 85, 105)
    draw.line((470, 295, 590, 375), fill=arrow_color, width=6)
    draw.polygon([(590, 375), (566, 364), (575, 395)], fill=arrow_color)
    draw.line((470, 625, 590, 435), fill=arrow_color, width=6)
    draw.polygon([(590, 435), (564, 433), (581, 456)], fill=arrow_color)
    draw.line((1020, 375, 1120, 295), fill=(220, 38, 38), width=6)
    draw.polygon([(1120, 295), (1094, 298), (1111, 320)], fill=(220, 38, 38))
    draw.line((1020, 445, 1120, 625), fill=(220, 38, 38), width=6)
    draw.polygon([(1120, 625), (1094, 620), (1112, 603)], fill=(220, 38, 38))

    draw.text((776, 552), "Mismatch", font=find_font(34, bold=True), fill=(220, 38, 38))
    foot = (
        "Code path: cfg_wrapper mixes cond/base logits for action sampling, but crossview_utils logs old_logprob "
        "from cache_latents['pi_logits'], which are conditioned-branch logits."
    )
    draw_wrapped_text(draw, (58, 800), foot, small_font, (89, 96, 109), 1480)
    image.save(out_path)
    return out_path


def render_hyperparam_table(
    summary: Dict[str, Dict[str, float]],
    old_label: str,
    new_label: str,
    args: argparse.Namespace,
    out_path: Path,
) -> Path:
    old = summary[old_label]
    new = summary[new_label]
    rows = [
        ["Collect episodes / iter", str(old["collect_episodes"]), str(new["collect_episodes"]), "More data per PPO step"],
        ["Eval episodes / iter", str(old["eval_episodes"]), str(new["eval_episodes"]), "Held constant"],
        ["Learning rate", f"{old['learning_rate']:.1e}", f"{new['learning_rate']:.1e}", "More conservative update"],
        ["PPO clip", str(args.old_ppo_clip), str(args.new_ppo_clip), "Tighter ratio band"],
        ["KL coef", str(args.old_kl_coef), str(args.new_kl_coef), "Stronger KL penalty"],
        ["Min successful fragments", str(args.old_min_successful_fragments), str(args.new_min_successful_fragments), "Skip empty-signal updates"],
    ]
    return render_table_png(
        "Hyperparameter Changes",
        "Old unstable run vs adjusted run",
        ["Metric", old_label, new_label, "Intent"],
        rows,
        out_path,
        col_widths=[300, 180, 180, 380],
    )


def render_stability_summary_table(
    summary: Dict[str, Dict[str, float]],
    old_label: str,
    new_label: str,
    out_path: Path,
) -> Path:
    old = summary[old_label]
    new = summary[new_label]
    rows = [
        ["Baseline eval success", format_pct(old["baseline_eval_success"]), format_pct(new["baseline_eval_success"]), "Both start at 10%"],
        ["Peak eval success", f"{format_pct(old['peak_eval_success'])} @ {int(old['peak_eval_iter']):03d}", f"{format_pct(new['peak_eval_success'])} @ {int(new['peak_eval_iter']):03d}", "Old spikes higher once"],
        ["Latest eval success", format_pct(old["latest_eval_success"]), format_pct(new["latest_eval_success"]), "Neither gives a clean monotonic rise"],
        ["Eval success std", format_float(old["eval_success_std"], 3), format_float(new["eval_success_std"], 3), "Adjusted run is less volatile"],
        ["Mean KL, first 5 common iters", format_float(old["mean_kl_first_common"], 4), format_float(new["mean_kl_first_common"], 4), "Adjusted run moves less aggressively"],
        ["KL std, first 5 common iters", format_float(old["kl_std_first_common"], 4), format_float(new["kl_std_first_common"], 4), "Adjusted run is stabler"],
        ["Collect success count / iter", format_count_range([int(old["collect_success_min"]), int(old["collect_success_max"])]), format_count_range([int(new["collect_success_min"]), int(new["collect_success_max"])]), "Adjusted run never goes to zero"],
    ]
    return render_table_png(
        "Stability Summary",
        "Empirical evidence that the current PPO regime is active but not reliable",
        ["Metric", old_label, new_label, "Read"],
        rows,
        out_path,
        col_widths=[310, 220, 220, 420],
        row_height=50,
    )


def render_module_table(rows: Sequence[ModuleRow], out_path: Path) -> Path:
    display_rows: List[List[str]] = []
    for row in rows[:8]:
        display_rows.append(
            [
                row.name,
                f"{row.trainable / 1e6:.2f}M",
                f"{row.frozen / 1e6:.2f}M",
                f"{row.total / 1e6:.2f}M",
            ]
        )
    return render_table_png(
        "Top-Level Module Trainability",
        "Current PPO code path: only view_backbone is explicitly frozen",
        ["Module", "Trainable", "Frozen", "Total"],
        display_rows,
        out_path,
        col_widths=[280, 180, 180, 180],
    )


def render_metric_cards(summary: Dict[str, Dict[str, float]], old_label: str, new_label: str, out_path: Path) -> Path:
    old = summary[old_label]
    new = summary[new_label]
    width, height = 1280, 540
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(38, bold=True)
    card_title_font = find_font(24, bold=True)
    value_font = find_font(42, bold=True)
    small_font = find_font(18)
    draw.text((46, 28), "Key Metric Cards", font=title_font, fill=(17, 24, 39))
    cards = [
        ("Peak eval success", f"{old['peak_eval_success']*100:.0f}%", f"{new['peak_eval_success']*100:.0f}%", "Old spikes once; new never collapses as hard"),
        ("Latest eval success", f"{old['latest_eval_success']*100:.0f}%", f"{new['latest_eval_success']*100:.0f}%", "Neither run proves robust monotonic learning"),
        ("Mean KL (first 5)", f"{old['mean_kl_first_common']:.4f}", f"{new['mean_kl_first_common']:.4f}", "Adjusted run is materially less aggressive"),
    ]
    x = 48
    for title, old_value, new_value, note in cards:
        draw.rounded_rectangle((x, 110, x + 370, 470), radius=24, fill=(252, 252, 253), outline=(224, 228, 234), width=2)
        draw.text((x + 20, 132), title, font=card_title_font, fill=(17, 24, 39))
        draw.text((x + 20, 190), old_label, font=small_font, fill=(99, 102, 110))
        draw.text((x + 20, 220), old_value, font=value_font, fill=(220, 38, 38))
        draw.text((x + 190, 190), new_label, font=small_font, fill=(99, 102, 110))
        draw.text((x + 190, 220), new_value, font=value_font, fill=(37, 99, 235))
        draw_wrapped_text(draw, (x + 20, 320), note, small_font, (55, 65, 81), 320)
        x += 392
    image.save(out_path)
    return out_path


def render_slide_empirical(
    figures_dir: Path,
    tables_dir: Path,
    slides_dir: Path,
    old_label: str,
    new_label: str,
) -> Path:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (243, 246, 250))
    draw = ImageDraw.Draw(image)
    title_font = find_font(46, bold=True)
    body_font = find_font(24)
    panel_title_font = find_font(26, bold=True)
    draw.text((60, 34), "Slide 1. Current PPO Setting Fails the Same-World Stability Test", font=title_font, fill=(17, 24, 39))
    draw.text((62, 92), "Fixed single world, fixed evaluation seeds, same task. Improvement should be stable if the update regime is healthy.", font=find_font(20), fill=(89, 96, 109))

    left = draw_panel(draw, (46, 140, 760, 1030), "Claim", panel_title_font)
    bullets = [
        "This is not a distribution-shift artifact. The smoke test uses one static world and fixed eval seeds.",
        f"The old run proves PPO is active because eval briefly rises from 10% to 70%, but then collapses to 0%.",
        f"The adjusted run lowers KL and prevents total collect collapse, but it still fails to produce a clean monotonic eval curve.",
        "Conclusion: the current PPO setting is not suitable for claiming reliable performance gains.",
    ]
    y = left[1]
    for bullet in bullets:
        draw.text((left[0], y), "•", font=body_font, fill=(37, 99, 235))
        used = draw_wrapped_text(draw, (left[0] + 22, y), bullet, body_font, (31, 41, 55), left[2] - left[0] - 20)
        y += used + 20

    paste_contain(image, figures_dir / "fig_eval_success_rate_comparison.png", (790, 140, 1860, 630))
    paste_contain(image, tables_dir / "table_hyperparameter_changes.png", (790, 670, 1340, 1030))
    paste_contain(image, figures_dir / "fig_key_metric_cards.png", (1360, 670, 1860, 1030))

    out_path = slides_dir / "slide_01_empirical.png"
    image.save(out_path)
    return out_path


def render_slide_update_regime(
    figures_dir: Path,
    tables_dir: Path,
    slides_dir: Path,
) -> Path:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (243, 246, 250))
    draw = ImageDraw.Draw(image)
    draw.text((60, 34), "Slide 2. The Update Regime Is Noisy Even After Conservative Tuning", font=find_font(46, bold=True), fill=(17, 24, 39))
    draw.text((62, 92), "Lower learning rate and stronger KL control helped, but did not create stable upward learning.", font=find_font(20), fill=(89, 96, 109))
    paste_contain(image, figures_dir / "fig_update_kl_comparison.png", (46, 140, 940, 610))
    paste_contain(image, figures_dir / "fig_collect_success_count_comparison.png", (980, 140, 1874, 610))
    paste_contain(image, figures_dir / "fig_update_clip_fraction_comparison.png", (46, 650, 940, 1030))
    paste_contain(image, tables_dir / "table_stability_summary.png", (980, 650, 1874, 1030))
    out_path = slides_dir / "slide_02_update_regime.png"
    image.save(out_path)
    return out_path


def render_slide_root_causes(
    figures_dir: Path,
    tables_dir: Path,
    diagrams_dir: Path,
    slides_dir: Path,
) -> Path:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (243, 246, 250))
    draw = ImageDraw.Draw(image)
    body_font = find_font(23)
    draw.text((60, 34), "Slide 3. Two Structural Problems Remain in the Current PPO Configuration", font=find_font(46, bold=True), fill=(17, 24, 39))
    draw.text((62, 92), "This is not just a hyperparameter issue. The current implementation and trainable footprint both work against stable PPO.", font=find_font(20), fill=(89, 96, 109))
    paste_contain(image, diagrams_dir / "diagram_cfg_behavior_mismatch.png", (46, 140, 1080, 860))
    paste_contain(image, figures_dir / "fig_trainable_ratio.png", (1110, 140, 1874, 640))
    paste_contain(image, tables_dir / "table_trainable_modules.png", (1110, 670, 1874, 1030))
    note_box = draw_panel(draw, (46, 880, 1080, 1030), "Why This Matters", find_font(24, bold=True))
    bullets = [
        "Behavior policy mismatch: rollout action is sampled from CFG-mixed logits, but old_logprob is recorded from conditioned logits.",
        "Over-large trainable scope: more than half of the model remains trainable during PPO post-training.",
    ]
    y = note_box[1]
    for bullet in bullets:
        draw.text((note_box[0], y), "•", font=body_font, fill=(220, 38, 38))
        used = draw_wrapped_text(draw, (note_box[0] + 22, y), bullet, body_font, (31, 41, 55), note_box[2] - note_box[0] - 10)
        y += used + 14
    out_path = slides_dir / "slide_03_root_causes.png"
    image.save(out_path)
    return out_path


def copy_existing_comparison_figures(comparison_dir: Path, figures_dir: Path) -> Dict[str, Path]:
    mapping = {
        "ppo_compare_eval_success_rate.png": "fig_eval_success_rate_comparison.png",
        "ppo_compare_update_kl.png": "fig_update_kl_comparison.png",
        "ppo_compare_collect_success_count.png": "fig_collect_success_count_comparison.png",
        "ppo_compare_update_clip_fraction.png": "fig_update_clip_fraction_comparison.png",
    }
    copied: Dict[str, Path] = {}
    for source_name, target_name in mapping.items():
        src = comparison_dir / source_name
        dst = figures_dir / target_name
        shutil.copy2(src, dst)
        copied[target_name] = dst
    return copied


def write_summary_md(
    out_path: Path,
    old_label: str,
    new_label: str,
    summary: Dict[str, Dict[str, float]],
    total_trainable: int,
    total_frozen: int,
) -> None:
    old = summary[old_label]
    new = summary[new_label]
    total = total_trainable + total_frozen
    train_ratio = total_trainable / float(total) if total else 0.0
    lines = [
        "# PPO Setting Issue Report",
        "",
        "## Main Message",
        "",
        "The current PPO setup is active but not valid for a strong stability claim.",
        "",
        "## Empirical Evidence",
        "",
        f"- Baseline eval success starts at {format_pct(old['baseline_eval_success'])} in both runs.",
        f"- Old unstable run peaks at {format_pct(old['peak_eval_success'])} on iter {int(old['peak_eval_iter']):03d}, then collapses to {format_pct(old['latest_eval_success'])}.",
        f"- Adjusted run peaks at {format_pct(new['peak_eval_success'])} on iter {int(new['peak_eval_iter']):03d}, and ends at {format_pct(new['latest_eval_success'])}.",
        f"- Mean KL over the first common iterations falls from {old['mean_kl_first_common']:.4f} to {new['mean_kl_first_common']:.4f}, so the adjusted run is less aggressive but still not convincingly monotonic.",
        "",
        "## Structural Root Causes",
        "",
        "- CFG behavior mismatch: rollout actions are sampled from CFG-mixed logits, but old_logprob is recorded from conditioned-only logits.",
        f"- Current trainable footprint is {total_trainable:,} / {total:,} params ({train_ratio * 100:.1f}%).",
        "- Only the view_backbone is explicitly frozen in the current code path.",
        "",
        "## Practical Takeaway",
        "",
        "Use these slides to argue that the current PPO result should be described as 'plumbing works, but the setting is not yet suitable for stable learning claims.'",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    diagrams_dir = out_dir / "diagrams"
    slides_dir = out_dir / "slides"
    for directory in (out_dir, figures_dir, tables_dir, diagrams_dir, slides_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase_rows = read_phase_rows(comparison_dir / "phase_comparison.csv")
    update_rows = read_update_rows(comparison_dir / "update_comparison.csv")
    compare_summary = load_json(comparison_dir / "ppo_compare_summary.json")
    old_label = compare_summary["old"]["label"]
    new_label = compare_summary["new"]["label"]
    summary = summarize_comparison(phase_rows, update_rows, old_label, new_label)

    copy_existing_comparison_figures(comparison_dir, figures_dir)

    total_trainable, total_frozen, module_rows = count_model_modules(Path(args.model_ckpt).resolve())

    render_hyperparam_table(summary, old_label, new_label, args, tables_dir / "table_hyperparameter_changes.png")
    render_stability_summary_table(summary, old_label, new_label, tables_dir / "table_stability_summary.png")
    render_module_table(module_rows, tables_dir / "table_trainable_modules.png")
    render_trainable_ratio_figure(total_trainable, total_frozen, figures_dir / "fig_trainable_ratio.png")
    render_cfg_mismatch_diagram(diagrams_dir / "diagram_cfg_behavior_mismatch.png")
    render_metric_cards(summary, old_label, new_label, figures_dir / "fig_key_metric_cards.png")

    slide_1 = render_slide_empirical(figures_dir, tables_dir, slides_dir, old_label, new_label)
    slide_2 = render_slide_update_regime(figures_dir, tables_dir, slides_dir)
    slide_3 = render_slide_root_causes(figures_dir, tables_dir, diagrams_dir, slides_dir)

    write_summary_md(out_dir / "speaker_notes.md", old_label, new_label, summary, total_trainable, total_frozen)

    manifest = {
        "comparison_dir": str(comparison_dir),
        "model_ckpt": str(Path(args.model_ckpt).resolve()),
        "old_label": old_label,
        "new_label": new_label,
        "slides": [str(slide_1), str(slide_2), str(slide_3)],
        "figures": sorted(str(path) for path in figures_dir.glob("*.png")),
        "tables": sorted(str(path) for path in tables_dir.glob("*.png")),
        "diagrams": sorted(str(path) for path in diagrams_dir.glob("*.png")),
        "summary": summary,
        "total_trainable": total_trainable,
        "total_frozen": total_frozen,
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
