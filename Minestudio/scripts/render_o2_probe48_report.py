#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "render_o2_generalization_vm1_report.py"
DEFAULT_IMPORT_ROOT = Path(
    "/home/gyulab/envgen2/Minestudio/outputs/imported_vm_runs/"
    "ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330_progress_vm1"
)


def load_helper_module():
    spec = importlib.util.spec_from_file_location("vm1_report_helper", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper_module()


TRAINED = {
    "name": "Final Trained Model",
    "source_label": "VM1 o2_model_eval_and_collect_probes/20260428_062132",
    "eval_bank": {"successes": 17, "episodes": 48, "rate": 17 / 48, "mean_steps": 132.56},
    "collect_overall": {"successes": 82, "episodes": 240, "rate": 82 / 240, "mean_steps": 131.72},
    "collect_blocks": {
        1: {"successes": 11, "episodes": 48, "rate": 11 / 48, "mean_steps": None},
        2: {"successes": 21, "episodes": 48, "rate": 21 / 48, "mean_steps": None},
        3: {"successes": 5, "episodes": 48, "rate": 5 / 48, "mean_steps": None},
        4: {"successes": 18, "episodes": 48, "rate": 18 / 48, "mean_steps": None},
        5: {"successes": 27, "episodes": 48, "rate": 27 / 48, "mean_steps": None},
    },
}

BASE = {
    "name": "ROCKET-2 Base Model",
    "source_label": "VM2 rollout-only 48-episode base probe",
    "eval_bank": {"successes": 6, "episodes": 48, "rate": 6 / 48, "mean_steps": 145.44},
    "collect_overall": {"successes": 46, "episodes": 240, "rate": 46 / 240, "mean_steps": 143.33},
    "collect_blocks": {
        1: {"successes": 3, "episodes": 48, "rate": 3 / 48, "mean_steps": 148.63},
        2: {"successes": 17, "episodes": 48, "rate": 17 / 48, "mean_steps": 134.77},
        3: {"successes": 1, "episodes": 48, "rate": 1 / 48, "mean_steps": 149.96},
        4: {"successes": 11, "episodes": 48, "rate": 11 / 48, "mean_steps": 142.60},
        5: {"successes": 14, "episodes": 48, "rate": 14 / 48, "mean_steps": 140.67},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPT-ready PNGs for 48-episode probe comparison.")
    parser.add_argument("--import-root", default=str(DEFAULT_IMPORT_ROOT), help="Imported VM1 run root.")
    parser.add_argument("--out-dir", default="", help="Optional output directory.")
    return parser.parse_args()


def fmt_pct(rate: float) -> str:
    return f"{rate * 100.0:.1f}%"


def fmt_pp(delta: float) -> str:
    return f"{delta * 100.0:+.1f}pp"


def fmt_steps(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def render_results_table(out_path: Path) -> None:
    rows = ["eval_bank"] + [f"block_{idx:03d}" for idx in range(1, 6)] + ["collect_overall"]
    col_labels = ["Base Success", "Trained Success", "Delta", "Base Steps", "Trained Steps"]
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[int, int, int]]] = []

    def add_row(base_entry: Dict, trained_entry: Dict) -> None:
        delta = trained_entry["rate"] - base_entry["rate"]
        delta_fill = (220, 252, 231) if delta >= 0 else (254, 226, 226)
        cell_text.append(
            [
                f"{base_entry['successes']}/{base_entry['episodes']}\n{fmt_pct(base_entry['rate'])}",
                f"{trained_entry['successes']}/{trained_entry['episodes']}\n{fmt_pct(trained_entry['rate'])}",
                fmt_pp(delta),
                fmt_steps(base_entry.get("mean_steps")),
                fmt_steps(trained_entry.get("mean_steps")),
            ]
        )
        cell_colors.append(
            [
                (254, 249, 195),
                (220, 252, 231),
                delta_fill,
                (243, 244, 246),
                (243, 244, 246),
            ]
        )

    add_row(BASE["eval_bank"], TRAINED["eval_bank"])
    for idx in range(1, 6):
        add_row(BASE["collect_blocks"][idx], TRAINED["collect_blocks"][idx])
    add_row(BASE["collect_overall"], TRAINED["collect_overall"])

    HELPER.render_table_image(
        title="48-Episode Probe Results",
        subtitle="Rollout-only comparison. No PPO updates are run here. final_eval_bank is excluded.",
        row_labels=rows,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=220,
        col_width=180,
        cell_height=74,
    )


def render_success_rate_chart(out_path: Path) -> None:
    labels = ["eval", "B1", "B2", "B3", "B4", "B5", "avg"]
    base_rates = [BASE["eval_bank"]["rate"]] + [BASE["collect_blocks"][i]["rate"] for i in range(1, 6)] + [BASE["collect_overall"]["rate"]]
    trained_rates = [TRAINED["eval_bank"]["rate"]] + [TRAINED["collect_blocks"][i]["rate"] for i in range(1, 6)] + [TRAINED["collect_overall"]["rate"]]

    width, height = 2280, 1180
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(42, bold=True)
    subtitle_font = HELPER.find_font(22)
    axis_font = HELPER.find_font(20)
    label_font = HELPER.find_font(18)
    small_font = HELPER.find_font(16)

    draw.text((72, 46), "48-Episode Probe: Success Rate Comparison", fill=(17, 24, 39), font=title_font)
    draw.text((72, 102), "Base vs final trained model on held-out eval_bank and collect block worlds.", fill=(75, 85, 99), font=subtitle_font)

    plot = (150, 220, 2140, 920)
    left, top, right, bottom = plot
    y_max = 0.65

    for tick_idx in range(8):
        rate = y_max * tick_idx / 7.0
        y = bottom - (bottom - top) * (rate / y_max)
        draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
        label = f"{int(round(rate * 100.0))}%"
        tw, th = HELPER.text_size(draw, label, axis_font)
        draw.text((left - tw - 18, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
    draw.line((left, top, left, bottom), fill=(55, 65, 81), width=3)
    draw.line((left, bottom, right, bottom), fill=(55, 65, 81), width=3)

    base_color = (234, 179, 8)
    trained_color = (16, 185, 129)
    group_w = (right - left) / len(labels)
    bar_w = group_w * 0.22

    def y_for(rate: float) -> float:
        return bottom - (bottom - top) * (rate / y_max)

    for idx, label in enumerate(labels):
        center_x = left + group_w * (idx + 0.5)
        bx0 = center_x - bar_w - 12
        bx1 = center_x - 12
        tx0 = center_x + 12
        tx1 = center_x + bar_w + 12

        by = y_for(base_rates[idx])
        ty = y_for(trained_rates[idx])
        draw.rectangle((bx0, by, bx1, bottom), fill=base_color)
        draw.rectangle((tx0, ty, tx1, bottom), fill=trained_color)
        draw.text((bx0, by - 24), fmt_pct(base_rates[idx]), fill=(120, 53, 15), font=small_font)
        draw.text((tx0, ty - 24), fmt_pct(trained_rates[idx]), fill=(6, 95, 70), font=small_font)
        tw, th = HELPER.text_size(draw, label, label_font)
        draw.text((center_x - tw / 2, bottom + 18), label, fill=(17, 24, 39), font=label_font)

    legend_x = left + 10
    legend_y = top - 84
    draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), fill=base_color)
    draw.text((legend_x + 34, legend_y - 1), "Base", fill=(31, 41, 55), font=label_font)
    lx2 = legend_x + 180
    draw.rectangle((lx2, legend_y, lx2 + 24, legend_y + 24), fill=trained_color)
    draw.text((lx2 + 34, legend_y - 1), "Trained", fill=(31, 41, 55), font=label_font)
    draw.text((72, 1018), f"Sources: {TRAINED['source_label']} | {BASE['source_label']}", fill=(107, 114, 128), font=small_font)
    image.save(out_path)


def render_delta_chart(out_path: Path) -> None:
    labels = ["eval_bank", "block_001", "block_002", "block_003", "block_004", "block_005", "collect_avg"]
    deltas = [
        TRAINED["eval_bank"]["rate"] - BASE["eval_bank"]["rate"],
        TRAINED["collect_blocks"][1]["rate"] - BASE["collect_blocks"][1]["rate"],
        TRAINED["collect_blocks"][2]["rate"] - BASE["collect_blocks"][2]["rate"],
        TRAINED["collect_blocks"][3]["rate"] - BASE["collect_blocks"][3]["rate"],
        TRAINED["collect_blocks"][4]["rate"] - BASE["collect_blocks"][4]["rate"],
        TRAINED["collect_blocks"][5]["rate"] - BASE["collect_blocks"][5]["rate"],
        TRAINED["collect_overall"]["rate"] - BASE["collect_overall"]["rate"],
    ]

    width, height = 2120, 1040
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(40, bold=True)
    subtitle_font = HELPER.find_font(22)
    label_font = HELPER.find_font(18)
    small_font = HELPER.find_font(16)

    draw.text((72, 42), "Lift Over Base Model", fill=(17, 24, 39), font=title_font)
    draw.text((72, 96), "Positive bars are percentage-point gains of the final trained model over ROCKET-2 base.", fill=(75, 85, 99), font=subtitle_font)

    plot = (250, 180, 2020, 920)
    left, top, right, bottom = plot
    x_zero = left + 250
    max_delta = max(deltas)
    scale = (right - x_zero - 90) / max_delta

    for idx, label in enumerate(labels):
        y0 = top + idx * 98
        y_mid = y0 + 36
        draw.line((left, y_mid, right, y_mid), fill=(243, 244, 246), width=1)
        tw, th = HELPER.text_size(draw, label, label_font)
        draw.text((left + 24, y_mid - th / 2), label, fill=(17, 24, 39), font=label_font)
        bar_len = deltas[idx] * scale
        fill = (16, 185, 129) if deltas[idx] >= 0 else (220, 38, 38)
        draw.rectangle((x_zero, y_mid - 18, x_zero + bar_len, y_mid + 18), fill=fill)
        draw.text((x_zero + bar_len + 14, y_mid - 12), fmt_pp(deltas[idx]), fill=fill, font=small_font)

    draw.line((x_zero, top - 8, x_zero, bottom + 8), fill=(55, 65, 81), width=3)
    image.save(out_path)


def render_mean_steps_chart(out_path: Path) -> None:
    labels = ["eval", "B1", "B2", "B3", "B4", "B5", "avg"]
    base_steps = [
        BASE["eval_bank"]["mean_steps"],
        BASE["collect_blocks"][1]["mean_steps"],
        BASE["collect_blocks"][2]["mean_steps"],
        BASE["collect_blocks"][3]["mean_steps"],
        BASE["collect_blocks"][4]["mean_steps"],
        BASE["collect_blocks"][5]["mean_steps"],
        BASE["collect_overall"]["mean_steps"],
    ]
    trained_steps = [
        TRAINED["eval_bank"]["mean_steps"],
        None,
        None,
        None,
        None,
        None,
        TRAINED["collect_overall"]["mean_steps"],
    ]

    width, height = 2280, 1180
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(42, bold=True)
    subtitle_font = HELPER.find_font(22)
    axis_font = HELPER.find_font(20)
    label_font = HELPER.find_font(18)
    small_font = HELPER.find_font(16)

    draw.text((72, 46), "Mean Steps per Episode", fill=(17, 24, 39), font=title_font)
    draw.text((72, 102), "Lower is better. Trained-model block-level mean steps were not reported, so only eval/overall are shown for trained.", fill=(75, 85, 99), font=subtitle_font)

    plot = (150, 220, 2140, 920)
    left, top, right, bottom = plot
    y_min = 120.0
    y_max = 150.0

    for tick_idx in range(7):
        value = y_min + (y_max - y_min) * tick_idx / 6.0
        y = bottom - (bottom - top) * ((value - y_min) / (y_max - y_min))
        draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
        label = f"{value:.0f}"
        tw, th = HELPER.text_size(draw, label, axis_font)
        draw.text((left - tw - 18, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
    draw.line((left, top, left, bottom), fill=(55, 65, 81), width=3)
    draw.line((left, bottom, right, bottom), fill=(55, 65, 81), width=3)

    base_color = (245, 158, 11)
    trained_color = (5, 150, 105)
    missing_color = (209, 213, 219)
    group_w = (right - left) / len(labels)
    bar_w = group_w * 0.22

    def y_for(value: float) -> float:
        return bottom - (bottom - top) * ((value - y_min) / (y_max - y_min))

    for idx, label in enumerate(labels):
        center_x = left + group_w * (idx + 0.5)
        bx0 = center_x - bar_w - 12
        bx1 = center_x - 12
        tx0 = center_x + 12
        tx1 = center_x + bar_w + 12

        by = y_for(base_steps[idx])
        draw.rectangle((bx0, by, bx1, bottom), fill=base_color)
        draw.text((bx0 - 4, by - 24), f"{base_steps[idx]:.1f}", fill=(120, 53, 15), font=small_font)

        if trained_steps[idx] is None:
            draw.rectangle((tx0, bottom - 6, tx1, bottom), fill=missing_color)
            draw.text((tx0 - 2, bottom - 44), "n/a", fill=(107, 114, 128), font=small_font)
        else:
            ty = y_for(trained_steps[idx])
            draw.rectangle((tx0, ty, tx1, bottom), fill=trained_color)
            draw.text((tx0 - 4, ty - 24), f"{trained_steps[idx]:.1f}", fill=(6, 95, 70), font=small_font)

        tw, th = HELPER.text_size(draw, label, label_font)
        draw.text((center_x - tw / 2, bottom + 18), label, fill=(17, 24, 39), font=label_font)

    legend_x = left + 10
    legend_y = top - 84
    draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), fill=base_color)
    draw.text((legend_x + 34, legend_y - 1), "Base", fill=(31, 41, 55), font=label_font)
    lx2 = legend_x + 180
    draw.rectangle((lx2, legend_y, lx2 + 24, legend_y + 24), fill=trained_color)
    draw.text((lx2 + 34, legend_y - 1), "Trained", fill=(31, 41, 55), font=label_font)
    lx3 = lx2 + 220
    draw.rectangle((lx3, legend_y, lx3 + 24, legend_y + 24), fill=missing_color)
    draw.text((lx3 + 34, legend_y - 1), "Not reported", fill=(31, 41, 55), font=label_font)
    image.save(out_path)


def render_slide(title: str, subtitle: str, bullets: Sequence[str], panels: Sequence[Tuple[str, Path, Tuple[int, int], Tuple[int, int]]], out_path: Path) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(42, bold=True)
    subtitle_font = HELPER.find_font(22)
    body_font = HELPER.find_font(24)
    small_font = HELPER.find_font(18)

    draw.text((64, 42), title, fill=(17, 24, 39), font=title_font)
    draw.text((64, 96), subtitle, fill=(75, 85, 99), font=subtitle_font)

    bullet_y = 148
    for bullet in bullets:
        draw.text((76, bullet_y), f"- {bullet}", fill=(31, 41, 55), font=body_font)
        bullet_y += 38

    for panel_title, path, xy, size in panels:
        px, py = xy
        pw, ph = size
        draw.rounded_rectangle([px, py, px + pw, py + ph], radius=18, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
        draw.text((px + 18, py + 12), panel_title, fill=(17, 24, 39), font=subtitle_font)
        panel = Image.open(path).convert("RGB")
        panel.thumbnail((pw - 32, ph - 52))
        panel_x = px + (pw - panel.width) // 2
        panel_y = py + 42 + (ph - 52 - panel.height) // 2
        image.paste(panel, (panel_x, panel_y))

    draw.text((64, 1030), "Rollout-only probe. final_eval_bank excluded because the bank asset is currently broken.", fill=(107, 114, 128), font=small_font)
    image.save(out_path)


def write_markdown(out_path: Path) -> None:
    delta_eval = TRAINED["eval_bank"]["rate"] - BASE["eval_bank"]["rate"]
    delta_overall = TRAINED["collect_overall"]["rate"] - BASE["collect_overall"]["rate"]
    delta_steps_eval = BASE["eval_bank"]["mean_steps"] - TRAINED["eval_bank"]["mean_steps"]
    delta_steps_collect = BASE["collect_overall"]["mean_steps"] - TRAINED["collect_overall"]["mean_steps"]
    block_deltas = {
        idx: TRAINED["collect_blocks"][idx]["rate"] - BASE["collect_blocks"][idx]["rate"]
        for idx in range(1, 6)
    }
    best_block = max(block_deltas, key=block_deltas.get)
    hardest_block = min(TRAINED["collect_blocks"], key=lambda idx: TRAINED["collect_blocks"][idx]["rate"])

    lines = [
        "# O2 Probe 48-Episode Comparison",
        "",
        "## Setup",
        "",
        "- This is a rollout-only probe. No PPO updates are run here.",
        "- `final_eval_bank` is excluded because the current bank asset is invalid.",
        "- Both models are measured on the same `eval_bank` and the same collect block worlds.",
        "- Each condition uses `48` episodes.",
        "",
        "## Models",
        "",
        f"- Trained model: `{TRAINED['source_label']}`",
        f"- Base model: `{BASE['source_label']}`",
        "",
        "## Key Results",
        "",
        f"- `eval_bank`: base `{fmt_pct(BASE['eval_bank']['rate'])}` vs trained `{fmt_pct(TRAINED['eval_bank']['rate'])}` = `{fmt_pp(delta_eval)}`.",
        f"- `collect_overall`: base `{fmt_pct(BASE['collect_overall']['rate'])}` vs trained `{fmt_pct(TRAINED['collect_overall']['rate'])}` = `{fmt_pp(delta_overall)}`.",
        f"- `eval_bank` mean steps: `{BASE['eval_bank']['mean_steps']:.2f} -> {TRAINED['eval_bank']['mean_steps']:.2f}` (`-{delta_steps_eval:.2f}` steps).",
        f"- `collect_overall` mean steps: `{BASE['collect_overall']['mean_steps']:.2f} -> {TRAINED['collect_overall']['mean_steps']:.2f}` (`-{delta_steps_collect:.2f}` steps).",
        "",
        "## Block-Level Comparison",
        "",
        "| World | Base | Trained | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| eval_bank | {fmt_pct(BASE['eval_bank']['rate'])} | {fmt_pct(TRAINED['eval_bank']['rate'])} | {fmt_pp(delta_eval)} |",
    ]
    for idx in range(1, 6):
        lines.append(
            f"| block_{idx:03d} | {fmt_pct(BASE['collect_blocks'][idx]['rate'])} | {fmt_pct(TRAINED['collect_blocks'][idx]['rate'])} | {fmt_pp(block_deltas[idx])} |"
        )
    lines.extend(
        [
            f"| collect_overall | {fmt_pct(BASE['collect_overall']['rate'])} | {fmt_pct(TRAINED['collect_overall']['rate'])} | {fmt_pp(delta_overall)} |",
            "",
            "## Interpretation",
            "",
            "- The trained model is clearly stronger than the base model on held-out `eval_bank`.",
            "- The gain is not limited to one easy block: all five collect block worlds improve over base.",
            f"- `block_{best_block:03d}` shows the largest lift.",
            f"- `block_{hardest_block:03d}` remains the hardest collect world even after training.",
            "- `block_004` remains the strongest training/usefulness counterexample: it improves substantially in probe, even though it was harmful as a training block inside the main chain.",
            "",
            "## Caveat",
            "",
            "- Trained-model block-level mean-step numbers were not available in the summary, so the mean-step chart only shows trained values for `eval_bank` and `collect_overall`.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (import_root / "probe48_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_slide": out_dir / "00_probe48_summary_slide.png",
        "success_chart": out_dir / "01_probe48_success_rate_chart.png",
        "delta_chart": out_dir / "02_probe48_delta_chart.png",
        "mean_steps_chart": out_dir / "03_probe48_mean_steps_chart.png",
        "results_table": out_dir / "04_probe48_results_table.png",
        "takeaways_slide": out_dir / "05_probe48_takeaways_slide.png",
        "markdown": out_dir / "probe48_analysis.md",
    }

    render_success_rate_chart(paths["success_chart"])
    render_delta_chart(paths["delta_chart"])
    render_mean_steps_chart(paths["mean_steps_chart"])
    render_results_table(paths["results_table"])

    delta_eval = TRAINED["eval_bank"]["rate"] - BASE["eval_bank"]["rate"]
    delta_collect = TRAINED["collect_overall"]["rate"] - BASE["collect_overall"]["rate"]
    best_block = max(range(1, 6), key=lambda idx: TRAINED["collect_blocks"][idx]["rate"] - BASE["collect_blocks"][idx]["rate"])

    render_slide(
        title="O2 Probe: 48-Episode Base vs Trained Comparison",
        subtitle="Rollout-only measurement on eval_bank and collect block worlds",
        bullets=[
            f"eval_bank improves from {fmt_pct(BASE['eval_bank']['rate'])} to {fmt_pct(TRAINED['eval_bank']['rate'])} ({fmt_pp(delta_eval)}).",
            f"collect overall improves from {fmt_pct(BASE['collect_overall']['rate'])} to {fmt_pct(TRAINED['collect_overall']['rate'])} ({fmt_pp(delta_collect)}).",
            f"block_005 is still the strongest world family for the final model; largest lift is on block_{best_block:03d}.",
            "block_003 remains hard, and block_004 remains the training/usefulness counterexample.",
        ],
        panels=[
            ("Success Rate Comparison", paths["success_chart"], (60, 300), (900, 330)),
            ("Lift Over Base", paths["delta_chart"], (980, 300), (860, 330)),
            ("Results Table", paths["results_table"], (210, 660), (1500, 300)),
        ],
        out_path=paths["summary_slide"],
    )

    render_slide(
        title="O2 Probe: Takeaways",
        subtitle="What the 48-episode probe changes relative to the earlier 16-episode estimate",
        bullets=[
            "The main-chain 43.8% eval result was a peak sample; 35.4% is a more stable estimate.",
            "The final trained model is still clearly above base on held-out eval_bank.",
            "The gain is broad, not confined to a single easy collect world.",
            "final_eval_bank is intentionally excluded from this probe package.",
        ],
        panels=[
            ("Success Rate Comparison", paths["success_chart"], (60, 300), (900, 330)),
            ("Mean Steps", paths["mean_steps_chart"], (980, 300), (860, 330)),
            ("Results Table", paths["results_table"], (210, 660), (1500, 300)),
        ],
        out_path=paths["takeaways_slide"],
    )

    write_markdown(paths["markdown"])
    manifest = {key: str(path) for key, path in paths.items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
