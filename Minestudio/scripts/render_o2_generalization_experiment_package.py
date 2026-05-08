#!/usr/bin/env python3
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "render_o2_generalization_vm1_report.py"


def load_helper_module():
    spec = importlib.util.spec_from_file_location("vm1_report_helper", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper_module()


TRAINED_PROBE = {
    "name": "Trained Final Model Probe",
    "source_label": "VM1 o2_model_eval_and_collect_probes/20260427_194145",
    "eval_bank": {"successes": 5, "episodes": 16, "rate": 0.3125},
    "collect_blocks": {
        1: {"successes": 5, "episodes": 16, "rate": 0.3125},
        2: {"successes": 7, "episodes": 16, "rate": 0.4375},
        3: {"successes": 5, "episodes": 16, "rate": 0.3125},
        4: {"successes": 11, "episodes": 16, "rate": 0.6875},
        5: {"successes": 8, "episodes": 16, "rate": 0.5},
    },
}

BASE_CONTROL = {
    "name": "Base Model Control",
    "source_label": "VM2 o2_base_collect_eval_control_repeats/20260427_195204/seed_001",
    "notes": "n=1 repeat; post-processing crashed after seed_001.",
    "eval_bank": {"successes": 3, "episodes": 16, "rate": 0.1875, "mean_steps": 146.8, "success_seeds": [1, 9, 13]},
    "collect_blocks": {
        1: {"successes": 0, "episodes": 16, "rate": 0.0, "mean_steps": 150.0},
        2: {"successes": 5, "episodes": 16, "rate": 0.3125, "mean_steps": 132.8, "success_seeds": [5, 6, 7, 10, 12]},
        3: {"successes": 1, "episodes": 16, "rate": 0.0625, "mean_steps": 145.1},
        4: {"successes": 4, "episodes": 16, "rate": 0.25, "mean_steps": 138.3},
        5: {"successes": 5, "episodes": 16, "rate": 0.3125, "mean_steps": 144.0, "success_seeds": [1, 8, 9, 14, 15]},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a combined experiment package for VM1 O2 generalization.")
    parser.add_argument("--import-root", required=True, help="Imported VM1 run root.")
    parser.add_argument("--out-dir", default="", help="Optional output directory.")
    return parser.parse_args()


def blend(base: Tuple[int, int, int], top: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return tuple(int(round(base[i] * (1.0 - alpha) + top[i] * alpha)) for i in range(3))


def fmt_pct(rate: float) -> str:
    return f"{rate * 100.0:.1f}%"


def ensure_vm1_report(import_root: Path) -> Path:
    analysis_dir = import_root / "analysis_report"
    required = [
        analysis_dir / "01_block_eval_summary.png",
        analysis_dir / "02_collect_eval_timeline.png",
        analysis_dir / "03_update_metrics.png",
        analysis_dir / "05_eval_bank_best_matrix.png",
        analysis_dir / "06_eval_bank_full_matrix.png",
        analysis_dir / "07_final_eval_bank_matrix.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing VM1 analysis assets: {missing}")
    return analysis_dir


def copy_existing_assets(src_dir: Path, out_dir: Path) -> Dict[str, Path]:
    mapping = {
        "raw_block_eval_summary": ("01_block_eval_summary.png", "30_raw_block_eval_summary.png"),
        "raw_collect_eval_timeline": ("02_collect_eval_timeline.png", "31_raw_collect_eval_timeline.png"),
        "raw_update_metrics": ("03_update_metrics.png", "32_raw_update_metrics.png"),
        "raw_eval_bank_best_matrix": ("05_eval_bank_best_matrix.png", "33_raw_eval_bank_best_matrix.png"),
        "raw_eval_bank_full_matrix": ("06_eval_bank_full_matrix.png", "34_raw_eval_bank_full_matrix.png"),
        "raw_final_eval_matrix": ("07_final_eval_bank_matrix.png", "35_raw_final_eval_bank_matrix.png"),
    }
    out_paths: Dict[str, Path] = {}
    for key, (src_name, dst_name) in mapping.items():
        dst = out_dir / dst_name
        shutil.copy2(src_dir / src_name, dst)
        out_paths[key] = dst
    return out_paths


def render_design_table(out_path: Path) -> None:
    row_labels = ["Main Chain", "Trained Probe", "Base Control"]
    col_labels = ["Model", "Worlds", "Episodes", "Purpose"]
    cell_text = [
        [
            "PPO chain",
            "5 collect blocks\n+ shared eval_bank\n+ final_eval_bank",
            "Collect 64\nEval 16\nx 2 iters/block",
            "Train a block chain\nand select best checkpoint",
        ],
        [
            "Best model\nB5-I2",
            "eval_bank\n+ collect blocks 1..5",
            "16 per world",
            "Measure how the trained model behaves\non eval and collect worlds",
        ],
        [
            "ROCKET-2 base",
            "eval_bank\n+ collect blocks 1..5",
            "16 per world\n(seed_001 only)",
            "Separate easiness of the worlds\nfrom actual learning gain",
        ],
    ]
    cell_colors = [
        [(219, 234, 254), (239, 246, 255), (239, 246, 255), (239, 246, 255)],
        [(220, 252, 231), (240, 253, 244), (240, 253, 244), (240, 253, 244)],
        [(254, 240, 138), (254, 249, 195), (254, 249, 195), (254, 249, 195)],
    ]
    HELPER.render_table_image(
        title="Experiment Design",
        subtitle="Three layers of evidence: training chain, trained-model probe, and base-model control.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=210,
        col_width=240,
        cell_height=90,
    )


def render_main_chain_table(blocks, out_path: Path) -> None:
    row_labels = [f"Block {b.block_idx}" for b in blocks]
    col_labels = ["Collect", "Baseline", "Iter1 C", "Iter1 E", "Iter2 C", "Iter2 E", "Best", "Final"]
    cell_text = []
    cell_colors = []
    for b in blocks:
        row = [
            b.collect_pattern,
            fmt_pct(b.baseline_rate),
            f"{b.iterations[0].collect_successes}/64\n{fmt_pct(b.iterations[0].collect_rate)}",
            fmt_pct(b.iterations[0].eval_rate),
            f"{b.iterations[1].collect_successes}/64\n{fmt_pct(b.iterations[1].collect_rate)}",
            fmt_pct(b.iterations[1].eval_rate),
            fmt_pct(b.best_eval_rate),
            fmt_pct(b.final_eval_rate) if b.final_eval_rate is not None else "-",
        ]
        colors = [
            (229, 231, 235),
            (224, 242, 254),
            (255, 247, 237),
            (239, 246, 255),
            (255, 247, 237),
            (239, 246, 255),
            (219, 234, 254),
            (254, 226, 226) if b.final_eval_rate is not None else (243, 244, 246),
        ]
        cell_text.append(row)
        cell_colors.append(colors)
    HELPER.render_table_image(
        title="Main Chain Results",
        subtitle="Collect success, eval success, and final final_eval_bank result by block.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=160,
        col_width=130,
        cell_height=58,
    )


def render_probe_results_table(blocks, out_path: Path) -> None:
    row_labels = ["eval_bank"] + [f"block_{idx:03d}" for idx in range(1, 6)] + ["collect avg"]
    col_labels = ["Base", "Trained", "Delta"]
    cell_text = []
    cell_colors = []

    def add_row(base_rate: float, trained_rate: float):
        delta = trained_rate - base_rate
        delta_text = f"{delta * 100.0:+.1f}pp"
        delta_fill = (220, 252, 231) if delta >= 0 else (254, 226, 226)
        cell_text.append([fmt_pct(base_rate), fmt_pct(trained_rate), delta_text])
        cell_colors.append([(254, 249, 195), (220, 252, 231), delta_fill])

    add_row(BASE_CONTROL["eval_bank"]["rate"], TRAINED_PROBE["eval_bank"]["rate"])
    base_sum = 0.0
    trained_sum = 0.0
    for idx in range(1, 6):
        base_rate = BASE_CONTROL["collect_blocks"][idx]["rate"]
        trained_rate = TRAINED_PROBE["collect_blocks"][idx]["rate"]
        base_sum += base_rate
        trained_sum += trained_rate
        add_row(base_rate, trained_rate)
    add_row(base_sum / 5.0, trained_sum / 5.0)

    HELPER.render_table_image(
        title="Probe Comparison: Base vs Trained",
        subtitle="The final trained model is compared against ROCKET-2 base on the same eval_bank and collect worlds.",
        row_labels=row_labels,
        col_labels=col_labels,
        cell_text=cell_text,
        cell_colors=cell_colors,
        out_path=out_path,
        row_label_width=170,
        col_width=180,
        cell_height=58,
    )


def render_probe_comparison_chart(out_path: Path) -> None:
    labels = ["eval", "B1", "B2", "B3", "B4", "B5"]
    base_rates = [BASE_CONTROL["eval_bank"]["rate"]] + [BASE_CONTROL["collect_blocks"][i]["rate"] for i in range(1, 6)]
    trained_rates = [TRAINED_PROBE["eval_bank"]["rate"]] + [TRAINED_PROBE["collect_blocks"][i]["rate"] for i in range(1, 6)]

    width, height = 2200, 1100
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(42, bold=True)
    subtitle_font = HELPER.find_font(22)
    axis_font = HELPER.find_font(20)
    label_font = HELPER.find_font(18)
    small_font = HELPER.find_font(16)

    draw.text((70, 46), "Base vs Trained Model on Eval Bank and Collect Worlds", fill=(17, 24, 39), font=title_font)
    draw.text((70, 102), "Each pair compares the same world family under ROCKET-2 base and the final trained model.", fill=(75, 85, 99), font=subtitle_font)

    plot = (140, 220, 2060, 860)
    left, top, right, bottom = plot
    y_max = 0.8

    for tick_idx in range(9):
        rate = y_max * tick_idx / 8.0
        y = bottom - (bottom - top) * (rate / y_max)
        draw.line((left, y, right, y), fill=(229, 231, 235), width=2)
        label = f"{rate:.1f}"
        tw, th = HELPER.text_size(draw, label, axis_font)
        draw.text((left - tw - 16, y - th / 2), label, fill=(55, 65, 81), font=axis_font)
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
    lx2 = legend_x + 160
    draw.rectangle((lx2, legend_y, lx2 + 24, legend_y + 24), fill=trained_color)
    draw.text((lx2 + 34, legend_y - 1), "Trained", fill=(31, 41, 55), font=label_font)
    image.save(out_path)


def render_probe_lift_chart(out_path: Path) -> None:
    labels = ["eval", "B1", "B2", "B3", "B4", "B5", "avg"]
    deltas = [
        TRAINED_PROBE["eval_bank"]["rate"] - BASE_CONTROL["eval_bank"]["rate"],
        TRAINED_PROBE["collect_blocks"][1]["rate"] - BASE_CONTROL["collect_blocks"][1]["rate"],
        TRAINED_PROBE["collect_blocks"][2]["rate"] - BASE_CONTROL["collect_blocks"][2]["rate"],
        TRAINED_PROBE["collect_blocks"][3]["rate"] - BASE_CONTROL["collect_blocks"][3]["rate"],
        TRAINED_PROBE["collect_blocks"][4]["rate"] - BASE_CONTROL["collect_blocks"][4]["rate"],
        TRAINED_PROBE["collect_blocks"][5]["rate"] - BASE_CONTROL["collect_blocks"][5]["rate"],
        (
            sum(TRAINED_PROBE["collect_blocks"][i]["rate"] for i in range(1, 6))
            - sum(BASE_CONTROL["collect_blocks"][i]["rate"] for i in range(1, 6))
        )
        / 5.0,
    ]

    width, height = 2000, 980
    image = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(image)
    title_font = HELPER.find_font(40, bold=True)
    subtitle_font = HELPER.find_font(22)
    axis_font = HELPER.find_font(20)
    label_font = HELPER.find_font(18)
    small_font = HELPER.find_font(16)

    draw.text((70, 44), "Lift Over Base Model", fill=(17, 24, 39), font=title_font)
    draw.text((70, 98), "Positive bars show percentage-point gain from training.", fill=(75, 85, 99), font=subtitle_font)

    plot = (220, 180, 1900, 860)
    left, top, right, bottom = plot
    x_zero = left + 240
    max_delta = max(deltas)
    scale = (right - x_zero - 80) / max_delta

    for idx, label in enumerate(labels):
        y0 = top + idx * 88
        y_mid = y0 + 34
        draw.line((left, y_mid, right, y_mid), fill=(243, 244, 246), width=1)
        tw, th = HELPER.text_size(draw, label, label_font)
        draw.text((left + 24, y_mid - th / 2), label, fill=(17, 24, 39), font=label_font)
        bar_len = deltas[idx] * scale
        fill = (16, 185, 129) if deltas[idx] >= 0 else (220, 38, 38)
        draw.rectangle((x_zero, y_mid - 18, x_zero + bar_len, y_mid + 18), fill=fill)
        draw.text((x_zero + bar_len + 14, y_mid - 12), f"{deltas[idx] * 100.0:+.1f}pp", fill=fill, font=small_font)

    draw.line((x_zero, top - 8, x_zero, bottom + 8), fill=(55, 65, 81), width=3)
    image.save(out_path)


def render_slide(
    title: str,
    subtitle: str,
    panels: Sequence[Tuple[str, Path, Tuple[int, int], Tuple[int, int]]],
    bullets: Sequence[str],
    out_path: Path,
) -> None:
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

    draw.text((64, 1030), "All charts and tables are exported as standalone PNG files in the same directory.", fill=(107, 114, 128), font=small_font)
    image.save(out_path)


def write_markdown(blocks, out_dir: Path, assets: Dict[str, Path]) -> None:
    avg_base = sum(BASE_CONTROL["collect_blocks"][i]["rate"] for i in range(1, 6)) / 5.0
    avg_trained = sum(TRAINED_PROBE["collect_blocks"][i]["rate"] for i in range(1, 6)) / 5.0
    delta_eval = TRAINED_PROBE["eval_bank"]["rate"] - BASE_CONTROL["eval_bank"]["rate"]
    delta_collect = avg_trained - avg_base
    block5 = blocks[-1]

    lines = [
        "# O2 Generalization Experiment Package",
        "",
        "## What This Experiment Means",
        "",
        "This package combines three pieces of evidence:",
        "",
        "1. The VM1 five-block PPO generalization chain.",
        "2. A probe that runs the final trained model on `eval_bank` and on each collect block world.",
        "3. A base-model control that runs ROCKET-2 on the same `eval_bank` and on the same collect block worlds.",
        "",
        "The goal is to separate three effects:",
        "",
        "- whether the chain produced a stronger model,",
        "- whether high collect success can be explained just by an easy world,",
        "- and which occluder patterns are genuinely useful or harmful for training.",
        "",
        "## Experiment Design",
        "",
        "- **Main chain**: five collect blocks, two PPO iterations per block, `64` collect episodes and `16` eval episodes per iteration.",
        "- **Shared eval bank**: every block is judged on the same fixed `eval_bank`, so cross-block comparisons are meaningful.",
        "- **Final eval bank**: the selected best model is tested once more on `final_eval_bank` for a harder final check.",
        "- **Trained-model probe**: the final checkpoint is rolled out on `eval_bank` and each collect block world for `16` episodes.",
        "- **Base control**: the ROCKET-2 base model is rolled out on the same worlds for `16` episodes; current evidence is `n=1` repeat.",
        "",
        "## Metrics",
        "",
        "- **Collect success rate**: how often the model succeeds on the collect world used for training.",
        "- **Eval-bank success rate**: how often the model succeeds on the fixed O2 `eval_bank`.",
        "- **Final-eval success rate**: how often the selected checkpoint succeeds on the harder `final_eval_bank`.",
        "- **Approx KL / clip fraction**: how strongly PPO moved the policy during each update.",
        "- **Per-pattern success matrix**: which exact occluder patterns succeed or fail.",
        "",
        "## Main Chain Results",
        "",
        f"- Best eval checkpoint: `block_005 / iter_002` at `{fmt_pct(block5.best_eval_rate)}` on `eval_bank`.",
        f"- Final eval result: `{fmt_pct(block5.final_eval_rate or 0.0)}` on `final_eval_bank`.",
        "- Strong collect patterns: `GLG/LGL` and `LGL/GGL`.",
        "- Harmful collect pattern: `LLG/GGL`.",
        "- Collect success is not a sufficient proxy by itself: `block_003` improved on eval even while collect weakened, and `block_004` increased collect but degraded eval.",
        "",
        "### Block Summary",
        "",
        "| Block | Collect Pattern | Baseline Eval | Iter1 Collect | Iter1 Eval | Iter2 Collect | Iter2 Eval | Best Eval | Final Eval |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in blocks:
        lines.append(
            "| "
            + f"{b.block_idx} | {b.collect_pattern} | {fmt_pct(b.baseline_rate)} | "
            + f"{b.iterations[0].collect_successes}/64 ({fmt_pct(b.iterations[0].collect_rate)}) | "
            + f"{fmt_pct(b.iterations[0].eval_rate)} | "
            + f"{b.iterations[1].collect_successes}/64 ({fmt_pct(b.iterations[1].collect_rate)}) | "
            + f"{fmt_pct(b.iterations[1].eval_rate)} | "
            + f"{fmt_pct(b.best_eval_rate)} | "
            + (fmt_pct(b.final_eval_rate) if b.final_eval_rate is not None else "-")
            + " |"
        )

    lines.extend(
        [
            "",
            "## Probe and Control Results",
            "",
            f"- `eval_bank`: trained `{fmt_pct(TRAINED_PROBE['eval_bank']['rate'])}` vs base `{fmt_pct(BASE_CONTROL['eval_bank']['rate'])}` = `{delta_eval * 100.0:+.1f}pp`.",
            f"- Collect-world average: trained `{fmt_pct(avg_trained)}` vs base `{fmt_pct(avg_base)}` = `{delta_collect * 100.0:+.1f}pp`.",
            "- Improvement appears on every collect block, so the final model gain is not explained only by world easiness.",
            "",
            "| World | Base | Trained | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| eval_bank | {fmt_pct(BASE_CONTROL['eval_bank']['rate'])} | {fmt_pct(TRAINED_PROBE['eval_bank']['rate'])} | {(TRAINED_PROBE['eval_bank']['rate'] - BASE_CONTROL['eval_bank']['rate']) * 100.0:+.1f}pp |",
        ]
    )
    for idx in range(1, 6):
        base_rate = BASE_CONTROL["collect_blocks"][idx]["rate"]
        trained_rate = TRAINED_PROBE["collect_blocks"][idx]["rate"]
        lines.append(
            f"| block_{idx:03d} | {fmt_pct(base_rate)} | {fmt_pct(trained_rate)} | {(trained_rate - base_rate) * 100.0:+.1f}pp |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main chain produced a real model improvement, not just a lucky eval sample.",
            "- `block_005` is the strongest evidence: it was the winning training block in the chain and still shows a clear lift over base in the probe.",
            "- `block_004` is the most interesting counterexample: the final model can solve the block-4 world much better than base, but training on block 4 at that point in the chain was harmful. This means the issue is not only world difficulty; update direction and chain order matter.",
            "- The control run also shows strong pattern sensitivity. Average difficulty may match `eval_bank`, but internal occluder structure is not uniform.",
            "",
            "## Caveats",
            "",
            "- The trained probe and the base control are both `16`-episode measurements, so point estimates still have variance.",
            "- The base control is currently only `seed_001` because post-processing failed after the first repeat.",
            "- The probe/control raw outputs were not imported locally for this package; probe sections use the recorded summary values from the executed runs.",
            "",
            "## Generated Assets",
            "",
        ]
    )
    for key, path in assets.items():
        lines.append(f"- `{path.name}`: {key}")

    (out_dir / "experiment_package.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import_root = Path(args.import_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (import_root / "experiment_package")
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = ensure_vm1_report(import_root)
    copied_assets = copy_existing_assets(analysis_dir, out_dir)
    blocks = HELPER.discover_blocks(import_root)

    asset_paths = {
        "design_table": out_dir / "10_experiment_design_table.png",
        "main_chain_table": out_dir / "11_main_chain_results_table.png",
        "probe_results_table": out_dir / "12_probe_results_table.png",
        "probe_comparison_chart": out_dir / "13_probe_comparison_chart.png",
        "probe_lift_chart": out_dir / "14_probe_lift_chart.png",
        "slide_exec": out_dir / "00_executive_summary_slide.png",
        "slide_design": out_dir / "01_experiment_design_slide.png",
        "slide_chain": out_dir / "02_main_chain_results_slide.png",
        "slide_probe": out_dir / "03_probe_vs_base_slide.png",
        "slide_patterns": out_dir / "04_pattern_takeaways_slide.png",
    }

    render_design_table(asset_paths["design_table"])
    render_main_chain_table(blocks, asset_paths["main_chain_table"])
    render_probe_results_table(blocks, asset_paths["probe_results_table"])
    render_probe_comparison_chart(asset_paths["probe_comparison_chart"])
    render_probe_lift_chart(asset_paths["probe_lift_chart"])

    avg_base = sum(BASE_CONTROL["collect_blocks"][i]["rate"] for i in range(1, 6)) / 5.0
    avg_trained = sum(TRAINED_PROBE["collect_blocks"][i]["rate"] for i in range(1, 6)) / 5.0

    render_slide(
        "O2 Generalization: Executive Summary",
        "Main training chain + trained-model probe + base-model control",
        [
            ("Main Chain by Block", copied_assets["raw_block_eval_summary"], (60, 320), (840, 340)),
            ("Probe Comparison", asset_paths["probe_comparison_chart"], (980, 320), (860, 340)),
        ],
        [
            f"Best main-chain checkpoint: Block 5 Iter 2 = {fmt_pct(blocks[-1].best_eval_rate)} on eval_bank.",
            f"Final eval result: {fmt_pct(blocks[-1].final_eval_rate or 0.0)} on final_eval_bank.",
            f"Probe lift on eval_bank: {fmt_pct(TRAINED_PROBE['eval_bank']['rate'])} vs {fmt_pct(BASE_CONTROL['eval_bank']['rate'])} = {(TRAINED_PROBE['eval_bank']['rate'] - BASE_CONTROL['eval_bank']['rate']) * 100.0:+.1f}pp.",
            f"Probe lift on collect blocks average: {fmt_pct(avg_trained)} vs {fmt_pct(avg_base)} = {(avg_trained - avg_base) * 100.0:+.1f}pp.",
        ],
        asset_paths["slide_exec"],
    )

    render_slide(
        "Experiment Design",
        "What was trained, what was probed, and what each metric means.",
        [("Design Table", asset_paths["design_table"], (140, 290), (1640, 620))],
        [
            "The chain uses five collect blocks and a shared fixed eval_bank, so cross-block comparisons are meaningful.",
            "The trained probe asks whether the final checkpoint really improved on the same worlds.",
            "The base control asks whether high collect success can be explained only by easy worlds.",
        ],
        asset_paths["slide_design"],
    )

    render_slide(
        "Main Chain Results",
        "How the five-block training chain behaved before any probe/control comparison.",
        [
            ("Collect vs Eval Timeline", copied_assets["raw_collect_eval_timeline"], (60, 300), (900, 330)),
            ("Block Summary Table", asset_paths["main_chain_table"], (1000, 300), (860, 330)),
            ("Update Metrics", copied_assets["raw_update_metrics"], (60, 670), (1800, 330)),
        ],
        [
            "Block 5 was the recovery and winning block.",
            "Block 4 is the strongest negative counterexample: collect rose while eval fell.",
            "Collect success alone is not a sufficient proxy for generalization.",
        ],
        asset_paths["slide_chain"],
    )

    render_slide(
        "Trained Model vs Base Control",
        "The final trained checkpoint is compared against ROCKET-2 base on the same worlds.",
        [
            ("Probe Comparison", asset_paths["probe_comparison_chart"], (60, 300), (900, 320)),
            ("Lift Over Base", asset_paths["probe_lift_chart"], (980, 300), (860, 320)),
            ("Probe Results Table", asset_paths["probe_results_table"], (280, 660), (1360, 300)),
        ],
        [
            "The trained model improves over base on eval_bank and on every collect block world.",
            "Block 5 remains strong outside the original training loop, which supports a real learning effect.",
            "Block 4 is solvable by the final model even though training on block 4 at that stage was harmful.",
        ],
        asset_paths["slide_probe"],
    )

    render_slide(
        "Pattern-Level Takeaways",
        "Success/failure is highly pattern-specific; average success rate hides large internal differences.",
        [
            ("Eval Bank Best Matrix", copied_assets["raw_eval_bank_best_matrix"], (40, 300), (900, 340)),
            ("Eval Bank Full Matrix", copied_assets["raw_eval_bank_full_matrix"], (980, 300), (900, 340)),
            ("Final Eval Matrix", copied_assets["raw_final_eval_matrix"], (520, 680), (880, 300)),
        ],
        [
            "The detailed matrices show which fixed-bank patterns flipped from failure to success over the chain.",
            "Final eval still succeeds on a nontrivial subset of the harder final_eval_bank.",
            "The control run confirms that occluder layout sensitivity is a real property of the base model.",
        ],
        asset_paths["slide_patterns"],
    )

    all_assets = {}
    all_assets.update(copied_assets)
    all_assets.update(asset_paths)
    write_markdown(blocks, out_dir, all_assets)
    all_assets["markdown"] = out_dir / "experiment_package.md"

    manifest = {key: str(path) for key, path in all_assets.items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
