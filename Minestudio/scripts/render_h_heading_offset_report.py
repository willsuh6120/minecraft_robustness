import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-summary", type=str, required=True)
    parser.add_argument("--chain-summary", type=str, required=True)
    parser.add_argument("--final-probe-md", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.1f}%"


def fmt_metric(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def summarize_update_metrics(train_history_path: Path) -> Optional[Dict[str, float]]:
    if not train_history_path.exists():
        return None
    history = load_json(train_history_path)
    if not history:
        return None
    last = history[-1]
    return {
        "mean_value_loss": float(last.get("mean_value_loss", 0.0) or 0.0),
        "mean_approx_kl": float(last.get("mean_approx_kl", 0.0) or 0.0),
        "mean_clip_fraction": float(last.get("mean_clip_fraction", 0.0) or 0.0),
    }


def main():
    args = parse_args()
    calibration = load_json(Path(args.calibration_summary))
    chain = load_json(Path(args.chain_summary))
    final_probe_lines = Path(args.final_probe_md).read_text(encoding="utf-8").splitlines()

    warnings: List[str] = []
    blocks = chain.get("blocks") or []
    block_rows: List[Dict[str, object]] = []
    update_metric_rows: List[Dict[str, float]] = []

    for block in blocks:
        pilot_summary_path = Path(str(block.get("pilot_summary_path") or ""))
        if not pilot_summary_path.exists():
            continue
        pilot = load_json(pilot_summary_path)
        min_success = int(((pilot.get("config") or {}).get("min_successful_fragments") or 0))

        for item in pilot.get("iteration_summaries") or []:
            collect_world = item.get("collect_world") or {}
            collect_plan_rows = collect_world.get("plan_rows") or []
            if not collect_plan_rows and collect_world.get("plan_json"):
                try:
                    collect_plan_rows = load_json(Path(str(collect_world["plan_json"])))
                except Exception:
                    collect_plan_rows = []
            variant_id = ""
            if collect_plan_rows:
                suggestions = (collect_plan_rows[0].get("world_generation_suggestions") or {})
                variant_id = str(suggestions.get("mine_heading_variant_id") or "")

            update_stats = item.get("update_episode_stats") or {}
            collect_success = int(update_stats.get("successful_episodes") or 0)
            collect_episodes = int(update_stats.get("episodes") or 0)
            if collect_success < min_success:
                warnings.append(
                    f"collect_success_below_min: block={block.get('block_tag')} iter={item.get('iteration')} "
                    f"success={collect_success}/{collect_episodes} min={min_success}"
                )

            eval_summary = item.get("eval_summary") or {}
            eval_success = None
            eval_episodes = None
            for value in (eval_summary.get("tasks") or {}).values():
                if isinstance(value, dict) and value.get("successful_episodes") is not None:
                    eval_success = int(value.get("successful_episodes") or 0)
                    eval_episodes = int(value.get("episodes") or 0)
                    break
            if eval_success is None or eval_episodes is None:
                for value in eval_summary.values():
                    if not isinstance(value, dict):
                        continue
                    episodes = value.get("episodes")
                    success_rate = value.get("auto_success_rate")
                    if episodes is None or success_rate is None:
                        continue
                    eval_episodes = int(episodes or 0)
                    eval_success = int(round(float(success_rate) * float(eval_episodes)))
                    break

            update_run_dir = Path(str(item.get("update_run_dir") or ""))
            metrics = summarize_update_metrics(update_run_dir / "train_history.json") if update_run_dir.exists() else None
            if metrics is not None:
                update_metric_rows.append(metrics)

            block_rows.append(
                {
                    "block_tag": str(block.get("block_tag") or ""),
                    "source_block_tag": str(block.get("source_block_tag") or ""),
                    "variant_id": variant_id,
                    "iteration": int(item.get("iteration") or 0),
                    "collect_success": collect_success,
                    "collect_episodes": collect_episodes,
                    "eval_success": eval_success,
                    "eval_episodes": eval_episodes,
                    "mean_value_loss": None if metrics is None else metrics["mean_value_loss"],
                    "mean_approx_kl": None if metrics is None else metrics["mean_approx_kl"],
                    "mean_clip_fraction": None if metrics is None else metrics["mean_clip_fraction"],
                }
            )

    if update_metric_rows and all(
        float(row["mean_approx_kl"]) <= 1e-4 and float(row["mean_clip_fraction"]) <= 1e-4
        for row in update_metric_rows
    ):
        warnings.append("all_updates_near_zero_kl_and_clip_fraction")
    if update_metric_rows and max(float(row["mean_value_loss"]) for row in update_metric_rows) >= 5.0:
        warnings.append("value_loss_anomalously_high")

    lines = [
        "# H Heading Offset Summary",
        "",
        f"calibration_summary: `{Path(args.calibration_summary).resolve()}`",
        f"chain_summary: `{Path(args.chain_summary).resolve()}`",
        f"final_probe: `{Path(args.final_probe_md).resolve()}`",
        "",
        "## Baseline",
        "",
        f"overall: `{int(calibration.get('overall_successful_episodes') or 0)}/{int(calibration.get('overall_episodes') or 0)}` "
        f"({fmt_pct(calibration.get('overall_success_rate'))})",
        "",
        "| idx | variant | layout | heading | sign | baseline |",
        "|---:|---|---|---:|---|---:|",
    ]

    for row in calibration.get("variants") or []:
        lines.append(
            f"| {int(row.get('instance_idx') or 0)} | {row.get('variant_id') or ''} | "
            f"{row.get('layout_case') or ''} | {int(row.get('heading_deg') or 0)} | "
            f"{row.get('heading_sign_label') or 'opp'} | "
            f"{int(row.get('successful_episodes') or 0)}/{int(row.get('episodes') or 0)} "
            f"({fmt_pct(row.get('success_rate'))}) |"
        )

    lines.extend(
        [
            "",
            "## PPO Blocks",
            "",
            "| block | source | variant | iter | collect | eval | approx_kl | clip_frac | value_loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in block_rows:
        eval_cell = ""
        if row["eval_success"] is not None and row["eval_episodes"] is not None:
            eval_rate = float(row["eval_success"]) / float(row["eval_episodes"]) if int(row["eval_episodes"]) > 0 else None
            eval_cell = f"{int(row['eval_success'])}/{int(row['eval_episodes'])} ({fmt_pct(eval_rate)})"
        collect_rate = float(row["collect_success"]) / float(row["collect_episodes"]) if int(row["collect_episodes"]) > 0 else None
        lines.append(
            f"| {row['block_tag']} | {row['source_block_tag']} | {row['variant_id']} | {int(row['iteration'])} | "
            f"{int(row['collect_success'])}/{int(row['collect_episodes'])} ({fmt_pct(collect_rate)}) | "
            f"{eval_cell} | "
            f"{fmt_metric(row['mean_approx_kl'])} | "
            f"{fmt_metric(row['mean_clip_fraction'])} | "
            f"{fmt_metric(row['mean_value_loss'])} |"
        )

    lines.extend(["", "## Final Probe", ""])
    lines.extend(final_probe_lines)

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
