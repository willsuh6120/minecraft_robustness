import argparse
import csv
from collections import Counter, defaultdict
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    resolve_interaction_task_specs,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    classify_factor_split,
    hard_factor_count,
    normalize_worldgen_suggestions,
    derive_factor_levels_from_review,
    empty_factor_levels,
    factor_levels_to_worldgen_suggestions,
    normalize_factor_levels,
    project_factor_levels_to_split,
    sample_factor_levels_for_split,
    world_factor_schema_for_task,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_vlm_review import (
    aggregate_worldgen_plan,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-group", type=str, default="rocket2_official")
    parser.add_argument("--task-group-path", type=str, default="/home/gyulab/envgen2/ROCKET-2/env_conf")
    parser.add_argument("--worldgen-source-dir", type=str, default="")
    parser.add_argument(
        "--mine-layout-backend",
        type=str,
        default="procedural",
        choices=["auto", "template", "procedural"],
    )
    parser.add_argument(
        "--mine-anchor-mode",
        type=str,
        default="source_or_fallback",
        choices=["source_or_fallback", "procedural_only"],
    )
    parser.add_argument("--env-source", type=str, default="rocket2_official", choices=["rocket2_official"])
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--tasks", type=str, required=True)
    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal-spec", type=str)
    goal_group.add_argument("--auto-goal", action="store_true")
    parser.add_argument("--eval-episodes-per-task", type=int, default=5)
    parser.add_argument("--collect-episodes-per-task", type=int, default=5)
    parser.add_argument("--collect-world-instances", type=int, default=1)
    parser.add_argument("--eval-world-instances", type=int, default=1)
    parser.add_argument("--final-eval-world-instances", type=int, default=0)
    parser.add_argument("--collect-parallel-workers", type=int, default=1)
    parser.add_argument("--eval-parallel-workers", type=int, default=1)
    parser.add_argument("--final-eval-parallel-workers", type=int, default=0)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=0)
    parser.add_argument("--sampling-base-seed", type=int, default=None)
    parser.add_argument("--sampling-seed-step", type=int, default=None)
    parser.add_argument("--episode-retries", type=int, default=1)
    parser.add_argument("--step-budget-override", type=int, default=0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--collect-video-mode",
        type=str,
        default="inherit",
        choices=["inherit", "skip", "save"],
    )
    parser.add_argument(
        "--baseline-video-mode",
        type=str,
        default="inherit",
        choices=["inherit", "skip", "save"],
    )
    parser.add_argument(
        "--eval-video-mode",
        type=str,
        default="inherit",
        choices=["inherit", "skip", "save"],
    )
    parser.add_argument(
        "--final-eval-video-mode",
        type=str,
        default="inherit",
        choices=["inherit", "skip", "save"],
    )
    parser.add_argument("--save-debug-assets", action="store_true")
    parser.add_argument("--model-path", type=str, default="hf:phython96/ROCKET-2-1x-22w")
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument("--env-reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg-policy-mode",
        type=str,
        default="full",
        choices=["full", "frozen_base"],
    )
    parser.add_argument("--cfg-base-ref-model-path", type=str, default="")
    parser.add_argument("--kl-anchor-model-path", type=str, default="")
    parser.add_argument("--bake-generated-goals", action="store_true")
    parser.add_argument("--bake-goal-base-seed", type=int, default=1)
    parser.add_argument("--bake-goal-episode-retries", type=int, default=2)
    parser.add_argument("--bake-goal-save-debug-assets", action="store_true")
    parser.add_argument(
        "--collect-world-mode",
        type=str,
        default="static",
        choices=["static", "random", "curriculum", "plan", "plan_exact", "targeted", "fixed_bank"],
    )
    parser.add_argument(
        "--eval-world-mode",
        type=str,
        default="static",
        choices=["static", "random", "curriculum", "plan", "plan_exact", "targeted", "fixed_bank"],
    )
    parser.add_argument(
        "--final-eval-world-mode",
        type=str,
        default="",
        choices=["", "static", "random", "curriculum", "plan", "plan_exact", "targeted", "fixed_bank"],
    )
    parser.add_argument("--worldgen-plan-json", type=str, default="")
    parser.add_argument("--collect-resample-plan-layout-seed", action="store_true")
    parser.add_argument("--worldgen-out-dir", type=str, default="")
    parser.add_argument("--collect-fixed-bank-dir", type=str, default="")
    parser.add_argument("--eval-fixed-bank-dir", type=str, default="")
    parser.add_argument("--final-eval-fixed-bank-dir", type=str, default="")
    parser.add_argument("--random-max-increased-factors", type=int, default=2)
    parser.add_argument(
        "--collect-factor-split",
        type=str,
        default="train_id",
        choices=["clean", "train_id", "ood", "stress"],
    )
    parser.add_argument(
        "--eval-factor-split",
        type=str,
        default="train_id",
        choices=["clean", "train_id", "ood", "stress"],
    )
    parser.add_argument(
        "--final-eval-factor-split",
        type=str,
        default="",
        choices=["", "clean", "train_id", "ood", "stress"],
    )
    parser.add_argument(
        "--targeted-review-backend",
        type=str,
        default="heuristic",
        choices=["heuristic", "openai"],
    )
    parser.add_argument("--targeted-review-model-id", type=str, default="gpt-4o")
    parser.add_argument(
        "--targeted-review-backend-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--targeted-review-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument("--targeted-review-max-mid-images", type=int, default=3)
    parser.add_argument(
        "--targeted-review-max-reprompt-images",
        type=int,
        dest="targeted_review_max_mid_images",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--targeted-review-overwrite", action="store_true")
    parser.add_argument("--targeted-bootstrap-on-success", action="store_true")
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--policy-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--update-fragment-batch-size", type=int, default=1)
    parser.add_argument(
        "--loss-focus-mode",
        type=str,
        default="uniform",
        choices=["uniform", "suffix_success", "prefix_success", "advantage_top_k", "lateral_top_k", "corridor_commit"],
    )
    parser.add_argument("--loss-focus-suffix-len", type=int, default=32)
    parser.add_argument("--loss-focus-context-len", type=int, default=96)
    parser.add_argument("--loss-focus-weight", type=float, default=4.0)
    parser.add_argument("--loss-focus-top-k", type=int, default=32)
    parser.add_argument(
        "--trainable-scope",
        type=str,
        default="heads",
        choices=["value", "heads", "heads_last", "crossview_small", "full_except_vision"],
    )
    parser.add_argument("--vf-warmup-iters", type=int, default=0)
    parser.add_argument("--zero-initial-vf", action="store_true")
    parser.add_argument("--calibrate-value-normalizer", action="store_true")
    parser.add_argument(
        "--final-eval-model-mode",
        type=str,
        default="last",
        choices=["last", "best_eval"],
    )
    parser.add_argument("--min-successful-fragments", type=int, default=0)
    parser.add_argument(
        "--collect-quota-mode",
        type=str,
        default="none",
        choices=["none", "success_failure_per_instance"],
        help="Optionally filter collect fragments before PPO update.",
    )
    parser.add_argument(
        "--collect-quota-successes-per-instance",
        type=int,
        default=0,
        help="For success_failure_per_instance, target this many successful fragments per collect world.",
    )
    parser.add_argument(
        "--collect-quota-failures-per-instance",
        type=int,
        default=0,
        help="For success_failure_per_instance, target this many failed fragments per collect world.",
    )
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-iteration-eval", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/interaction_crossview_ppo_pilot")
    parser.add_argument("--resume-pilot-dir", type=str, default="")
    return parser.parse_args()


def latest_subdir(path: Path) -> Path:
    children = sorted([child for child in path.iterdir() if child.is_dir()])
    if not children:
        raise RuntimeError(f"No run directories found under {path}")
    return children[-1]


def resolve_cfg_base_ref_model_path(args: argparse.Namespace) -> str:
    if str(args.cfg_policy_mode or "full") != "frozen_base":
        return ""
    explicit_path = str(args.cfg_base_ref_model_path or "").strip()
    if explicit_path:
        return explicit_path
    return str(args.model_path)


def resolve_kl_anchor_model_path(args: argparse.Namespace) -> str:
    explicit_path = str(args.kl_anchor_model_path or "").strip()
    if explicit_path:
        return explicit_path
    return str(args.model_path)


def resolve_sampling_seed_step(args: argparse.Namespace) -> Optional[int]:
    if args.sampling_base_seed is None:
        return None
    if args.sampling_seed_step is None:
        return int(args.seed_step)
    return int(args.sampling_seed_step)


def resolve_phase_skip_video(args: argparse.Namespace, phase_label: str) -> bool:
    label = str(phase_label or "").strip().lower()
    video_mode = "inherit"
    if "collect" in label:
        video_mode = str(args.collect_video_mode or "inherit").strip().lower()
    elif label == "baseline_eval":
        video_mode = str(args.baseline_video_mode or "inherit").strip().lower()
    elif label == "final_eval":
        video_mode = str(args.final_eval_video_mode or "inherit").strip().lower()
    else:
        video_mode = str(args.eval_video_mode or "inherit").strip().lower()

    if video_mode == "skip":
        return True
    if video_mode == "save":
        return False
    return bool(args.skip_video)


def run_command(cmd: List[str], env):
    print("[interaction-crossview-ppo-pilot] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def start_command(cmd: List[str], env):
    print("[interaction-crossview-ppo-pilot] starting:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def format_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, seconds = divmod(int(round(total)), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def planned_rollout_episodes_total(args: argparse.Namespace) -> int:
    baseline = 0 if bool(getattr(args, "skip_baseline_eval", False)) else int(args.eval_episodes_per_task)
    per_iteration = int(args.collect_episodes_per_task)
    if not bool(getattr(args, "skip_iteration_eval", False)):
        per_iteration += int(args.eval_episodes_per_task)
    final_eval = 0 if bool(args.skip_final_eval) else int(args.eval_episodes_per_task)
    return baseline + max(0, int(args.train_iters)) * per_iteration + final_eval


def redacted_args_dict(args: argparse.Namespace) -> Dict[str, object]:
    data = dict(vars(args))
    if str(data.get("targeted_review_api_key", "")).strip():
        data["targeted_review_api_key"] = "***REDACTED***"
    return data


def load_summary(summary_json_path: Path) -> Dict[str, Dict]:
    rows = json.loads(summary_json_path.read_text(encoding="utf-8"))
    return {row["benchmark_name"]: row for row in rows}


def merge_summaries(summary_dicts: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for summary in summary_dicts:
        for benchmark_name, row in summary.items():
            grouped[str(benchmark_name)].append(row)

    merged: Dict[str, Dict] = {}
    for benchmark_name, rows in grouped.items():
        total_episodes = sum(int(row.get("episodes", 0) or 0) for row in rows)
        total_reward_supported = sum(int(row.get("reward_supported_episodes", 0) or 0) for row in rows)
        total_trainable = sum(int(row.get("trainable_episodes", 0) or 0) for row in rows)
        total_auto_eval_supported = sum(int(row.get("auto_eval_supported_episodes", 0) or 0) for row in rows)
        total_success = sum(
            float(row.get("auto_success_rate", 0.0) or 0.0) * float(int(row.get("auto_eval_supported_episodes", 0) or 0))
            for row in rows
        )
        total_reward = sum(
            float(row.get("mean_reward", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0))
            for row in rows
        )
        total_steps = sum(
            float(row.get("mean_steps", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0))
            for row in rows
        )
        total_core_wall_time = sum(
            float(row.get("mean_core_episode_wall_time_sec", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0))
            for row in rows
        )
        total_episode_wall_time = sum(
            float(row.get("mean_episode_wall_time_sec", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0))
            for row in rows
        )
        total_video_write_wall_time = sum(
            float(row.get("mean_video_write_wall_time_sec", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0))
            for row in rows
        )
        merged[benchmark_name] = {
            "benchmark_name": benchmark_name,
            "episodes": total_episodes,
            "reward_supported_episodes": total_reward_supported,
            "trainable_episodes": total_trainable,
            "auto_eval_supported_episodes": total_auto_eval_supported,
            "auto_success_rate": (total_success / float(total_auto_eval_supported)) if total_auto_eval_supported > 0 else 0.0,
            "mean_reward": (total_reward / float(total_episodes)) if total_episodes > 0 else 0.0,
            "mean_steps": (total_steps / float(total_episodes)) if total_episodes > 0 else 0.0,
            "mean_core_episode_wall_time_sec": (total_core_wall_time / float(total_episodes)) if total_episodes > 0 else 0.0,
            "mean_episode_wall_time_sec": (total_episode_wall_time / float(total_episodes)) if total_episodes > 0 else 0.0,
            "mean_video_write_wall_time_sec": (total_video_write_wall_time / float(total_episodes)) if total_episodes > 0 else 0.0,
        }
    return merged


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_task_names(tasks: str) -> List[str]:
    return [task.strip() for task in str(tasks).split(",") if task.strip()]


def sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def find_rollout_run_dirs(root: Path) -> List[Path]:
    if (root / "summary.json").exists():
        return [root]
    run_dirs: List[Path] = []
    seen = set()
    for summary_path in sorted(root.rglob("summary.json")):
        run_dir = summary_path.parent
        resolved = str(run_dir.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        run_dirs.append(run_dir)
    return run_dirs


def load_aggregated_summary(run_root: Path) -> Dict[str, Dict]:
    run_dirs = find_rollout_run_dirs(run_root)
    if not run_dirs:
        raise RuntimeError(f"No rollout summary.json files found under {run_root}")
    return merge_summaries([load_summary(run_dir / "summary.json") for run_dir in run_dirs])


def summarize_summary_metrics(summary: Dict[str, Dict]) -> Dict[str, float]:
    total_episodes = 0
    total_auto_eval_supported = 0
    weighted_success = 0.0
    weighted_reward = 0.0
    for row in summary.values():
        episodes = int(row.get("episodes", 0) or 0)
        auto_eval_supported = int(row.get("auto_eval_supported_episodes", 0) or 0)
        total_episodes += episodes
        total_auto_eval_supported += auto_eval_supported
        weighted_success += float(row.get("auto_success_rate", 0.0) or 0.0) * float(auto_eval_supported)
        weighted_reward += float(row.get("mean_reward", 0.0) or 0.0) * float(episodes)
    return {
        "auto_success_rate": (weighted_success / float(total_auto_eval_supported)) if total_auto_eval_supported > 0 else 0.0,
        "mean_reward": (weighted_reward / float(total_episodes)) if total_episodes > 0 else 0.0,
    }


def select_best_eval_model_selection(iteration_summaries: List[Dict], fallback_model_path: str) -> Dict[str, object]:
    best_key = None
    best_selection = None
    for item in iteration_summaries:
        model_path = str(item.get("updated_model_path", "")).strip()
        eval_summary = item.get("eval_summary") or {}
        if not model_path or not eval_summary:
            continue
        metrics = summarize_summary_metrics(eval_summary)
        key = (
            float(metrics["auto_success_rate"]),
            float(metrics["mean_reward"]),
            int(item.get("iteration", 0) or 0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_selection = {
                "mode": "best_eval",
                "model_path": model_path,
                "iteration": int(item.get("iteration", 0) or 0),
                "auto_success_rate": float(metrics["auto_success_rate"]),
                "mean_reward": float(metrics["mean_reward"]),
            }
    if best_selection is not None:
        return best_selection
    last_iteration = int(iteration_summaries[-1].get("iteration", 0) or 0) if iteration_summaries else 0
    return {
        "mode": "best_eval_fallback_last",
        "model_path": str(fallback_model_path),
        "iteration": last_iteration,
        "auto_success_rate": 0.0,
        "mean_reward": 0.0,
    }


def select_final_eval_model_selection(mode: str, iteration_summaries: List[Dict], current_model_path: str) -> Dict[str, object]:
    if str(mode) == "best_eval":
        return select_best_eval_model_selection(iteration_summaries, fallback_model_path=current_model_path)
    last_iteration = int(iteration_summaries[-1].get("iteration", 0) or 0) if iteration_summaries else 0
    return {
        "mode": "last",
        "model_path": str(current_model_path),
        "iteration": last_iteration,
    }


def load_pilot_state(out_root: Path) -> Dict[str, Any]:
    state_path = out_root / "pilot_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _slim_world_record(world: Any) -> Any:
    """Strip instance_worlds bulk from a world record for compact disk storage."""
    if not isinstance(world, dict):
        return world
    return {k: v for k, v in world.items() if k != "instance_worlds"}


def _slim_phase_timing(timing: Any) -> Any:
    """Strip per-instance instance_details from a phase timing record."""
    if not isinstance(timing, dict):
        return timing
    return {k: v for k, v in timing.items() if k != "instance_details"}


def _slim_state_for_disk(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copy of state with bulky redundant fields stripped."""
    s = dict(state)
    for key in ("baseline_world", "final_eval_world"):
        if isinstance(s.get(key), dict):
            s[key] = _slim_world_record(s[key])
    timing = s.get("timing")
    if isinstance(timing, dict):
        new_timing = dict(timing)
        for tkey in ("baseline_phase", "final_eval_phase"):
            if isinstance(new_timing.get(tkey), dict):
                new_timing[tkey] = _slim_phase_timing(new_timing[tkey])
        s["timing"] = new_timing
    new_iters = []
    for it in s.get("iteration_summaries") or []:
        it = dict(it)
        for wkey in ("collect_world", "eval_world"):
            if isinstance(it.get(wkey), dict):
                it[wkey] = _slim_world_record(it[wkey])
        for tkey in ("collect_phase_timing", "eval_phase_timing"):
            if isinstance(it.get(tkey), dict):
                it[tkey] = _slim_phase_timing(it[tkey])
        new_iters.append(it)
    s["iteration_summaries"] = new_iters
    return s


def write_pilot_state(out_root: Path, state: Dict[str, Any]) -> None:
    state_path = out_root / "pilot_state.json"
    state_path.write_text(
        json.dumps(normalize_jsonable(_slim_state_for_disk(state)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def collect_rollout_episode_rows(run_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run_dir in find_rollout_run_dirs(run_root):
        episodes_csv = run_dir / "episodes.csv"
        if not episodes_csv.exists():
            continue
        with episodes_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def summarize_rollout_episode_rows(run_root: Path) -> Dict[str, float]:
    rows = collect_rollout_episode_rows(run_root)
    total = len(rows)
    successful = sum(1 for row in rows if str(row.get("auto_success", "")).strip().lower() == "true")
    reward_positive = 0
    for row in rows:
        try:
            reward_positive += 1 if float(row.get("episode_reward", 0.0) or 0.0) > 0.0 else 0
        except Exception:
            continue
    success_rate = (float(successful) / float(total)) if total > 0 else 0.0
    return {
        "episodes": int(total),
        "successful_episodes": int(successful),
        "reward_positive_episodes": int(reward_positive),
        "success_rate": float(success_rate),
    }


def parse_success_value(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "success"}


def write_simple_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_fragment_records_for_run_dir(run_dir: Path, instance_idx: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for fragment_path in sorted(run_dir.glob("*/seed_*_ep_*/episode_fragment.pt")):
        episode_dir = fragment_path.parent
        result_path = episode_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        success = parse_success_value(result.get("auto_success"))
        reward = result.get("episode_reward", 0.0)
        try:
            reward_value = float(reward or 0.0)
        except Exception:
            reward_value = 0.0
        records.append(
            {
                "instance_idx": int(instance_idx),
                "run_dir": str(run_dir),
                "episode_dir": str(episode_dir),
                "fragment_path": str(fragment_path),
                "result_path": str(result_path),
                "task_config_name": str(result.get("task_config_name", "")),
                "episode_index": int(result.get("episode_index", len(records)) or 0),
                "seed": int(result.get("seed", 0) or 0),
                "sampling_seed": result.get("sampling_seed"),
                "num_steps": int(result.get("num_steps", 0) or 0),
                "stop_reason": str(result.get("stop_reason", "")),
                "auto_success": bool(success),
                "episode_reward": float(reward_value),
            }
        )
    return records


def copy_fragment_record_to_quota_dir(record: Dict[str, object], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    src_fragment = Path(str(record["fragment_path"]))
    dst_fragment = destination_dir / "episode_fragment.pt"
    try:
        os.symlink(src_fragment, dst_fragment)
    except Exception:
        shutil.copy2(src_fragment, dst_fragment)

    src_result = Path(str(record.get("result_path") or ""))
    if src_result.exists():
        shutil.copy2(src_result, destination_dir / "result.json")
    (destination_dir / "source_episode.json").write_text(
        json.dumps(normalize_jsonable(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_collect_quota_update_input(
    args: argparse.Namespace,
    collect_root: Path,
    collect_phase: Dict[str, object],
    iteration_idx: int,
) -> tuple[Path, Dict[str, object]]:
    mode = str(args.collect_quota_mode or "none").strip()
    if mode == "none":
        return collect_root, {}
    if mode != "success_failure_per_instance":
        raise ValueError(f"Unsupported collect quota mode: {mode}")

    success_quota = max(0, int(args.collect_quota_successes_per_instance))
    failure_quota = max(0, int(args.collect_quota_failures_per_instance))
    target_per_instance = success_quota + failure_quota
    if target_per_instance <= 0:
        return collect_root, {}

    quota_root = collect_root / "_quota_balanced_update_input"
    if quota_root.exists():
        shutil.rmtree(quota_root)
    quota_root.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(str(path)) for path in (collect_phase.get("run_dirs") or [])]
    if not run_dirs:
        run_dirs = find_rollout_run_dirs(collect_root)

    selected_records: List[Dict[str, object]] = []
    per_instance: List[Dict[str, object]] = []
    for instance_idx, run_dir in enumerate(run_dirs):
        records = collect_fragment_records_for_run_dir(run_dir, instance_idx=instance_idx)
        successes = [record for record in records if bool(record["auto_success"])]
        failures = [record for record in records if not bool(record["auto_success"])]
        selected = list(successes[:success_quota]) + list(failures[:failure_quota])
        selected_keys = {str(record["fragment_path"]) for record in selected}

        if len(selected) < target_per_instance:
            for record in failures[failure_quota:] + successes[success_quota:]:
                key = str(record["fragment_path"])
                if key in selected_keys:
                    continue
                selected.append(record)
                selected_keys.add(key)
                if len(selected) >= target_per_instance:
                    break

        selected_successes = sum(1 for record in selected if bool(record["auto_success"]))
        selected_failures = len(selected) - selected_successes
        for selected_idx, record in enumerate(selected):
            label = "success" if bool(record["auto_success"]) else "failure"
            source_episode = sanitize_token(Path(str(record["episode_dir"])).name)
            destination_dir = quota_root / f"instance_{instance_idx:03d}" / f"{selected_idx:03d}_{label}_{source_episode}"
            record = dict(record)
            record["quota_instance_idx"] = int(instance_idx)
            record["quota_selected_idx"] = int(selected_idx)
            record["quota_label"] = label
            record["quota_destination_dir"] = str(destination_dir)
            copy_fragment_record_to_quota_dir(record, destination_dir)
            selected_records.append(record)

        per_instance.append(
            {
                "instance_idx": int(instance_idx),
                "run_dir": str(run_dir),
                "available_episodes": int(len(records)),
                "available_successes": int(len(successes)),
                "available_failures": int(len(failures)),
                "requested_successes": int(success_quota),
                "requested_failures": int(failure_quota),
                "selected_episodes": int(len(selected)),
                "selected_successes": int(selected_successes),
                "selected_failures": int(selected_failures),
                "success_quota_met": bool(selected_successes >= success_quota),
                "failure_quota_met": bool(selected_failures >= failure_quota),
            }
        )

    selected_total = len(selected_records)
    selected_success_total = sum(1 for record in selected_records if bool(record["auto_success"]))
    selected_reward_positive = sum(1 for record in selected_records if float(record.get("episode_reward", 0.0) or 0.0) > 0.0)
    selected_stats = {
        "episodes": int(selected_total),
        "successful_episodes": int(selected_success_total),
        "reward_positive_episodes": int(selected_reward_positive),
        "success_rate": (float(selected_success_total) / float(selected_total)) if selected_total > 0 else 0.0,
    }
    if selected_total <= 0:
        raise RuntimeError(f"Collect quota selected no trainable fragments under {collect_root}")

    episode_rows = [
        {
            "instance_idx": int(record["instance_idx"]),
            "quota_instance_idx": int(record["quota_instance_idx"]),
            "quota_selected_idx": int(record["quota_selected_idx"]),
            "task_config_name": str(record.get("task_config_name", "")),
            "episode_index": int(record.get("episode_index", 0) or 0),
            "seed": int(record.get("seed", 0) or 0),
            "sampling_seed": record.get("sampling_seed"),
            "num_steps": int(record.get("num_steps", 0) or 0),
            "auto_success": "true" if bool(record.get("auto_success")) else "false",
            "episode_reward": float(record.get("episode_reward", 0.0) or 0.0),
            "source_episode_dir": str(record.get("episode_dir", "")),
            "quota_destination_dir": str(record.get("quota_destination_dir", "")),
        }
        for record in selected_records
    ]
    write_simple_csv(quota_root / "episodes.csv", episode_rows)
    summary_rows = [
        {
            "benchmark_name": "quota_balanced_collect",
            "episodes": int(selected_total),
            "reward_supported_episodes": int(selected_total),
            "trainable_episodes": int(selected_total),
            "auto_eval_supported_episodes": int(selected_total),
            "auto_success_rate": (float(selected_success_total) / float(selected_total)) if selected_total > 0 else 0.0,
            "mean_reward": (
                sum(float(record.get("episode_reward", 0.0) or 0.0) for record in selected_records) / float(selected_total)
            )
            if selected_total > 0
            else 0.0,
            "mean_steps": (
                sum(float(record.get("num_steps", 0) or 0) for record in selected_records) / float(selected_total)
            )
            if selected_total > 0
            else 0.0,
        }
    ]
    (quota_root / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "mode": mode,
        "iteration": int(iteration_idx),
        "collect_root": str(collect_root),
        "quota_root": str(quota_root),
        "requested_successes_per_instance": int(success_quota),
        "requested_failures_per_instance": int(failure_quota),
        "target_per_instance": int(target_per_instance),
        "instances": int(len(per_instance)),
        "selected_episode_stats": selected_stats,
        "quota_instances_success_met": int(sum(1 for row in per_instance if bool(row["success_quota_met"]))),
        "quota_instances_failure_met": int(sum(1 for row in per_instance if bool(row["failure_quota_met"]))),
        "per_instance": per_instance,
        "selected_records": selected_records,
    }
    (quota_root / "quota_manifest.json").write_text(
        json.dumps(normalize_jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "collect_quota_selected",
                "iteration_idx": int(iteration_idx),
                "mode": mode,
                "quota_root": str(quota_root),
                "selected_episodes": int(selected_total),
                "selected_successes": int(selected_success_total),
                "requested_successes_per_instance": int(success_quota),
                "requested_failures_per_instance": int(failure_quota),
                "quota_instances_success_met": int(manifest["quota_instances_success_met"]),
                "quota_instances_failure_met": int(manifest["quota_instances_failure_met"]),
            },
            ensure_ascii=False,
        ),
    )
    return quota_root, manifest


def rollout_phase_complete(run_root: Path, task_count: int, episodes_per_task: int) -> bool:
    if not run_root.exists():
        return False
    expected_episodes = max(0, int(task_count)) * max(0, int(episodes_per_task))
    if expected_episodes <= 0:
        return False
    try:
        stats = summarize_rollout_episode_rows(run_root)
    except Exception:
        return False
    return int(stats.get("episodes", 0) or 0) == expected_episodes


def has_phase_artifacts(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def archive_incomplete_phase_root(path: Path, tag: str = "resume_backup") -> Optional[Path]:
    if not has_phase_artifacts(path):
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.parent / f"{path.name}__{tag}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = path.parent / f"{path.name}__{tag}_{timestamp}_{suffix:02d}"
        suffix += 1
    path.rename(candidate)
    return candidate


def latest_model_checkpoint(update_root: Path) -> Optional[Path]:
    if not update_root.exists():
        return None
    candidates = sorted(update_root.glob("*/model.pt"))
    direct_model = update_root / "model.pt"
    if direct_model.exists():
        candidates.append(direct_model)
    if not candidates:
        return None
    return candidates[-1]


def build_resumed_phase_record(
    *,
    phase_label: str,
    phase_root: Path,
    episodes_per_task_total: int,
) -> Dict[str, object]:
    return {
        "phase_label": str(phase_label),
        "run_dirs": [str(path) for path in find_rollout_run_dirs(phase_root)],
        "instance_details": [],
        "episodes_per_task_total": int(episodes_per_task_total),
        "wall_time_sec": 0.0,
        "resumed": True,
    }


def estimate_recorded_pilot_wall_time(state: Dict[str, Any]) -> float:
    timing = state.get("timing") or {}
    total = 0.0
    baseline_phase = timing.get("baseline_phase") or {}
    final_eval_phase = timing.get("final_eval_phase") or {}
    total += float(baseline_phase.get("wall_time_sec", 0.0) or 0.0)
    total += float(final_eval_phase.get("wall_time_sec", 0.0) or 0.0)
    for item in state.get("iteration_summaries", []) or []:
        total += float(item.get("iteration_wall_time_sec", 0.0) or 0.0)
    return total


def build_resumed_world_record(
    *,
    args,
    mode: str,
    phase_label: str,
    world_instances: int = 1,
    fixed_bank_dir: str = "",
) -> Dict[str, object]:
    if mode == "static":
        return {
            "mode": "static",
            "task_group": str(args.task_group),
            "task_group_path": str(args.task_group_path),
            "generated": False,
            "num_instances": 1,
            "phase_label": phase_label,
            "resumed": True,
        }
    if mode == "fixed_bank" and str(fixed_bank_dir).strip():
        world = load_fixed_bank(Path(fixed_bank_dir), phase_label=phase_label)
        world["resumed"] = True
        return world
    return {
        "mode": str(mode),
        "generated": True,
        "num_instances": max(1, int(world_instances)),
        "phase_label": str(phase_label),
        "resumed": True,
    }


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [normalize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_jsonable(item) for item in value]
    return value


def summarize_plan_rows(plan_rows: List[Dict]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for row in plan_rows:
        task_name = str(row.get("task_config_name") or row.get("task_key") or "").strip()
        notes = str((row.get("world_generation_suggestions") or {}).get("notes", "")).strip()
        summaries.append(
            {
                "task_config_name": task_name,
                "factor_levels": normalize_factor_levels(task_name, row.get("factor_levels") or {}),
                "requested_split_label": str(row.get("requested_split_label", "")),
                "computed_split_label": str(row.get("computed_split_label", "")),
                "primary_factors_majority": list(row.get("primary_factors_majority", []) or []),
                "severity_majority": int(row.get("severity_majority", 0) or 0),
                "layout_changes": list((row.get("world_generation_suggestions") or {}).get("layout_changes", []) or []),
                "notes": notes[:240],
            }
        )
    return summaries


def load_plan_rows(plan_json_path: Optional[str]) -> Dict[str, Dict]:
    if not plan_json_path:
        return {}
    rows = json.loads(Path(plan_json_path).read_text(encoding="utf-8"))
    mapping: Dict[str, Dict] = {}
    for row in rows:
        task_config_name = str(row.get("task_config_name", "")).strip()
        if task_config_name:
            mapping[task_config_name] = row
        task_key = str(row.get("task_key", "")).strip()
        if task_key:
            mapping.setdefault(task_key, row)
    return mapping


def to_plan_row_mapping(rows: List[Dict]) -> Dict[str, Dict]:
    mapping: Dict[str, Dict] = {}
    for row in rows:
        task_config_name = str(row.get("task_config_name", "")).strip()
        if task_config_name:
            mapping[task_config_name] = row
        task_key = str(row.get("task_key", "")).strip()
        if task_key:
            mapping.setdefault(task_key, row)
    return mapping


def resolve_factor_split_for_mode(mode: str, explicit_split_label: str) -> str:
    split = str(explicit_split_label or "").strip().lower()
    if split:
        return split
    if mode in {"random", "curriculum", "targeted"}:
        return "train_id"
    return ""


def curriculum_stage(iteration_idx: int, total_iters: int) -> int:
    if total_iters <= 1:
        return 1
    progress = float(max(0, iteration_idx - 1)) / float(max(1, total_iters - 1))
    return min(4, 1 + int(progress * 4.0))


def build_random_suggestions(task_key: str, rng: random.Random, max_increased_factors: int) -> Dict:
    factor_names = list(world_factor_schema_for_task(task_key)["allowed_factors"].keys())
    max_count = max(1, min(int(max_increased_factors), len(factor_names)))
    count = rng.randint(1, max_count)
    selected = set(rng.sample(factor_names, count))
    suggestions = {
        name: ("increase" if name in selected else "keep")
        for name in factor_names
    }
    suggestions["layout_changes"] = ["keep_layout"]
    suggestions["notes"] = f"Random shift sampler selected {sorted(selected)}."
    return normalize_worldgen_suggestions(task_key, suggestions)


def build_random_factor_levels(task_key: str, rng: random.Random, max_increased_factors: int) -> Dict[str, int]:
    factor_codes = list(world_factor_schema_for_task(task_key)["primary_factor_codes"])
    levels = empty_factor_levels(task_key)
    max_count = max(1, min(int(max_increased_factors), len(factor_codes)))
    count = rng.randint(1, max_count)
    selected = rng.sample(factor_codes, count)
    for code in selected:
        levels[code] = rng.choice([1, 2])
    return levels


def build_curriculum_suggestions(task_key: str, stage_idx: int) -> Dict:
    factor_names = list(world_factor_schema_for_task(task_key)["allowed_factors"].keys())
    order = [name for name in ("view_difficulty", "visibility", "distractors", "path_difficulty") if name in factor_names]
    selected = set(order[: max(1, min(int(stage_idx), len(order)))])
    suggestions = {
        name: ("increase" if name in selected else "keep")
        for name in factor_names
    }
    suggestions["layout_changes"] = ["keep_layout"]
    suggestions["notes"] = f"Scalar curriculum stage {int(stage_idx)} with increased factors {sorted(selected)}."
    return normalize_worldgen_suggestions(task_key, suggestions)


def build_curriculum_factor_levels(task_key: str, stage_idx: int) -> Dict[str, int]:
    order = [code for code in ("H", "R", "O", "C", "P", "A") if code in world_factor_schema_for_task(task_key)["primary_factor_codes"]]
    levels = empty_factor_levels(task_key)
    if not order:
        return levels
    stage_idx = max(1, int(stage_idx))
    if stage_idx == 1:
        levels[order[0]] = 1
    elif stage_idx == 2:
        for code in order[:2]:
            levels[code] = 1
    elif stage_idx == 3:
        for code in order[:2]:
            levels[code] = 2
        for code in order[2:3]:
            levels[code] = 1
    else:
        for code in order[: min(4, len(order))]:
            levels[code] = 2
    return levels


def horizontal_distance(player_pos: Dict, target_world_center: List[float]) -> float:
    return (
        (float(player_pos.get("x", 0.0)) - float(target_world_center[0])) ** 2
        + (float(player_pos.get("z", 0.0)) - float(target_world_center[2])) ** 2
    ) ** 0.5


def travel_span(trajectory_rows: List[Dict]) -> float:
    if not trajectory_rows:
        return 0.0
    start_pos = trajectory_rows[0].get("player_pos") or {}
    start_x = float(start_pos.get("x", 0.0))
    start_z = float(start_pos.get("z", 0.0))
    best = 0.0
    for row in trajectory_rows:
        player_pos = row.get("player_pos") or {}
        dx = float(player_pos.get("x", 0.0)) - start_x
        dz = float(player_pos.get("z", 0.0)) - start_z
        best = max(best, (dx * dx + dz * dz) ** 0.5)
    return best


def infer_targeted_plan_row_from_episode(task_key: str, task_config_name: str, result: Dict, trajectory_rows: List[Dict]) -> Dict:
    success = bool(result.get("auto_success"))
    if success:
        factor_levels = derive_factor_levels_from_review(task_key, "success", [], 0)
        suggestions = normalize_worldgen_suggestions(
            task_key,
            {
                "visibility": "keep",
                "distractors": "keep",
                "path_difficulty": "keep",
                "view_difficulty": "keep",
                "layout_changes": ["keep_layout"],
                "notes": "Successful episode. Keep current world factors.",
            },
        )
        return {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "primary_failure_mode_majority": "success",
            "primary_factors_majority": [],
            "severity_majority": 0,
            "factor_levels": factor_levels,
            "trainable_with_rl_majority": True,
            "world_generation_suggestions": suggestions,
        }

    goal_metadata = result.get("goal_metadata") or {}
    target_world_center = goal_metadata.get("target_world_center")
    if not (isinstance(target_world_center, list) and len(target_world_center) == 3 and trajectory_rows):
        suggestions = normalize_worldgen_suggestions(
            task_key,
            {
                "visibility": "keep",
                "distractors": "keep",
                "path_difficulty": "keep",
                "view_difficulty": "increase",
                "layout_changes": ["keep_layout"],
                "notes": "Missing target or trajectory data. Default to view-focused targeted shift.",
            },
        )
        return {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "primary_failure_mode_majority": "ambiguous",
            "primary_factors_majority": [],
            "severity_majority": 1,
            "factor_levels": derive_factor_levels_from_review(task_key, "ambiguous", [], 1),
            "trainable_with_rl_majority": True,
            "world_generation_suggestions": suggestions,
        }

    distances = [
        horizontal_distance(row.get("player_pos") or {}, target_world_center)
        for row in trajectory_rows
        if isinstance(row.get("player_pos"), dict)
    ]
    if not distances:
        distances = [999.0]
    start_distance = float(distances[0])
    min_distance = float(min(distances))
    end_distance = float(distances[-1])
    improvement = float(start_distance - min_distance)
    span = float(travel_span(trajectory_rows))
    auto_progress = result.get("auto_progress") or {}
    mined_count = float(auto_progress.get("mine_block", 0) or 0)
    inventory_delta = float(auto_progress.get("inventory_delta", 0) or 0)
    progress_score = mined_count + inventory_delta

    if span < 1.0 and improvement < 0.5:
        failure_mode = "target_acquisition_failure"
        primary_factors = ["R", "H"]
        severity = 1
        suggestions = {
            "visibility": "keep",
            "distractors": "keep",
            "path_difficulty": "keep",
            "view_difficulty": "increase",
            "layout_changes": ["keep_layout"],
            "notes": f"Heuristic targeted sampler: little movement (span={span:.2f}) and little approach (improvement={improvement:.2f}).",
        }
    elif min_distance > 4.0 and improvement >= 0.5:
        failure_mode = "path_failure"
        primary_factors = ["P", "R"]
        severity = 2 if min_distance > 6.0 else 1
        suggestions = {
            "visibility": "keep",
            "distractors": "keep",
            "path_difficulty": "increase",
            "view_difficulty": "keep",
            "layout_changes": ["keep_layout"],
            "notes": f"Heuristic targeted sampler: some approach (improvement={improvement:.2f}) but never reached target neighborhood (min_distance={min_distance:.2f}).",
        }
    elif min_distance <= 3.0 and progress_score <= 0.0:
        failure_mode = "final_actuation_failure"
        primary_factors = ["A"]
        severity = 1
        suggestions = {
            "visibility": "decrease",
            "distractors": "keep",
            "path_difficulty": "decrease",
            "view_difficulty": "decrease",
            "layout_changes": ["make_target_visible_at_step0"],
            "notes": f"Heuristic targeted sampler: reached target neighborhood (min_distance={min_distance:.2f}) but no interaction progress.",
        }
    elif improvement < 0.5 and span >= 2.0:
        failure_mode = "distractor_confusion"
        primary_factors = ["C"]
        severity = 1
        suggestions = {
            "visibility": "keep",
            "distractors": "increase",
            "path_difficulty": "keep",
            "view_difficulty": "keep",
            "layout_changes": ["add_similar_distractor"],
            "notes": f"Heuristic targeted sampler: agent moved around (span={span:.2f}) without meaningful target approach (improvement={improvement:.2f}).",
        }
    elif end_distance > start_distance + 1.0:
        failure_mode = "visibility_recovery_failure"
        primary_factors = ["O", "H"]
        severity = 2
        suggestions = {
            "visibility": "increase",
            "distractors": "keep",
            "path_difficulty": "keep",
            "view_difficulty": "increase",
            "layout_changes": ["keep_layout"],
            "notes": f"Heuristic targeted sampler: agent drifted away from target (start={start_distance:.2f}, end={end_distance:.2f}).",
        }
    else:
        failure_mode = "timeout_unstable_recovery"
        primary_factors = ["H", "O"]
        severity = 1
        suggestions = {
            "visibility": "keep",
            "distractors": "increase",
            "path_difficulty": "keep",
            "view_difficulty": "increase",
            "layout_changes": ["keep_layout"],
            "notes": f"Heuristic targeted sampler fallback. start={start_distance:.2f}, min={min_distance:.2f}, end={end_distance:.2f}, span={span:.2f}.",
        }

    return {
        "task_config_name": task_config_name,
        "task_key": task_key,
        "primary_failure_mode_majority": failure_mode,
        "primary_factors_majority": primary_factors,
        "severity_majority": severity,
        "factor_levels": derive_factor_levels_from_review(task_key, failure_mode, primary_factors, severity),
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": normalize_worldgen_suggestions(task_key, suggestions),
    }


def aggregate_targeted_plan_rows(reference_run_dir: Path) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for episode_dir in sorted(reference_run_dir.glob("*/seed_*_ep_*")):
        result_path = episode_dir / "result.json"
        trajectory_path = episode_dir / "trajectory.jsonl"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trajectory_rows = load_jsonl(trajectory_path) if trajectory_path.exists() else []
        task_config_name = str(result.get("task_config_name", "")).strip()
        task_key = str((result.get("goal_metadata") or {}).get("task_key") or task_config_name).strip()
        if not task_config_name:
            continue
        grouped[task_config_name].append(
            infer_targeted_plan_row_from_episode(
                task_key=task_key,
                task_config_name=task_config_name,
                result=result,
                trajectory_rows=trajectory_rows,
            )
        )

    plan_rows: Dict[str, Dict] = {}
    for task_config_name, rows in grouped.items():
        primary_modes = Counter(str(row.get("primary_failure_mode_majority", "ambiguous")) for row in rows)
        primary_factor_votes = Counter()
        severity_votes = Counter()
        factor_votes = {
            "visibility": Counter(),
            "distractors": Counter(),
            "path_difficulty": Counter(),
            "view_difficulty": Counter(),
        }
        layout_votes = Counter()
        note_lines: List[str] = []
        task_key = str(rows[0].get("task_key") or task_config_name)
        for row in rows:
            suggestions = row.get("world_generation_suggestions") or {}
            for code in row.get("primary_factors_majority", []) or []:
                primary_factor_votes[str(code)] += 1
            severity_votes[int(row.get("severity_majority", 1) or 1)] += 1
            for factor_name in factor_votes:
                factor_votes[factor_name][str(suggestions.get(factor_name, "keep"))] += 1
            for tag in suggestions.get("layout_changes", []) or []:
                layout_votes[str(tag)] += 1
            note = str(suggestions.get("notes", "")).strip()
            if note:
                note_lines.append(note)
        aggregated = {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "primary_failure_mode_majority": primary_modes.most_common(1)[0][0] if primary_modes else "ambiguous",
            "primary_factors_majority": [code for code, _ in primary_factor_votes.most_common(3)],
            "severity_majority": severity_votes.most_common(1)[0][0] if severity_votes else 1,
            "factor_levels": derive_factor_levels_from_review(
                task_key,
                primary_modes.most_common(1)[0][0] if primary_modes else "ambiguous",
                [code for code, _ in primary_factor_votes.most_common(3)],
                severity_votes.most_common(1)[0][0] if severity_votes else 1,
            ),
            "trainable_with_rl_majority": True,
            "world_generation_suggestions": normalize_worldgen_suggestions(
                task_key,
                {
                    "visibility": factor_votes["visibility"].most_common(1)[0][0] if factor_votes["visibility"] else "keep",
                    "distractors": factor_votes["distractors"].most_common(1)[0][0] if factor_votes["distractors"] else "keep",
                    "path_difficulty": factor_votes["path_difficulty"].most_common(1)[0][0] if factor_votes["path_difficulty"] else "keep",
                    "view_difficulty": factor_votes["view_difficulty"].most_common(1)[0][0] if factor_votes["view_difficulty"] else "keep",
                    "layout_changes": [tag for tag, _ in layout_votes.most_common(3)] or ["keep_layout"],
                    "notes": f"Heuristic targeted plan aggregated from {len(rows)} episode(s). " + (" | ".join(note_lines[:3]) if note_lines else ""),
                },
            ),
        }
        plan_rows[task_config_name] = aggregated
        plan_rows.setdefault(task_key, aggregated)
    return plan_rows


def collect_episode_dirs(run_dirs: List[Path]) -> List[Path]:
    episode_dirs: List[Path] = []
    for run_dir in run_dirs:
        episode_dirs.extend(sorted(run_dir.glob("*/seed_*_ep_*")))
    return episode_dirs


def aggregate_targeted_plan_rows_from_run_dirs(run_dirs: List[Path]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for episode_dir in collect_episode_dirs(run_dirs):
        result_path = episode_dir / "result.json"
        trajectory_path = episode_dir / "trajectory.jsonl"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trajectory_rows = load_jsonl(trajectory_path) if trajectory_path.exists() else []
        task_config_name = str(result.get("task_config_name", "")).strip()
        task_key = str((result.get("goal_metadata") or {}).get("task_key") or task_config_name).strip()
        if not task_config_name:
            continue
        grouped[task_config_name].append(
            infer_targeted_plan_row_from_episode(
                task_key=task_key,
                task_config_name=task_config_name,
                result=result,
                trajectory_rows=trajectory_rows,
            )
        )

    plan_rows: Dict[str, Dict] = {}
    for task_config_name, rows in grouped.items():
        primary_modes = Counter(str(row.get("primary_failure_mode_majority", "ambiguous")) for row in rows)
        primary_factor_votes = Counter()
        severity_votes = Counter()
        factor_votes = {
            "visibility": Counter(),
            "distractors": Counter(),
            "path_difficulty": Counter(),
            "view_difficulty": Counter(),
        }
        layout_votes = Counter()
        note_lines: List[str] = []
        task_key = str(rows[0].get("task_key") or task_config_name)
        for row in rows:
            suggestions = row.get("world_generation_suggestions") or {}
            for code in row.get("primary_factors_majority", []) or []:
                primary_factor_votes[str(code)] += 1
            severity_votes[int(row.get("severity_majority", 1) or 1)] += 1
            for factor_name in factor_votes:
                factor_votes[factor_name][str(suggestions.get(factor_name, "keep"))] += 1
            for tag in suggestions.get("layout_changes", []) or []:
                layout_votes[str(tag)] += 1
            note = str(suggestions.get("notes", "")).strip()
            if note:
                note_lines.append(note)
        aggregated = {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "primary_failure_mode_majority": primary_modes.most_common(1)[0][0] if primary_modes else "ambiguous",
            "primary_factors_majority": [code for code, _ in primary_factor_votes.most_common(3)],
            "severity_majority": severity_votes.most_common(1)[0][0] if severity_votes else 1,
            "factor_levels": derive_factor_levels_from_review(
                task_key,
                primary_modes.most_common(1)[0][0] if primary_modes else "ambiguous",
                [code for code, _ in primary_factor_votes.most_common(3)],
                severity_votes.most_common(1)[0][0] if severity_votes else 1,
            ),
            "trainable_with_rl_majority": True,
            "world_generation_suggestions": normalize_worldgen_suggestions(
                task_key,
                {
                    "visibility": factor_votes["visibility"].most_common(1)[0][0] if factor_votes["visibility"] else "keep",
                    "distractors": factor_votes["distractors"].most_common(1)[0][0] if factor_votes["distractors"] else "keep",
                    "path_difficulty": factor_votes["path_difficulty"].most_common(1)[0][0] if factor_votes["path_difficulty"] else "keep",
                    "view_difficulty": factor_votes["view_difficulty"].most_common(1)[0][0] if factor_votes["view_difficulty"] else "keep",
                    "layout_changes": [tag for tag, _ in layout_votes.most_common(3)] or ["keep_layout"],
                    "notes": f"Heuristic targeted plan aggregated from {len(rows)} episode(s). " + (" | ".join(note_lines[:3]) if note_lines else ""),
                },
            ),
        }
        plan_rows[task_config_name] = aggregated
        plan_rows.setdefault(task_key, aggregated)
    return plan_rows


def bootstrap_targeted_row(task_key: str, task_config_name: str, iteration_idx: int) -> Dict:
    stage_idx = max(1, min(int(iteration_idx), 2))
    suggestions = build_curriculum_suggestions(task_key=task_key, stage_idx=stage_idx)
    notes = str(suggestions.get("notes", "")).strip()
    suggestions["notes"] = (
        f"Targeted bootstrap from successful prior episodes. {notes}".strip()
    )
    return {
        "task_config_name": task_config_name,
        "task_key": task_key,
        "primary_failure_mode_majority": "success_bootstrap",
        "primary_factors_majority": ["H"],
        "severity_majority": min(2, max(1, int(stage_idx))),
        "factor_levels": build_curriculum_factor_levels(task_key=task_key, stage_idx=stage_idx),
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": normalize_worldgen_suggestions(task_key, suggestions),
    }


def build_plan_rows_for_mode(
    args,
    task_specs,
    mode: str,
    iteration_idx: int,
    phase_label: str,
    plan_rows_by_task: Dict[str, Dict],
    targeted_rows_by_task: Optional[Dict[str, Dict]] = None,
    factor_split_label: str = "",
) -> List[Dict]:
    if mode == "static":
        return []

    rng_seed = int(args.base_seed) * 10_000 + int(iteration_idx) * 101 + sum(ord(ch) for ch in phase_label)
    rng = random.Random(rng_seed)
    requested_split_label = resolve_factor_split_for_mode(mode, factor_split_label)
    rows: List[Dict] = []
    for task_spec in task_specs:
        task_key = str(task_spec.task_key or task_spec.task_config_name)
        task_config_name = str(task_spec.task_config_name)
        if mode in {"plan", "plan_exact"}:
            plan_row = plan_rows_by_task.get(task_config_name) or plan_rows_by_task.get(task_key)
            if plan_row is None:
                raise KeyError(
                    f"worldgen plan row not found for task '{task_config_name}'. "
                    f"Provide --worldgen-plan-json with a matching task_config_name."
                )
            suggestions = normalize_worldgen_suggestions(task_key, plan_row.get("world_generation_suggestions") or {})
            base_factor_levels = normalize_factor_levels(task_key, plan_row.get("factor_levels") or {})
            if mode == "plan" and requested_split_label:
                factor_levels = project_factor_levels_to_split(
                    task_key,
                    base_factor_levels,
                    requested_split_label,
                    preferred_codes=plan_row.get("primary_factors_majority") or [],
                )
                suggestions = factor_levels_to_worldgen_suggestions(task_key, factor_levels)
                original_notes = str((plan_row.get("world_generation_suggestions") or {}).get("notes", "")).strip()
                if original_notes:
                    suggestions["notes"] = f"{suggestions.get('notes', '').strip()} {original_notes}".strip()
            else:
                factor_levels = base_factor_levels
            row = {
                "task_config_name": task_config_name,
                "task_key": task_key,
                "primary_failure_mode_majority": plan_row.get("primary_failure_mode_majority", "unknown"),
                "primary_factors_majority": list(plan_row.get("primary_factors_majority", []) or []),
                "severity_majority": int(plan_row.get("severity_majority", 1) or 1),
                "factor_levels": factor_levels,
                "trainable_with_rl_majority": bool(plan_row.get("trainable_with_rl_majority", True)),
                "world_generation_suggestions": suggestions,
            }
            if (
                mode == "plan_exact"
                and bool(getattr(args, "collect_resample_plan_layout_seed", False))
                and "collect" in str(phase_label or "").lower()
            ):
                row["layout_seed"] = int(rng.randint(0, 2**31 - 1))
        elif mode == "targeted":
            targeted_row = (targeted_rows_by_task or {}).get(task_config_name) or (targeted_rows_by_task or {}).get(task_key)
            if targeted_row is None:
                raise KeyError(
                    f"targeted world rows not found for task '{task_config_name}'. "
                    f"Provide prior rollout results or fall back to --collect-world-mode plan."
                )
            primary_mode = str(targeted_row.get("primary_failure_mode_majority", "ambiguous"))
            if primary_mode == "success" and bool(args.targeted_bootstrap_on_success):
                row = bootstrap_targeted_row(
                    task_key=task_key,
                    task_config_name=task_config_name,
                    iteration_idx=iteration_idx,
                )
            else:
                row = {
                    "task_config_name": task_config_name,
                    "task_key": task_key,
                    "primary_failure_mode_majority": primary_mode,
                    "primary_factors_majority": list(targeted_row.get("primary_factors_majority", []) or []),
                    "severity_majority": int(targeted_row.get("severity_majority", 1) or 1),
                    "factor_levels": project_factor_levels_to_split(
                        task_key,
                        normalize_factor_levels(task_key, targeted_row.get("factor_levels") or {}),
                        requested_split_label or "train_id",
                        preferred_codes=targeted_row.get("primary_factors_majority") or [],
                    ),
                    "trainable_with_rl_majority": bool(targeted_row.get("trainable_with_rl_majority", True)),
                    "world_generation_suggestions": factor_levels_to_worldgen_suggestions(
                        task_key,
                        project_factor_levels_to_split(
                            task_key,
                            normalize_factor_levels(task_key, targeted_row.get("factor_levels") or {}),
                            requested_split_label or "train_id",
                            preferred_codes=targeted_row.get("primary_factors_majority") or [],
                        ),
                    ),
                }
                original_notes = str((targeted_row.get("world_generation_suggestions") or {}).get("notes", "")).strip()
                if original_notes:
                    row["world_generation_suggestions"]["notes"] = (
                        f"{row['world_generation_suggestions'].get('notes', '').strip()} {original_notes}".strip()
                    )
        elif mode == "random":
            factor_levels = sample_factor_levels_for_split(
                task_key,
                requested_split_label or "train_id",
                rng,
            )
            row = {
                "task_config_name": task_config_name,
                "task_key": task_key,
                "primary_failure_mode_majority": "random_shift_sampler",
                "primary_factors_majority": [code for code, value in factor_levels.items() if int(value) > 0],
                "severity_majority": max([int(value) for value in factor_levels.values()] or [0]),
                "factor_levels": factor_levels,
                "trainable_with_rl_majority": True,
                "world_generation_suggestions": factor_levels_to_worldgen_suggestions(task_key, factor_levels),
            }
            row["world_generation_suggestions"]["notes"] = (
                f"Random split-aware sampler requested {requested_split_label or 'train_id'} with factor levels {factor_levels}."
            )
        elif mode == "curriculum":
            stage_idx = curriculum_stage(iteration_idx=iteration_idx, total_iters=int(args.train_iters))
            base_factor_levels = build_curriculum_factor_levels(task_key=task_key, stage_idx=stage_idx)
            factor_levels = project_factor_levels_to_split(
                task_key,
                base_factor_levels,
                requested_split_label or "train_id",
                preferred_codes=[code for code, value in base_factor_levels.items() if int(value) > 0],
            )
            row = {
                "task_config_name": task_config_name,
                "task_key": task_key,
                "primary_failure_mode_majority": f"curriculum_stage_{int(stage_idx)}",
                "primary_factors_majority": [code for code, value in factor_levels.items() if int(value) > 0],
                "severity_majority": max([int(value) for value in factor_levels.values()] or [0]),
                "factor_levels": factor_levels,
                "trainable_with_rl_majority": True,
                "world_generation_suggestions": factor_levels_to_worldgen_suggestions(task_key, factor_levels),
            }
            row["world_generation_suggestions"]["notes"] = (
                f"Scalar curriculum stage {int(stage_idx)} projected to split {requested_split_label or 'train_id'} with factor levels {factor_levels}."
            )
        else:
            raise ValueError(f"Unsupported world mode: {mode}")

        row["world_generation_suggestions"] = normalize_worldgen_suggestions(
            task_key,
            row.get("world_generation_suggestions") or {},
        )
        row["layout_seed"] = int(rng.randint(0, 2**31 - 1))
        row["world_instance_index"] = 0
        row["template_index"] = int(row.get("template_index", 0) or 0)
        row["requested_split_label"] = requested_split_label or str(row.get("requested_split_label") or classify_factor_split(task_key, row.get("factor_levels") or {}))
        row["computed_split_label"] = classify_factor_split(task_key, row.get("factor_levels") or {})
        row["hard_factor_count"] = int(hard_factor_count(task_key, row.get("factor_levels") or {}))
        rows.append(row)
    return rows


def materialize_plan_rows_for_instance(plan_rows: List[Dict], instance_idx: int) -> List[Dict]:
    rows = copy.deepcopy(plan_rows)
    for row in rows:
        base_template_index = int(row.get("template_index", 0) or 0)
        stable_payload = {
            "task_config_name": str(row.get("task_config_name", "")),
            "factor_levels": normalize_factor_levels(
                str(row.get("task_key") or row.get("task_config_name") or ""),
                row.get("factor_levels") or {},
            ),
            "requested_split_label": str(row.get("requested_split_label", "")),
            "base_layout_seed": int(row.get("layout_seed", 0) or 0),
            "instance_idx": int(instance_idx),
        }
        row["world_instance_index"] = int(instance_idx)
        row["template_index"] = int(base_template_index + instance_idx)
        row["layout_seed"] = abs(hash(json.dumps(stable_payload, sort_keys=True, ensure_ascii=False))) % (2**31 - 1)
    return rows


def build_worldgen_cmd(
    plan_json_path: Path,
    source_env_conf_dir: Path,
    out_dir: Path,
    mine_layout_backend: str,
    mine_anchor_mode: str,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen",
        "--plan-json",
        str(plan_json_path),
        "--env-conf-dir",
        str(source_env_conf_dir),
        "--out-dir",
        str(out_dir),
        "--mine-layout-backend",
        str(mine_layout_backend),
        "--mine-anchor-mode",
        str(mine_anchor_mode),
    ]


def build_bake_goal_cmd(
    *,
    task_group: str,
    task_group_path: Path,
    tasks: str,
    protocol_name: str,
    base_seed: int,
    episode_retries: int,
    save_debug_assets: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets",
        "--task-group",
        str(task_group),
        "--task-group-path",
        str(task_group_path),
        "--protocol",
        str(protocol_name),
        "--tasks",
        str(tasks),
        "--base-seed",
        str(base_seed),
        "--episode-retries",
        str(episode_retries),
    ]
    if save_debug_assets:
        cmd.append("--save-debug-assets")
    return cmd


def load_fixed_bank(bank_dir: Path, phase_label: str) -> Dict[str, object]:
    manifest_path = bank_dir / "bank_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fixed bank manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_instances = list(manifest.get("instance_worlds") or [])
    if not raw_instances:
        raise RuntimeError(f"No fixed-bank instances found in: {manifest_path}")
    instance_worlds: List[Dict[str, object]] = []
    base_name = sanitize_token(f"{bank_dir.name}_{phase_label}_fixed")
    for item in raw_instances:
        instance_idx = int(item.get("instance_idx", len(instance_worlds)))
        task_group_path = str(item.get("generated_task_group_dir") or item.get("task_group_path") or "").strip()
        if not task_group_path:
            raise RuntimeError(f"Missing generated_task_group_dir in fixed bank manifest row: {item}")
        instance_worlds.append(
            {
                "instance_idx": instance_idx,
                "mode": "fixed_bank",
                "task_group": sanitize_token(f"{base_name}_instance_{instance_idx:03d}"),
                "task_group_path": task_group_path,
                "generated": False,
                "phase_label": phase_label,
                "fixed_bank_dir": str(bank_dir),
                "manifest_path": str(item.get("manifest_path") or ""),
                "plan_json": str(item.get("plan_json") or ""),
                "plan_rows": list(item.get("plan_rows") or []),
                "split_label": str(item.get("split_label") or manifest.get("split_label") or ""),
            }
        )
    return {
        "mode": "fixed_bank",
        "task_group": sanitize_token(f"{base_name}_root"),
        "task_group_path": str(instance_worlds[0]["task_group_path"]),
        "generated": False,
        "phase_label": phase_label,
        "fixed_bank_dir": str(bank_dir),
        "manifest_path": str(manifest_path),
        "num_instances": len(instance_worlds),
        "instance_worlds": instance_worlds,
        "split_label": str(manifest.get("split_label") or ""),
    }


def build_vlm_review_cmd(args, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_vlm_review",
        "--run-dir",
        str(run_dir),
        "--tasks",
        args.tasks,
        "--model-id",
        str(args.targeted_review_model_id),
        "--backend-url",
        str(args.targeted_review_backend_url),
        "--api-key",
        str(args.targeted_review_api_key),
        "--max-mid-images",
        str(args.targeted_review_max_mid_images),
    ]
    if args.targeted_review_overwrite:
        cmd.append("--overwrite")
    return cmd


def aggregate_targeted_rows_with_backend(reference_run_dir: Path, args, env) -> Dict[str, Dict]:
    run_dirs = find_rollout_run_dirs(reference_run_dir)
    if not run_dirs:
        raise RuntimeError(f"No rollout run directories found under {reference_run_dir}")
    backend = str(args.targeted_review_backend)
    if backend == "heuristic":
        return aggregate_targeted_plan_rows_from_run_dirs(run_dirs)
    if backend != "openai":
        raise ValueError(f"Unsupported targeted review backend: {backend}")
    if not str(args.targeted_review_api_key).strip():
        raise ValueError(
            "OpenAI targeted review requires OPENAI_API_KEY or --targeted-review-api-key."
        )
    review_rows: List[Dict] = []
    for run_dir in run_dirs:
        run_command(build_vlm_review_cmd(args, run_dir), env)
        review_rows_path = run_dir / "vlm_review" / "vlm_reviews.jsonl"
        if not review_rows_path.exists():
            raise FileNotFoundError(f"VLM review rows not found: {review_rows_path}")
        review_rows.extend(load_jsonl(review_rows_path))
    if not review_rows:
        raise RuntimeError(f"No VLM review rows collected under {reference_run_dir}")
    return to_plan_row_mapping(aggregate_worldgen_plan(review_rows))


def resolve_phase_world(
    args,
    task_specs,
    mode: str,
    iteration_idx: int,
    phase_label: str,
    factor_split_label: str,
    plan_rows_by_task: Dict[str, Dict],
    targeted_rows_by_task: Optional[Dict[str, Dict]],
    env,
    worldgen_root: Path,
    world_instances: int = 1,
    fixed_bank_dir: str = "",
) -> Dict[str, object]:
    if mode == "fixed_bank":
        if not str(fixed_bank_dir).strip():
            raise ValueError(f"Mode '{mode}' requires a fixed bank directory for phase '{phase_label}'.")
        return load_fixed_bank(Path(fixed_bank_dir), phase_label=phase_label)
    if mode == "static":
        return {
            "mode": "static",
            "task_group": str(args.task_group),
            "task_group_path": str(args.task_group_path),
            "generated": False,
            "num_instances": 1,
        }

    phase_dir = worldgen_root / sanitize_token(phase_label)
    phase_dir.mkdir(parents=True, exist_ok=True)
    task_group_name = sanitize_token(f"{args.task_group}_{phase_label}_{mode}")
    source_env_conf_dir = Path(args.worldgen_source_dir or args.task_group_path)
    instance_worlds: List[Dict[str, object]] = []
    instance_count = max(1, int(world_instances))
    base_plan_rows = build_plan_rows_for_mode(
        args=args,
        task_specs=task_specs,
        mode=mode,
        iteration_idx=iteration_idx,
        phase_label=phase_label,
        plan_rows_by_task=plan_rows_by_task,
        targeted_rows_by_task=targeted_rows_by_task,
        factor_split_label=factor_split_label,
    )
    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "phase_world_plan",
                "phase_label": phase_label,
                "iteration_idx": int(iteration_idx),
                "mode": str(mode),
                "factor_split_label": str(factor_split_label or ""),
                "requested_split_label": resolve_factor_split_for_mode(mode, factor_split_label),
                "world_instances": int(instance_count),
                "plan_rows": summarize_plan_rows(base_plan_rows),
            },
            ensure_ascii=False,
        ),
    )
    for instance_idx in range(instance_count):
        instance_dir = phase_dir / f"instance_{instance_idx:03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        plan_rows = materialize_plan_rows_for_instance(base_plan_rows, instance_idx=instance_idx)
        plan_json_path = instance_dir / "worldgen_plan.json"
        plan_json_path.write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        generated_groups_root = instance_dir / "generated_task_groups"
        generated_groups_root.mkdir(parents=True, exist_ok=True)
        run_command(
            build_worldgen_cmd(
                plan_json_path=plan_json_path,
                source_env_conf_dir=source_env_conf_dir,
                out_dir=generated_groups_root,
                mine_layout_backend=args.mine_layout_backend,
                mine_anchor_mode=args.mine_anchor_mode,
            ),
            env,
        )
        generated_dir = latest_subdir(generated_groups_root)
        if args.auto_goal and args.bake_generated_goals:
            run_command(
                build_bake_goal_cmd(
                    task_group=sanitize_token(f"{task_group_name}_instance_{instance_idx:03d}"),
                    task_group_path=generated_dir,
                    tasks=args.tasks,
                    protocol_name=args.protocol,
                    base_seed=int(args.bake_goal_base_seed),
                    episode_retries=int(args.bake_goal_episode_retries),
                    save_debug_assets=bool(args.bake_goal_save_debug_assets),
                ),
                env,
            )
        manifest_path = generated_dir / "worldgen_manifest.json"
        instance_worlds.append(
            {
                "instance_idx": instance_idx,
                "mode": mode,
                "task_group": sanitize_token(f"{task_group_name}_instance_{instance_idx:03d}"),
                "task_group_path": str(generated_dir),
                "generated": True,
                "phase_label": phase_label,
                "plan_json": str(plan_json_path),
                "generated_task_group_dir": str(generated_dir),
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                "plan_rows": plan_rows,
            }
        )
    return {
        "mode": mode,
        "task_group": task_group_name,
        "task_group_path": str(instance_worlds[0]["task_group_path"]),
        "generated": True,
        "phase_label": phase_label,
        "plan_json": str(instance_worlds[0]["plan_json"]),
        "generated_task_group_dir": str(instance_worlds[0]["generated_task_group_dir"]),
        "manifest_path": str(instance_worlds[0]["manifest_path"]),
        "plan_rows": instance_worlds[0]["plan_rows"],
        "num_instances": len(instance_worlds),
        "instance_worlds": instance_worlds,
    }


def build_eval_cmd(
    args,
    out_dir: Path,
    model_path: str,
    episodes_per_task: int,
    save_fragments: bool,
    task_group: Optional[str] = None,
    task_group_path: Optional[str] = None,
    base_seed_override: Optional[int] = None,
    sampling_base_seed_override: Optional[int] = None,
    skip_video_override: Optional[bool] = None,
) -> List[str]:
    cfg_base_ref_model_path = resolve_cfg_base_ref_model_path(args)
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout",
        "--env-source",
        args.env_source,
        "--protocol",
        args.protocol,
        "--task-group",
        str(task_group or args.task_group),
        "--task-group-path",
        str(task_group_path or args.task_group_path),
        "--tasks",
        args.tasks,
        "--episodes-per-task",
        str(episodes_per_task),
        "--base-seed",
        str(args.base_seed if base_seed_override is None else base_seed_override),
        "--seed-step",
        str(args.seed_step),
        "--episode-retries",
        str(args.episode_retries),
        "--step-budget-override",
        str(args.step_budget_override),
        "--model-path",
        model_path,
        "--out-dir",
        str(out_dir),
    ]
    sampling_seed_step = resolve_sampling_seed_step(args)
    effective_sampling_base_seed = args.sampling_base_seed if sampling_base_seed_override is None else sampling_base_seed_override
    if effective_sampling_base_seed is not None:
        cmd.extend(["--sampling-base-seed", str(effective_sampling_base_seed)])
        if sampling_seed_step is not None:
            cmd.extend(["--sampling-seed-step", str(sampling_seed_step)])
    if args.auto_goal:
        cmd.append("--auto-goal")
    else:
        cmd.extend(["--goal-spec", str(args.goal_spec)])
    if args.cfg_coef is not None:
        cmd.extend(["--cfg-coef", str(args.cfg_coef)])
    cmd.extend(["--env-reward-scale", str(args.env_reward_scale)])
    cmd.extend(["--cfg-policy-mode", str(args.cfg_policy_mode)])
    if cfg_base_ref_model_path:
        cmd.extend(["--cfg-base-ref-model-path", cfg_base_ref_model_path])
    # PPO collection fragments should end at success so the update is not
    # dominated by post-success tail behavior. Keep the CLI flag for eval paths,
    # but force success-stop when we are saving fragments for policy updates.
    if args.stop_on_success or save_fragments:
        cmd.append("--stop-on-success")
    if bool(args.skip_video) if skip_video_override is None else bool(skip_video_override):
        cmd.append("--skip-video")
    if args.save_debug_assets:
        cmd.append("--save-debug-assets")
    if save_fragments:
        cmd.append("--save-ppo-fragments")
    return cmd


def distribute_episode_counts(total_episodes: int, num_instances: int) -> List[int]:
    total = max(1, int(total_episodes))
    count = max(1, int(num_instances))
    active = min(total, count)
    base = total // active
    remainder = total % active
    return [base + (1 if idx < remainder else 0) for idx in range(active)]


def execute_phase_rollouts(
    args,
    phase_root: Path,
    model_path: str,
    episodes_per_task: int,
    save_fragments: bool,
    phase_world: Dict[str, object],
    env,
    parallel_workers: int = 1,
) -> Dict[str, object]:
    instance_worlds = list(phase_world.get("instance_worlds") or [phase_world])
    phase_label = str(phase_world.get("phase_label") or phase_root.name)
    skip_video_for_phase = resolve_phase_skip_video(args, phase_label)
    phase_started_at = time.perf_counter()
    requested_parallel_workers = max(1, int(parallel_workers))
    enable_episode_parallel_workers = (
        requested_parallel_workers > 1 and len(instance_worlds) == 1 and int(episodes_per_task) > 1
    )
    worker_worlds: List[Dict[str, object]]
    worker_episode_counts: List[int]
    if enable_episode_parallel_workers:
        worker_count = min(requested_parallel_workers, int(episodes_per_task))
        worker_worlds = [instance_worlds[0] for _ in range(worker_count)]
        worker_episode_counts = distribute_episode_counts(episodes_per_task, worker_count)
    else:
        active_instance_count = min(len(instance_worlds), max(1, int(episodes_per_task)))
        worker_worlds = list(instance_worlds[:active_instance_count])
        worker_episode_counts = distribute_episode_counts(episodes_per_task, len(worker_worlds))
    enable_instance_parallel_workers = (
        not enable_episode_parallel_workers and requested_parallel_workers > 1 and len(worker_worlds) > 1
    )
    parallel_workers_active = 1
    parallel_mode = "serial"
    if enable_episode_parallel_workers:
        parallel_workers_active = int(len(worker_worlds))
        parallel_mode = "episodes"
    elif enable_instance_parallel_workers:
        parallel_workers_active = int(min(requested_parallel_workers, len(worker_worlds)))
        parallel_mode = "instances"
    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "phase_start",
                "phase_label": phase_label,
                "num_instances": int(len(instance_worlds)),
                "parallel_workers_requested": int(requested_parallel_workers),
                "parallel_workers_active": int(parallel_workers_active),
                "parallel_mode": parallel_mode,
                "episodes_per_task_total": int(sum(worker_episode_counts)),
                "save_fragments": bool(save_fragments),
            },
            ensure_ascii=False,
        ),
    )
    run_dirs: List[Path] = []
    instance_details: List[Dict[str, object]] = []
    if enable_episode_parallel_workers:
        worker_processes = []
        episode_prefix = 0
        sampling_seed_step = resolve_sampling_seed_step(args)
        for worker_idx, (worker_world, worker_episodes) in enumerate(zip(worker_worlds, worker_episode_counts)):
            worker_root = phase_root / f"worker_{worker_idx:03d}"
            worker_root.mkdir(parents=True, exist_ok=True)
            worker_started_at = time.perf_counter()
            base_seed_override = int(args.base_seed) + int(episode_prefix) * max(1, int(args.seed_step) or 1)
            sampling_base_seed_override = None
            if args.sampling_base_seed is not None:
                sampling_base_seed_override = int(args.sampling_base_seed) + int(episode_prefix) * max(1, int(sampling_seed_step) or 1)
            episode_prefix += int(worker_episodes)
            effective_task_group = sanitize_token(f"{worker_world['task_group']}_worker_{worker_idx:03d}")
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_instance_start",
                        "phase_label": phase_label,
                        "instance_progress": f"{worker_idx + 1}/{len(worker_worlds)}",
                        "instance_idx": int(worker_idx),
                        "episodes_per_task": int(worker_episodes),
                        "task_group": str(effective_task_group),
                        "parallel_worker": True,
                    },
                    ensure_ascii=False,
                ),
            )
            proc = start_command(
                build_eval_cmd(
                    args,
                    worker_root,
                    model_path,
                    worker_episodes,
                    save_fragments,
                    task_group=effective_task_group,
                    task_group_path=worker_world["task_group_path"],
                    base_seed_override=base_seed_override,
                    sampling_base_seed_override=sampling_base_seed_override,
                    skip_video_override=skip_video_for_phase,
                ),
                env,
            )
            worker_processes.append(
                {
                    "worker_idx": int(worker_idx),
                    "worker_root": worker_root,
                    "episodes_per_task": int(worker_episodes),
                    "task_group": str(effective_task_group),
                    "task_group_path": str(worker_world["task_group_path"]),
                    "started_at": worker_started_at,
                    "process": proc,
                }
            )
        worker_failures: List[tuple[int, int]] = []
        for item in worker_processes:
            proc = item["process"]
            return_code = int(proc.wait())
            worker_wall_time = time.perf_counter() - float(item["started_at"])
            if return_code != 0:
                worker_failures.append((int(item["worker_idx"]), return_code))
                continue
            run_dir = latest_subdir(Path(item["worker_root"]))
            run_dirs.append(run_dir)
            instance_details.append(
                {
                    "instance_idx": int(item["worker_idx"]),
                    "episodes_per_task": int(item["episodes_per_task"]),
                    "task_group": str(item["task_group"]),
                    "task_group_path": str(item["task_group_path"]),
                    "run_dir": str(run_dir),
                    "wall_time_sec": round(float(worker_wall_time), 4),
                    "parallel_worker": True,
                }
            )
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_instance_done",
                        "phase_label": phase_label,
                        "instance_progress": f"{int(item['worker_idx']) + 1}/{len(worker_worlds)}",
                        "instance_idx": int(item["worker_idx"]),
                        "episodes_per_task": int(item["episodes_per_task"]),
                        "run_dir": str(run_dir),
                        "wall_time_sec": round(float(worker_wall_time), 4),
                        "wall_time_human": format_seconds(worker_wall_time),
                        "parallel_worker": True,
                    },
                    ensure_ascii=False,
                ),
            )
        if worker_failures:
            raise RuntimeError(f"Parallel rollout workers failed: {worker_failures}")
    elif enable_instance_parallel_workers:
        sampling_seed_step = resolve_sampling_seed_step(args)
        instance_results: Dict[int, Dict[str, object]] = {}
        running_instances: List[Dict[str, object]] = []
        instance_failures: List[tuple[int, int]] = []
        next_instance_idx = 0
        while next_instance_idx < len(worker_worlds) or running_instances:
            while next_instance_idx < len(worker_worlds) and len(running_instances) < requested_parallel_workers:
                instance_idx = int(next_instance_idx)
                instance_world = worker_worlds[instance_idx]
                instance_episodes = int(worker_episode_counts[instance_idx])
                instance_root = phase_root / f"instance_{instance_idx:03d}"
                instance_root.mkdir(parents=True, exist_ok=True)
                instance_started_at = time.perf_counter()
                sampling_base_seed_override = None
                if args.sampling_base_seed is not None:
                    sampling_base_seed_override = int(args.sampling_base_seed) + int(instance_idx) * max(
                        1,
                        int(sampling_seed_step) or 1,
                    )
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_instance_start",
                            "phase_label": phase_label,
                            "instance_progress": f"{instance_idx + 1}/{len(worker_worlds)}",
                            "instance_idx": int(instance_idx),
                            "episodes_per_task": int(instance_episodes),
                            "task_group": str(instance_world["task_group"]),
                            "parallel_worker": True,
                            "parallel_mode": "instances",
                        },
                        ensure_ascii=False,
                    ),
                )
                proc = start_command(
                    build_eval_cmd(
                        args,
                        instance_root,
                        model_path,
                        instance_episodes,
                        save_fragments,
                        task_group=instance_world["task_group"],
                        task_group_path=instance_world["task_group_path"],
                        base_seed_override=int(args.base_seed) + int(instance_idx) * max(1, int(args.seed_step) or 1),
                        sampling_base_seed_override=sampling_base_seed_override,
                        skip_video_override=skip_video_for_phase,
                    ),
                    env,
                )
                running_instances.append(
                    {
                        "instance_idx": int(instance_idx),
                        "instance_world": instance_world,
                        "instance_root": instance_root,
                        "episodes_per_task": int(instance_episodes),
                        "started_at": instance_started_at,
                        "process": proc,
                    }
                )
                next_instance_idx += 1

            completed_idx = None
            while completed_idx is None:
                for running_idx, item in enumerate(running_instances):
                    return_code = item["process"].poll()
                    if return_code is not None:
                        item["return_code"] = int(return_code)
                        completed_idx = running_idx
                        break
                if completed_idx is None:
                    time.sleep(0.25)

            item = running_instances.pop(int(completed_idx))
            instance_idx = int(item["instance_idx"])
            return_code = int(item["return_code"])
            instance_wall_time = time.perf_counter() - float(item["started_at"])
            if return_code != 0:
                instance_failures.append((instance_idx, return_code))
                continue
            run_dir = latest_subdir(Path(item["instance_root"]))
            instance_results[instance_idx] = {
                "run_dir": run_dir,
                "detail": {
                    "instance_idx": int(instance_idx),
                    "episodes_per_task": int(item["episodes_per_task"]),
                    "task_group": str(item["instance_world"]["task_group"]),
                    "task_group_path": str(item["instance_world"]["task_group_path"]),
                    "run_dir": str(run_dir),
                    "wall_time_sec": round(float(instance_wall_time), 4),
                    "parallel_worker": True,
                    "parallel_mode": "instances",
                },
            }
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_instance_done",
                        "phase_label": phase_label,
                        "instance_progress": f"{instance_idx + 1}/{len(worker_worlds)}",
                        "instance_idx": int(instance_idx),
                        "episodes_per_task": int(item["episodes_per_task"]),
                        "run_dir": str(run_dir),
                        "wall_time_sec": round(float(instance_wall_time), 4),
                        "wall_time_human": format_seconds(instance_wall_time),
                        "parallel_worker": True,
                        "parallel_mode": "instances",
                    },
                    ensure_ascii=False,
                ),
            )

        if instance_failures:
            raise RuntimeError(f"Parallel fixed-bank rollout workers failed: {instance_failures}")
        for instance_idx in range(len(worker_worlds)):
            result = instance_results.get(int(instance_idx))
            if result is None:
                continue
            run_dirs.append(Path(result["run_dir"]))
            instance_details.append(dict(result["detail"]))
    else:
        for instance_idx, (instance_world, instance_episodes) in enumerate(zip(worker_worlds, worker_episode_counts)):
            instance_root = phase_root if len(worker_worlds) == 1 else (phase_root / f"instance_{instance_idx:03d}")
            instance_root.mkdir(parents=True, exist_ok=True)
            instance_started_at = time.perf_counter()
            sampling_base_seed_override = None
            sampling_seed_step = resolve_sampling_seed_step(args)
            if args.sampling_base_seed is not None:
                sampling_base_seed_override = int(args.sampling_base_seed) + int(instance_idx) * max(1, int(sampling_seed_step) or 1)
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_instance_start",
                        "phase_label": phase_label,
                        "instance_progress": f"{instance_idx + 1}/{len(worker_worlds)}",
                        "instance_idx": int(instance_idx),
                        "episodes_per_task": int(instance_episodes),
                        "task_group": str(instance_world["task_group"]),
                    },
                    ensure_ascii=False,
                ),
            )
            run_command(
                build_eval_cmd(
                    args,
                    instance_root,
                    model_path,
                    instance_episodes,
                    save_fragments,
                    task_group=instance_world["task_group"],
                    task_group_path=instance_world["task_group_path"],
                    base_seed_override=int(args.base_seed) + int(instance_idx) * max(1, int(args.seed_step) or 1),
                    sampling_base_seed_override=sampling_base_seed_override,
                    skip_video_override=skip_video_for_phase,
                ),
                env,
            )
            run_dir = latest_subdir(instance_root)
            instance_wall_time = time.perf_counter() - instance_started_at
            run_dirs.append(run_dir)
            instance_details.append(
                {
                    "instance_idx": int(instance_idx),
                    "episodes_per_task": int(instance_episodes),
                    "task_group": str(instance_world["task_group"]),
                    "task_group_path": str(instance_world["task_group_path"]),
                    "run_dir": str(run_dir),
                    "wall_time_sec": round(float(instance_wall_time), 4),
                }
            )
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_instance_done",
                        "phase_label": phase_label,
                        "instance_progress": f"{instance_idx + 1}/{len(worker_worlds)}",
                        "instance_idx": int(instance_idx),
                        "episodes_per_task": int(instance_episodes),
                        "run_dir": str(run_dir),
                        "wall_time_sec": round(float(instance_wall_time), 4),
                        "wall_time_human": format_seconds(instance_wall_time),
                    },
                    ensure_ascii=False,
                ),
            )
    phase_wall_time = time.perf_counter() - phase_started_at
    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "phase_done",
                "phase_label": phase_label,
                "num_instances": int(len(instance_worlds)),
                "parallel_workers_active": int(parallel_workers_active),
                "parallel_mode": parallel_mode,
                "episodes_per_task_total": int(sum(worker_episode_counts)),
                "wall_time_sec": round(float(phase_wall_time), 4),
                "wall_time_human": format_seconds(phase_wall_time),
            },
            ensure_ascii=False,
        ),
    )
    return {
        "phase_label": phase_label,
        "run_dirs": [str(path) for path in run_dirs],
        "instance_details": instance_details,
        "episodes_per_task_total": int(sum(worker_episode_counts)),
        "wall_time_sec": round(float(phase_wall_time), 4),
    }


def build_update_cmd(args, run_dir: Path, out_dir: Path, model_path: str, iteration_idx: int) -> List[str]:
    cfg_base_ref_model_path = resolve_cfg_base_ref_model_path(args)
    kl_anchor_model_path = resolve_kl_anchor_model_path(args)
    cmd = [
        sys.executable,
        "-m",
        "minestudio.tutorials.inference.evaluate_rocket.interaction_ppo_update",
        "--run-dir",
        str(run_dir),
        "--model-path",
        model_path,
        "--model-kind",
        "rocket2",
        "--cfg-coef",
        str(args.cfg_coef),
        "--cfg-policy-mode",
        str(args.cfg_policy_mode),
        "--output-dir",
        str(out_dir),
        "--epochs",
        str(args.ppo_epochs),
        "--iteration-idx",
        str(iteration_idx),
        "--learning-rate",
        str(args.ppo_learning_rate),
        "--ppo-clip",
        str(args.ppo_clip),
        "--gamma",
        str(args.ppo_gamma),
        "--vf-coef",
        str(args.vf_coef),
        "--policy-coef",
        str(args.policy_coef),
        "--entropy-coef",
        str(args.entropy_coef),
        "--kl-coef",
        str(args.kl_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--update-fragment-batch-size",
        str(args.update_fragment_batch_size),
        "--loss-focus-mode",
        str(args.loss_focus_mode),
        "--loss-focus-suffix-len",
        str(args.loss_focus_suffix_len),
        "--loss-focus-context-len",
        str(args.loss_focus_context_len),
        "--loss-focus-weight",
        str(args.loss_focus_weight),
        "--loss-focus-top-k",
        str(args.loss_focus_top_k),
        "--trainable-scope",
        str(args.trainable_scope),
        "--vf-warmup-iters",
        str(args.vf_warmup_iters),
    ]
    if cfg_base_ref_model_path:
        cmd.extend(["--cfg-base-ref-model-path", cfg_base_ref_model_path])
    if kl_anchor_model_path:
        cmd.extend(["--kl-anchor-model-path", kl_anchor_model_path])
    if args.normalize_advantage:
        cmd.append("--normalize-advantage")
    if args.clip_vloss:
        cmd.append("--clip-vloss")
    if args.zero_initial_vf:
        cmd.append("--zero-initial-vf")
    if args.calibrate_value_normalizer:
        cmd.append("--calibrate-value-normalizer")
    return cmd


def main():
    args = parse_args()
    task_names = parse_task_names(args.tasks)
    if len(task_names) != 1:
        raise ValueError("interaction_crossview_ppo_pilot currently supports exactly one task via --tasks.")
    task_specs = resolve_interaction_task_specs(task_names, env_source=args.env_source)
    plan_rows_by_task = load_plan_rows(args.worldgen_plan_json)
    final_eval_world_mode = args.final_eval_world_mode or args.eval_world_mode
    resume_pilot_dir = str(args.resume_pilot_dir or "").strip()
    resume_mode = bool(resume_pilot_dir)
    out_root = Path(resume_pilot_dir) if resume_mode else (Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S"))
    if resume_mode and not out_root.exists():
        raise FileNotFoundError(f"resume pilot dir not found: {out_root}")
    baseline_root = out_root / "baseline"
    iterations_root = out_root / "iterations"
    final_eval_root = out_root / "final_eval"
    worldgen_root = Path(args.worldgen_out_dir) if args.worldgen_out_dir else (out_root / "worldgen")
    for root in (baseline_root, iterations_root, final_eval_root, worldgen_root):
        root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pilot_started_at = time.perf_counter()
    resume_state = load_pilot_state(out_root) if resume_mode else {}
    recorded_resume_wall_time = estimate_recorded_pilot_wall_time(resume_state)

    current_model_path = str(resume_state.get("current_model_path") or args.model_path)
    cfg_base_ref_model_path = resolve_cfg_base_ref_model_path(args)
    kl_anchor_model_path = resolve_kl_anchor_model_path(args)
    latest_targeted_rows: Dict[str, Dict] = dict(resume_state.get("latest_targeted_rows") or {})
    baseline_world: Dict[str, object] = dict(resume_state.get("baseline_world") or {})
    baseline_phase: Dict[str, object] = dict((resume_state.get("timing") or {}).get("baseline_phase") or {})
    baseline_run_dirs: List[str] = list(resume_state.get("baseline_run_dirs") or [])
    baseline_run_dir = str(resume_state.get("baseline_run_dir") or "")
    baseline_summary: Dict[str, Dict] = dict(resume_state.get("baseline_summary") or {})
    iteration_summaries = list(resume_state.get("iteration_summaries") or [])
    final_eval_world: Dict[str, object] = dict(resume_state.get("final_eval_world") or {})
    final_eval_phase: Dict[str, object] = dict((resume_state.get("timing") or {}).get("final_eval_phase") or {})
    final_eval_run_dirs: List[str] = list(resume_state.get("final_eval_run_dirs") or [])
    final_eval_run_dir = str(resume_state.get("final_eval_run_dir") or "")
    final_eval_summary: Dict[str, Dict] = dict(resume_state.get("final_eval_summary") or {})
    best_eval_selection: Dict[str, object] = select_best_eval_model_selection(iteration_summaries, fallback_model_path=current_model_path)
    final_eval_model_selection: Dict[str, object] = dict(resume_state.get("final_eval_model_selection") or {})
    if not final_eval_model_selection:
        final_eval_model_selection = select_final_eval_model_selection(
            args.final_eval_model_mode,
            iteration_summaries,
            current_model_path,
        )

    def persist_state(status: str = "running") -> Dict[str, Any]:
        state = {
            "status": str(status),
            "config": redacted_args_dict(args),
            "timing": {
                "planned_rollout_episodes_total": int(planned_rollout_episodes_total(args)),
                "baseline_phase": baseline_phase,
                "final_eval_phase": final_eval_phase,
            },
            "latest_targeted_rows": latest_targeted_rows,
            "targeted_review_backend": args.targeted_review_backend,
            "targeted_review_model_id": args.targeted_review_model_id,
            "baseline_world": baseline_world,
            "baseline_run_dir": str(baseline_run_dir),
            "baseline_run_dirs": [str(path) for path in baseline_run_dirs],
            "baseline_run_root": str(baseline_root),
            "baseline_summary": baseline_summary,
            "iteration_summaries": iteration_summaries,
            "current_model_path": str(current_model_path),
            "final_model_path": str(current_model_path),
            "best_eval_selection": best_eval_selection,
            "final_eval_world": final_eval_world,
            "final_eval_run_dir": str(final_eval_run_dir),
            "final_eval_run_dirs": [str(path) for path in final_eval_run_dirs],
            "final_eval_run_root": str(final_eval_root),
            "final_eval_model_selection": final_eval_model_selection,
            "final_eval_summary": final_eval_summary,
        }
        write_pilot_state(out_root, state)
        return state

    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "pilot_start",
                "tasks": task_names,
                "train_iters": int(args.train_iters),
                "planned_rollout_episodes_total": int(planned_rollout_episodes_total(args)),
                "baseline_eval_episodes": 0 if bool(args.skip_baseline_eval) else int(args.eval_episodes_per_task),
                "collect_episodes_per_iteration": int(args.collect_episodes_per_task),
                "eval_episodes_per_iteration": 0 if bool(args.skip_iteration_eval) else int(args.eval_episodes_per_task),
                "final_eval_episodes": 0 if bool(args.skip_final_eval) else int(args.eval_episodes_per_task),
                "skip_video": bool(args.skip_video),
                "env_reward_scale": float(args.env_reward_scale),
                "min_successful_fragments": int(args.min_successful_fragments),
                "collect_quota_mode": str(args.collect_quota_mode),
                "collect_quota_successes_per_instance": int(args.collect_quota_successes_per_instance),
                "collect_quota_failures_per_instance": int(args.collect_quota_failures_per_instance),
                "cfg_policy_mode": str(args.cfg_policy_mode),
                "cfg_base_ref_model_path": str(cfg_base_ref_model_path),
                "kl_anchor_model_path": str(kl_anchor_model_path),
                "ppo_gamma": float(args.ppo_gamma),
                "max_grad_norm": float(args.max_grad_norm),
                "final_eval_model_mode": str(args.final_eval_model_mode),
                "skip_baseline_eval": bool(args.skip_baseline_eval),
                "skip_iteration_eval": bool(args.skip_iteration_eval),
                "resume_mode": bool(resume_mode),
                "resume_pilot_dir": str(out_root) if resume_mode else "",
                "recorded_resume_wall_time_sec": round(float(recorded_resume_wall_time), 4),
                "completed_iterations": int(len(iteration_summaries)),
            },
            ensure_ascii=False,
        ),
    )

    if bool(args.skip_baseline_eval):
        baseline_world = {}
        baseline_phase = {}
        baseline_run_dirs = []
        baseline_run_dir = ""
        baseline_summary = {}
        print(
            "[interaction-crossview-ppo-pilot][progress]",
            json.dumps(
                {
                    "event": "baseline_eval_skipped",
                    "phase_label": "baseline_eval",
                },
                ensure_ascii=False,
            ),
        )
        persist_state(status="running")
    else:
        baseline_complete = bool(baseline_summary) or rollout_phase_complete(
            baseline_root,
            task_count=len(task_names),
            episodes_per_task=int(args.eval_episodes_per_task),
        )
        if baseline_complete:
            if not baseline_world:
                baseline_world = build_resumed_world_record(
                    args=args,
                    mode=args.eval_world_mode,
                    phase_label="baseline_eval",
                    world_instances=args.eval_world_instances,
                    fixed_bank_dir=args.eval_fixed_bank_dir,
                )
            if not baseline_phase:
                baseline_phase = build_resumed_phase_record(
                    phase_label="baseline_eval",
                    phase_root=baseline_root,
                    episodes_per_task_total=int(args.eval_episodes_per_task),
                )
            if not baseline_run_dirs:
                baseline_run_dirs = list(baseline_phase["run_dirs"])
            if not baseline_run_dir and baseline_run_dirs:
                baseline_run_dir = str(baseline_run_dirs[0])
            if not baseline_summary:
                baseline_summary = load_aggregated_summary(baseline_root)
            if not latest_targeted_rows:
                latest_targeted_rows = aggregate_targeted_rows_with_backend(baseline_root, args, env)
            if resume_mode:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_resume",
                            "phase_label": "baseline_eval",
                            "phase_root": str(baseline_root),
                        },
                        ensure_ascii=False,
                    ),
                )
            persist_state(status="running")
        else:
            archived_baseline = None
            if resume_mode and has_phase_artifacts(baseline_root):
                archived_baseline = archive_incomplete_phase_root(baseline_root)
                baseline_root.mkdir(parents=True, exist_ok=True)
            if archived_baseline is not None:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_archive_incomplete",
                            "phase_label": "baseline_eval",
                            "phase_root": str(baseline_root),
                            "archived_root": str(archived_baseline),
                        },
                        ensure_ascii=False,
                    ),
                )
            baseline_world = resolve_phase_world(
                args=args,
                task_specs=task_specs,
                mode=args.eval_world_mode,
                iteration_idx=0,
                phase_label="baseline_eval",
                factor_split_label=args.eval_factor_split,
                plan_rows_by_task=plan_rows_by_task,
                targeted_rows_by_task=latest_targeted_rows,
                env=env,
                worldgen_root=worldgen_root,
                world_instances=args.eval_world_instances,
                fixed_bank_dir=args.eval_fixed_bank_dir,
            )
            baseline_phase = execute_phase_rollouts(
                args=args,
                phase_root=baseline_root,
                model_path=current_model_path,
                episodes_per_task=args.eval_episodes_per_task,
                save_fragments=False,
                phase_world=baseline_world,
                env=env,
                parallel_workers=max(1, int(args.eval_parallel_workers)),
            )
            baseline_run_dirs = list(baseline_phase["run_dirs"])
            baseline_run_dir = str(baseline_run_dirs[0])
            baseline_summary = load_aggregated_summary(baseline_root)
            latest_targeted_rows = aggregate_targeted_rows_with_backend(baseline_root, args, env)
            persist_state(status="running")

    start_iteration = int(len(iteration_summaries) + 1)
    for iteration_idx in range(start_iteration, int(args.train_iters) + 1):
        iteration_started_at = time.perf_counter()
        print(
            "[interaction-crossview-ppo-pilot][progress]",
            json.dumps(
                {
                    "event": "iteration_start",
                    "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                    "iteration_idx": int(iteration_idx),
                },
                ensure_ascii=False,
            ),
        )
        iter_dir = iterations_root / f"iter_{iteration_idx:03d}"
        collect_root = iter_dir / "collect"
        update_root = iter_dir / "update"
        eval_root = iter_dir / "eval"
        for root in (collect_root, update_root, eval_root):
            root.mkdir(parents=True, exist_ok=True)

        collect_phase_label = f"iter_{iteration_idx:03d}_collect"
        collect_complete = rollout_phase_complete(
            collect_root,
            task_count=len(task_names),
            episodes_per_task=int(args.collect_episodes_per_task),
        )
        if collect_complete:
            collect_world = build_resumed_world_record(
                args=args,
                mode=args.collect_world_mode,
                phase_label=collect_phase_label,
                world_instances=args.collect_world_instances,
                fixed_bank_dir=args.collect_fixed_bank_dir,
            )
            collect_phase = build_resumed_phase_record(
                phase_label=collect_phase_label,
                phase_root=collect_root,
                episodes_per_task_total=int(args.collect_episodes_per_task),
            )
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "phase_resume",
                        "phase_label": collect_phase_label,
                        "phase_root": str(collect_root),
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            archived_collect = None
            if resume_mode and has_phase_artifacts(collect_root):
                archived_collect = archive_incomplete_phase_root(collect_root)
                collect_root.mkdir(parents=True, exist_ok=True)
            if archived_collect is not None:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_archive_incomplete",
                            "phase_label": collect_phase_label,
                            "phase_root": str(collect_root),
                            "archived_root": str(archived_collect),
                        },
                        ensure_ascii=False,
                    ),
                )
            collect_world = resolve_phase_world(
                args=args,
                task_specs=task_specs,
                mode=args.collect_world_mode,
                iteration_idx=iteration_idx,
                phase_label=collect_phase_label,
                factor_split_label=args.collect_factor_split,
                plan_rows_by_task=plan_rows_by_task,
                targeted_rows_by_task=latest_targeted_rows,
                env=env,
                worldgen_root=worldgen_root,
                world_instances=args.collect_world_instances,
                fixed_bank_dir=args.collect_fixed_bank_dir,
            )
            collect_phase = execute_phase_rollouts(
                args=args,
                phase_root=collect_root,
                model_path=current_model_path,
                episodes_per_task=args.collect_episodes_per_task,
                save_fragments=True,
                phase_world=collect_world,
                env=env,
                parallel_workers=max(1, int(args.collect_parallel_workers)),
            )
        collect_run_dirs = list(collect_phase["run_dirs"])
        collect_run_dir = collect_run_dirs[0]
        latest_targeted_rows = aggregate_targeted_rows_with_backend(collect_root, args, env)
        collect_episode_stats = summarize_rollout_episode_rows(collect_root)
        update_input_root = collect_root
        collect_quota_selection: Dict[str, object] = {}
        update_episode_stats = dict(collect_episode_stats)
        if str(args.collect_quota_mode or "none") != "none":
            update_input_root, collect_quota_selection = build_collect_quota_update_input(
                args=args,
                collect_root=collect_root,
                collect_phase=collect_phase,
                iteration_idx=iteration_idx,
            )
            update_episode_stats = dict(
                collect_quota_selection.get("selected_episode_stats") or summarize_rollout_episode_rows(update_input_root)
            )

        update_started_at = time.perf_counter()
        update_run_dir: Optional[Path] = None
        updated_model_path = Path(current_model_path)
        update_skipped = False
        existing_update_checkpoint = latest_model_checkpoint(update_root)
        min_successful_fragments = max(0, int(args.min_successful_fragments))
        if int(update_episode_stats["successful_episodes"]) < min_successful_fragments:
            update_skipped = True
            update_wall_time = 0.0
            if existing_update_checkpoint is None and resume_mode and has_phase_artifacts(update_root):
                archived_update = archive_incomplete_phase_root(update_root)
                update_root.mkdir(parents=True, exist_ok=True)
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_archive_incomplete",
                            "phase_label": f"iter_{iteration_idx:03d}_update",
                            "phase_root": str(update_root),
                            "archived_root": str(archived_update),
                        },
                        ensure_ascii=False,
                    ),
                )
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "update_skipped",
                        "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                        "iteration_idx": int(iteration_idx),
                        "successful_episodes": int(update_episode_stats["successful_episodes"]),
                        "episodes": int(update_episode_stats["episodes"]),
                        "raw_collect_successful_episodes": int(collect_episode_stats["successful_episodes"]),
                        "raw_collect_episodes": int(collect_episode_stats["episodes"]),
                        "min_successful_fragments": int(min_successful_fragments),
                        "reason": "insufficient_successful_fragments",
                        "update_input_root": str(update_input_root),
                        "model_path_kept": str(updated_model_path),
                    },
                    ensure_ascii=False,
                ),
            )
        elif existing_update_checkpoint is not None:
            update_wall_time = 0.0
            update_run_dir = existing_update_checkpoint.parent
            updated_model_path = existing_update_checkpoint
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "update_resume",
                        "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                        "iteration_idx": int(iteration_idx),
                        "update_run_dir": str(update_run_dir),
                        "model_path": str(updated_model_path),
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "update_start",
                        "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                        "iteration_idx": int(iteration_idx),
                        "successful_episodes": int(update_episode_stats["successful_episodes"]),
                        "episodes": int(update_episode_stats["episodes"]),
                        "raw_collect_successful_episodes": int(collect_episode_stats["successful_episodes"]),
                        "raw_collect_episodes": int(collect_episode_stats["episodes"]),
                        "update_input_root": str(update_input_root),
                    },
                    ensure_ascii=False,
                ),
            )
            archived_update = None
            if resume_mode and has_phase_artifacts(update_root):
                archived_update = archive_incomplete_phase_root(update_root)
                update_root.mkdir(parents=True, exist_ok=True)
            if archived_update is not None:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_archive_incomplete",
                            "phase_label": f"iter_{iteration_idx:03d}_update",
                            "phase_root": str(update_root),
                            "archived_root": str(archived_update),
                        },
                        ensure_ascii=False,
                    ),
                )
            run_command(build_update_cmd(args, update_input_root, update_root, current_model_path, int(iteration_idx)), env)
            update_wall_time = time.perf_counter() - update_started_at
            update_run_dir = latest_subdir(update_root)
            updated_model_path = update_run_dir / "model.pt"
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "update_done",
                        "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                        "iteration_idx": int(iteration_idx),
                        "update_run_dir": str(update_run_dir),
                        "wall_time_sec": round(float(update_wall_time), 4),
                        "wall_time_human": format_seconds(update_wall_time),
                    },
                    ensure_ascii=False,
                ),
            )

        eval_phase_label = f"iter_{iteration_idx:03d}_eval"
        eval_world = {}
        eval_phase = {}
        eval_run_dirs = []
        eval_run_dir = ""
        eval_summary = {}
        if bool(args.skip_iteration_eval):
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "iteration_eval_skipped",
                        "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                        "iteration_idx": int(iteration_idx),
                        "phase_label": eval_phase_label,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            eval_complete = rollout_phase_complete(
                eval_root,
                task_count=len(task_names),
                episodes_per_task=int(args.eval_episodes_per_task),
            )
            if eval_complete:
                eval_world = build_resumed_world_record(
                    args=args,
                    mode=args.eval_world_mode,
                    phase_label=eval_phase_label,
                    world_instances=args.eval_world_instances,
                    fixed_bank_dir=args.eval_fixed_bank_dir,
                )
                eval_phase = build_resumed_phase_record(
                    phase_label=eval_phase_label,
                    phase_root=eval_root,
                    episodes_per_task_total=int(args.eval_episodes_per_task),
                )
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_resume",
                            "phase_label": eval_phase_label,
                            "phase_root": str(eval_root),
                        },
                        ensure_ascii=False,
                    ),
                )
            else:
                archived_eval = None
                if resume_mode and has_phase_artifacts(eval_root):
                    archived_eval = archive_incomplete_phase_root(eval_root)
                    eval_root.mkdir(parents=True, exist_ok=True)
                if archived_eval is not None:
                    print(
                        "[interaction-crossview-ppo-pilot][progress]",
                        json.dumps(
                            {
                                "event": "phase_archive_incomplete",
                                "phase_label": eval_phase_label,
                                "phase_root": str(eval_root),
                                "archived_root": str(archived_eval),
                            },
                            ensure_ascii=False,
                        ),
                    )
                eval_world = resolve_phase_world(
                    args=args,
                    task_specs=task_specs,
                    mode=args.eval_world_mode,
                    iteration_idx=iteration_idx,
                    phase_label=eval_phase_label,
                    factor_split_label=args.eval_factor_split,
                    plan_rows_by_task=plan_rows_by_task,
                    targeted_rows_by_task=latest_targeted_rows,
                    env=env,
                    worldgen_root=worldgen_root,
                    world_instances=args.eval_world_instances,
                    fixed_bank_dir=args.eval_fixed_bank_dir,
                )
                eval_phase = execute_phase_rollouts(
                    args=args,
                    phase_root=eval_root,
                    model_path=str(updated_model_path),
                    episodes_per_task=args.eval_episodes_per_task,
                    save_fragments=False,
                    phase_world=eval_world,
                    env=env,
                    parallel_workers=max(1, int(args.eval_parallel_workers)),
                )
            eval_run_dirs = list(eval_phase["run_dirs"])
            eval_run_dir = str(eval_run_dirs[0]) if eval_run_dirs else ""
            eval_summary = load_aggregated_summary(eval_root)
        iteration_wall_time = time.perf_counter() - iteration_started_at

        iteration_summaries.append(
            {
                "iteration": iteration_idx,
                "input_model_path": str(current_model_path),
                "collect_world": collect_world,
                "collect_run_dir": str(collect_run_dir),
                "collect_run_dirs": [str(path) for path in collect_run_dirs],
                "collect_run_root": str(collect_root),
                "collect_phase_timing": collect_phase,
                "collect_episode_stats": collect_episode_stats,
                "collect_quota_selection": collect_quota_selection,
                "update_input_run_root": str(update_input_root),
                "update_episode_stats": update_episode_stats,
                "update_skipped": bool(update_skipped),
                "update_run_dir": str(update_run_dir) if update_run_dir is not None else "",
                "update_wall_time_sec": round(float(update_wall_time), 4),
                "eval_world": eval_world,
                "eval_run_dir": str(eval_run_dir),
                "eval_run_dirs": [str(path) for path in eval_run_dirs],
                "eval_run_root": str(eval_root),
                "eval_phase_timing": eval_phase,
                "updated_model_path": str(updated_model_path),
                "eval_summary": eval_summary,
                "iteration_wall_time_sec": round(float(iteration_wall_time), 4),
            }
        )
        print(
            "[interaction-crossview-ppo-pilot][progress]",
            json.dumps(
                {
                    "event": "iteration_done",
                    "iteration_progress": f"{iteration_idx}/{int(args.train_iters)}",
                    "iteration_idx": int(iteration_idx),
                    "iteration_wall_time_sec": round(float(iteration_wall_time), 4),
                    "iteration_wall_time_human": format_seconds(iteration_wall_time),
                    "remaining_iterations": int(max(0, int(args.train_iters) - iteration_idx)),
                },
                ensure_ascii=False,
            ),
        )
        if (not bool(args.skip_iteration_eval)) and args.eval_world_mode != "static":
            latest_targeted_rows = aggregate_targeted_rows_with_backend(eval_root, args, env)
        current_model_path = str(updated_model_path)
        best_eval_selection = select_best_eval_model_selection(iteration_summaries, fallback_model_path=current_model_path)
        final_eval_model_selection = select_final_eval_model_selection(
            args.final_eval_model_mode,
            iteration_summaries,
            current_model_path,
        )
        persist_state(status="running")

    selected_final_eval_model = final_eval_model_selection or select_final_eval_model_selection(
        args.final_eval_model_mode,
        iteration_summaries,
        current_model_path,
    )
    if bool(args.skip_final_eval):
        final_eval_model_selection = selected_final_eval_model
        final_eval_world = {}
        final_eval_phase = {}
        final_eval_run_dirs = []
        final_eval_run_dir = ""
        final_eval_summary = {}
        print(
            "[interaction-crossview-ppo-pilot][progress]",
            json.dumps(
                {
                    "event": "final_eval_skipped",
                    "mode": str(final_eval_model_selection.get("mode", "")),
                    "model_path": str(final_eval_model_selection.get("model_path", current_model_path)),
                    "iteration": int(final_eval_model_selection.get("iteration", 0) or 0),
                },
                ensure_ascii=False,
            ),
        )
    else:
        final_eval_complete = bool(final_eval_summary) or rollout_phase_complete(
            final_eval_root,
            task_count=len(task_names),
            episodes_per_task=int(args.eval_episodes_per_task),
        )
        if final_eval_complete:
            if not final_eval_world:
                final_eval_world = build_resumed_world_record(
                    args=args,
                    mode=final_eval_world_mode,
                    phase_label="final_eval",
                    world_instances=(args.final_eval_world_instances or args.eval_world_instances),
                    fixed_bank_dir=args.final_eval_fixed_bank_dir or args.eval_fixed_bank_dir,
                )
            if not final_eval_phase:
                final_eval_phase = build_resumed_phase_record(
                    phase_label="final_eval",
                    phase_root=final_eval_root,
                    episodes_per_task_total=int(args.eval_episodes_per_task),
                )
            if not final_eval_run_dirs:
                final_eval_run_dirs = list(final_eval_phase["run_dirs"])
            if not final_eval_run_dir and final_eval_run_dirs:
                final_eval_run_dir = str(final_eval_run_dirs[0])
            if not final_eval_summary:
                final_eval_summary = load_aggregated_summary(final_eval_root)
            if resume_mode:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_resume",
                            "phase_label": "final_eval",
                            "phase_root": str(final_eval_root),
                        },
                        ensure_ascii=False,
                    ),
                )
        else:
            final_eval_model_selection = selected_final_eval_model
            print(
                "[interaction-crossview-ppo-pilot][progress]",
                json.dumps(
                    {
                        "event": "final_eval_model_selected",
                        "mode": str(final_eval_model_selection.get("mode", "")),
                        "model_path": str(final_eval_model_selection.get("model_path", current_model_path)),
                        "iteration": int(final_eval_model_selection.get("iteration", 0) or 0),
                        "auto_success_rate": float(final_eval_model_selection.get("auto_success_rate", 0.0) or 0.0),
                        "mean_reward": float(final_eval_model_selection.get("mean_reward", 0.0) or 0.0),
                    },
                    ensure_ascii=False,
                ),
            )
            archived_final_eval = None
            if resume_mode and has_phase_artifacts(final_eval_root):
                archived_final_eval = archive_incomplete_phase_root(final_eval_root)
                final_eval_root.mkdir(parents=True, exist_ok=True)
            if archived_final_eval is not None:
                print(
                    "[interaction-crossview-ppo-pilot][progress]",
                    json.dumps(
                        {
                            "event": "phase_archive_incomplete",
                            "phase_label": "final_eval",
                            "phase_root": str(final_eval_root),
                            "archived_root": str(archived_final_eval),
                        },
                        ensure_ascii=False,
                    ),
                )
            final_eval_world = resolve_phase_world(
                args=args,
                task_specs=task_specs,
                mode=final_eval_world_mode,
                iteration_idx=int(args.train_iters),
                phase_label="final_eval",
                factor_split_label=args.final_eval_factor_split or args.eval_factor_split,
                plan_rows_by_task=plan_rows_by_task,
                targeted_rows_by_task=latest_targeted_rows,
                env=env,
                worldgen_root=worldgen_root,
                world_instances=(args.final_eval_world_instances or args.eval_world_instances),
                fixed_bank_dir=args.final_eval_fixed_bank_dir or args.eval_fixed_bank_dir,
            )
            final_eval_phase = execute_phase_rollouts(
                args=args,
                phase_root=final_eval_root,
                model_path=str(final_eval_model_selection.get("model_path", current_model_path)),
                episodes_per_task=args.eval_episodes_per_task,
                save_fragments=False,
                phase_world=final_eval_world,
                env=env,
                parallel_workers=max(1, int(args.final_eval_parallel_workers or args.eval_parallel_workers)),
            )
            final_eval_run_dirs = list(final_eval_phase["run_dirs"])
            final_eval_run_dir = str(final_eval_run_dirs[0])
            final_eval_summary = load_aggregated_summary(final_eval_root)

    total_pilot_wall_time = recorded_resume_wall_time + (time.perf_counter() - pilot_started_at)

    pilot_summary = {
        "config": redacted_args_dict(args),
        "timing": {
            "planned_rollout_episodes_total": int(planned_rollout_episodes_total(args)),
            "baseline_phase": baseline_phase,
            "final_eval_phase": final_eval_phase,
            "total_pilot_wall_time_sec": round(float(total_pilot_wall_time), 4),
            "total_pilot_wall_time_human": format_seconds(total_pilot_wall_time),
        },
        "latest_targeted_rows": latest_targeted_rows,
        "targeted_review_backend": args.targeted_review_backend,
        "targeted_review_model_id": args.targeted_review_model_id,
        "baseline_world": baseline_world,
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_run_dirs": [str(path) for path in baseline_run_dirs],
        "baseline_run_root": str(baseline_root),
        "baseline_summary": baseline_summary,
        "iteration_summaries": iteration_summaries,
        "best_eval_selection": best_eval_selection,
        "final_model_path": current_model_path,
        "final_eval_world": final_eval_world,
        "final_eval_run_dir": str(final_eval_run_dir),
        "final_eval_run_dirs": [str(path) for path in final_eval_run_dirs],
        "final_eval_run_root": str(final_eval_root),
        "final_eval_model_selection": final_eval_model_selection,
        "final_eval_summary": final_eval_summary,
    }
    pilot_summary = normalize_jsonable(_slim_state_for_disk(pilot_summary))
    write_pilot_state(out_root, {**pilot_summary, "status": "done", "current_model_path": str(current_model_path)})
    (out_root / "pilot_summary.json").write_text(
        json.dumps(pilot_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[interaction-crossview-ppo-pilot] done")
    print(
        "[interaction-crossview-ppo-pilot][progress]",
        json.dumps(
            {
                "event": "pilot_done",
                "out_root": str(out_root),
                "total_pilot_wall_time_sec": round(float(total_pilot_wall_time), 4),
                "total_pilot_wall_time_human": format_seconds(total_pilot_wall_time),
            },
            ensure_ascii=False,
        ),
    )
    print(json.dumps(pilot_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
