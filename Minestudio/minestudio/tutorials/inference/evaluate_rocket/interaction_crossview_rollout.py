import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np
import torch
import yaml

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.crossview_utils import (
    AUTO_GOAL_SUPPORTED_TASKS,
    CrossViewSession,
    load_goal_spec,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_protocol import (
    apply_protocol_to_task_specs,
    resolve_interaction_protocol,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    InteractionBenchmarkTaskSpec,
    resolve_interaction_task_specs,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_posttrain_rewards import (
    InteractionPostTrainRewardTracker,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_success_checker import (
    InteractionBenchmarkSuccessTracker,
)
from minestudio.tutorials.inference.evaluate_rocket.success_checker import inventory_totals


def sanitize_mapping(data):
    if isinstance(data, Mapping):
        return {str(k): sanitize_mapping(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_mapping(item) for item in data]
    if isinstance(data, tuple):
        return [sanitize_mapping(item) for item in data]
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, np.generic):
            return data.item()
    except Exception:
        pass
    return data


def _counter_scalar(value) -> float:
    try:
        if isinstance(value, np.ndarray):
            return float(value.item())
        if isinstance(value, np.generic):
            return float(value.item())
        if isinstance(value, (int, float)):
            return float(value)
    except Exception:
        return 0.0
    return 0.0


def counter_delta(current, initial) -> Dict[str, float]:
    current = current if isinstance(current, Mapping) else {}
    initial = initial if isinstance(initial, Mapping) else {}
    keys = sorted({str(key) for key in current.keys()} | {str(key) for key in initial.keys()})
    deltas: Dict[str, float] = {}
    for key in keys:
        delta = _counter_scalar(current.get(key, 0.0)) - _counter_scalar(initial.get(key, 0.0))
        if abs(delta) > 1e-9:
            deltas[key] = int(delta) if float(delta).is_integer() else float(delta)
    return deltas


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_video(path: Path, frames: List):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def format_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, seconds = divmod(int(round(total)), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def resolve_sampling_seed_step(args) -> Optional[int]:
    if args.sampling_base_seed is None:
        return None
    if args.sampling_seed_step is None:
        return int(args.seed_step)
    return int(args.sampling_seed_step)


def compute_episode_sampling_seed(args, episode_idx: int) -> Optional[int]:
    if args.sampling_base_seed is None:
        return None
    sampling_seed_step = resolve_sampling_seed_step(args)
    return int(args.sampling_base_seed + int(episode_idx) * int(sampling_seed_step))


def set_sampling_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int % (2 ** 32))
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def annotate_frame(frame, benchmark_name: str, goal_image, goal_mask, goal_segment_type: str, global_step: Optional[int] = None):
    annotated = frame.copy()
    if goal_image is not None:
        inset = cv2.resize(goal_image, (160, 160), interpolation=cv2.INTER_LINEAR)
        inset_rgb = inset.copy()
        if goal_mask is not None:
            inset_mask = cv2.resize((np.asarray(goal_mask) > 0).astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
            if np.any(inset_mask > 0):
                inset_float = inset_rgb.astype(np.float32)
                mask_bool = inset_mask > 0
                inset_float[mask_bool] = 0.55 * inset_float[mask_bool] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
                inset_rgb = np.clip(inset_float, 0.0, 255.0).astype(np.uint8)
        y0, x0 = 8, annotated.shape[1] - 168
        annotated[y0 : y0 + 160, x0 : x0 + 160] = inset_rgb
        cv2.rectangle(annotated, (x0, y0), (x0 + 160, y0 + 160), (255, 255, 255), 2)
    lines = [benchmark_name, f"goal={goal_segment_type}"]
    if global_step is not None:
        lines.append(f"step={global_step}")
    overlay = annotated.copy()
    line_height = 18
    width = max(220, min(annotated.shape[1] - 20, 12 * max((len(line) for line in lines), default=10)))
    height = max(28, 14 + line_height * len(lines))
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
    for idx, line in enumerate(lines):
        y = 28 + idx * line_height
        cv2.putText(annotated, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def write_annotated_video(path: Path, frames: List, benchmark_name: str, goal_image, goal_mask, goal_segment_type: str):
    annotated_frames = []
    for frame_idx, frame in enumerate(frames):
        annotated_frames.append(
            annotate_frame(
                frame=frame,
                benchmark_name=benchmark_name,
                goal_image=goal_image,
                goal_mask=goal_mask,
                goal_segment_type=goal_segment_type,
                global_step=frame_idx + 1,
            )
        )
    write_video(path, annotated_frames)


def _clone_tensor_tree(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().clone()
    if isinstance(data, Mapping):
        return {str(key): _clone_tensor_tree(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_clone_tensor_tree(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_clone_tensor_tree(value) for value in data)
    return data


def _stack_tensor_tree(items):
    if not items:
        raise ValueError("Cannot stack an empty tensor tree.")
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items, dim=0)
    if isinstance(first, Mapping):
        return {
            key: _stack_tensor_tree([item[key] for item in items])
            for key in first.keys()
        }
    if isinstance(first, list):
        return [_stack_tensor_tree([item[idx] for item in items]) for idx in range(len(first))]
    if isinstance(first, tuple):
        return tuple(_stack_tensor_tree([item[idx] for item in items]) for idx in range(len(first)))
    return list(items)


def build_episode_fragment(
    task_spec: InteractionBenchmarkTaskSpec,
    seed: int,
    fragment_steps: List[Dict],
    trainable: bool,
    stop_reason: str,
    bootstrap_value_raw: float = 0.0,
    bootstrap_value_norm: float = 0.0,
    bootstrap_valid: bool = False,
    bootstrap_reason: str = "",
):
    if not fragment_steps:
        return None

    payload = {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "task_key": task_spec.task_key,
        "seed": int(seed),
        "trainable": bool(trainable),
        "stop_reason": stop_reason,
        "sequence_length": int(len(fragment_steps)),
        "image": _stack_tensor_tree([step["image"] for step in fragment_steps]),
        "obj_mask": _stack_tensor_tree([step["obj_mask"] for step in fragment_steps]),
        "obj_id": _stack_tensor_tree([step["obj_id"] for step in fragment_steps]).long(),
        "action": _stack_tensor_tree([step["action"] for step in fragment_steps]),
        "old_logprob": _stack_tensor_tree([step["old_logprob"] for step in fragment_steps]).float(),
        "old_value": _stack_tensor_tree([step["old_value"] for step in fragment_steps]).float(),
        "old_value_norm": _stack_tensor_tree([step["old_value_norm"] for step in fragment_steps]).float(),
        "old_value_raw": _stack_tensor_tree([step["old_value_raw"] for step in fragment_steps]).float(),
        "reward": _stack_tensor_tree([step["reward"] for step in fragment_steps]).float(),
        "done": _stack_tensor_tree([step["done"] for step in fragment_steps]).bool(),
        "env_done": _stack_tensor_tree([step["env_done"] for step in fragment_steps]).bool(),
        "terminated": _stack_tensor_tree([step["terminated"] for step in fragment_steps]).bool(),
        "truncated": _stack_tensor_tree([step["truncated"] for step in fragment_steps]).bool(),
        "first": _stack_tensor_tree([step["first"] for step in fragment_steps]).bool(),
        "subtask_index": _stack_tensor_tree([step["subtask_index"] for step in fragment_steps]).long(),
        "interaction_type": [str(step["interaction_type"]) for step in fragment_steps],
        "prompt_text": [str(step["prompt_text"]) for step in fragment_steps],
        "segment_area": _stack_tensor_tree([step["segment_area"] for step in fragment_steps]).long(),
        "bootstrap_value": float(bootstrap_value_raw),
        "bootstrap_value_raw": float(bootstrap_value_raw),
        "bootstrap_value_norm": float(bootstrap_value_norm),
        "bootstrap_valid": bool(bootstrap_valid),
        "bootstrap_reason": str(bootstrap_reason),
    }
    optional_tensor_keys = [
        "cross_view_image",
        "cross_view_obj_mask",
        "cross_view_obj_id",
        "env_prev_action",
    ]
    for key in optional_tensor_keys:
        if key in fragment_steps[0]:
            payload[key] = _stack_tensor_tree([step[key] for step in fragment_steps])
    return payload


def compute_fragment_bootstrap(session: CrossViewSession, stop_reason: str):
    if bool(session.last_terminated):
        return 0.0, 0.0, False, "terminated"
    if bool(session.last_truncated) or str(stop_reason) == "max_steps":
        value = session.estimate_current_policy_value()
        if value is not None:
            reason = "truncated" if bool(session.last_truncated) else "max_steps"
            return float(value.get("value_raw", 0.0)), float(value.get("value_norm", 0.0)), True, reason
    return 0.0, 0.0, False, str(stop_reason)


def capture_fragment_step(session: CrossViewSession, global_step: int, subtask_index: int, interaction_type: str, prompt_text: str, rl_reward: float):
    if session.last_model_input is None or session.last_policy_action is None:
        return None
    if (
        session.last_policy_logprob is None
        or session.last_policy_value is None
        or session.last_policy_value_raw is None
    ):
        return None
    return {
        "global_step": int(global_step),
        "subtask_index": torch.tensor(int(subtask_index), dtype=torch.long),
        "interaction_type": interaction_type,
        "prompt_text": prompt_text,
        "image": torch.from_numpy(session.last_model_input["image"]).to(dtype=torch.uint8),
        "obj_mask": session.last_model_input["obj_mask"].to(dtype=torch.uint8),
        "obj_id": torch.tensor(int(session.last_model_input["obj_id"]), dtype=torch.long),
        "cross_view_image": session.last_model_input["cross_view_image"].to(dtype=torch.uint8),
        "cross_view_obj_mask": session.last_model_input["cross_view_obj_mask"].to(dtype=torch.uint8),
        "cross_view_obj_id": torch.tensor(int(session.last_model_input["cross_view_obj_id"]), dtype=torch.long),
        "env_prev_action": _clone_tensor_tree(session.last_model_input["env_prev_action"]),
        "action": _clone_tensor_tree(session.last_policy_action),
        "old_logprob": torch.tensor(float(session.last_policy_logprob), dtype=torch.float32),
        "old_value": torch.tensor(float(session.last_policy_value), dtype=torch.float32),
        "old_value_norm": torch.tensor(float(session.last_policy_value), dtype=torch.float32),
        "old_value_raw": torch.tensor(float(session.last_policy_value_raw), dtype=torch.float32),
        "reward": torch.tensor(float(rl_reward), dtype=torch.float32),
        "done": torch.tensor(bool(session.last_terminated or session.last_truncated), dtype=torch.bool),
        "env_done": torch.tensor(bool(session.last_terminated or session.last_truncated), dtype=torch.bool),
        "terminated": torch.tensor(bool(session.last_terminated), dtype=torch.bool),
        "truncated": torch.tensor(bool(session.last_truncated), dtype=torch.bool),
        "first": torch.tensor(bool(global_step == 1), dtype=torch.bool),
        "segment_area": torch.tensor(int(session.last_segment_area), dtype=torch.long),
    }


def capture_step_snapshot(
    session: CrossViewSession,
    global_step: int,
    subtask_index: int,
    target_text: str,
    prompt_text: str,
    interaction_type: str,
    rl_reward: float,
    reward_reason: str,
    reward_metrics: Dict[str, float],
    initial_info: Optional[Mapping] = None,
):
    info = session.info
    initial_info = initial_info if isinstance(initial_info, Mapping) else {}
    mine_block = sanitize_mapping(info.get("mine_block", {}))
    pickup = sanitize_mapping(info.get("pickup", {}))
    use_item = sanitize_mapping(info.get("use_item", {}))
    craft_item = sanitize_mapping(info.get("craft_item", {}))
    kill_entity = sanitize_mapping(info.get("kill_entity", {}))
    player_pos = sanitize_mapping(info.get("player_pos") or info.get("location_stats") or {})
    return {
        "global_step": global_step,
        "subtask_index": subtask_index,
        "target_text": target_text,
        "prompt_text": prompt_text,
        "interaction_type": interaction_type,
        "env_reward": float(session.last_reward),
        "rl_reward": float(rl_reward),
        "reward_reason": reward_reason,
        "reward_metrics": reward_metrics,
        "terminated": bool(session.last_terminated),
        "truncated": bool(session.last_truncated),
        "last_action_summary": session.last_action_summary,
        "policy_logprob": session.last_policy_logprob,
        "policy_value": session.last_policy_value,
        "policy_value_norm": session.last_policy_value,
        "policy_value_raw": session.last_policy_value_raw,
        "segment_area": int(session.last_segment_area),
        "player_pos": player_pos,
        "inventory_totals": inventory_totals(info),
        "initial_use_item": sanitize_mapping(initial_info.get("use_item", {})),
        "initial_pickup": sanitize_mapping(initial_info.get("pickup", {})),
        "initial_craft_item": sanitize_mapping(initial_info.get("craft_item", {})),
        "initial_mine_block": sanitize_mapping(initial_info.get("mine_block", {})),
        "initial_kill_entity": sanitize_mapping(initial_info.get("kill_entity", {})),
        "use_item": use_item,
        "pickup": pickup,
        "craft_item": craft_item,
        "mine_block": mine_block,
        "kill_entity": kill_entity,
        "delta_use_item": counter_delta(use_item, initial_info.get("use_item", {})),
        "delta_pickup": counter_delta(pickup, initial_info.get("pickup", {})),
        "delta_craft_item": counter_delta(craft_item, initial_info.get("craft_item", {})),
        "delta_mine_block": counter_delta(mine_block, initial_info.get("mine_block", {})),
        "delta_kill_entity": counter_delta(kill_entity, initial_info.get("kill_entity", {})),
        "equipped_items": sanitize_mapping(info.get("equipped_items", {})),
    }


def run_crossview_episode(
    task_spec: InteractionBenchmarkTaskSpec,
    seed: int,
    session: CrossViewSession,
    warmup_noop_steps: int,
    stop_on_success: bool,
    step_budget_override: Optional[int] = None,
    env_reward_scale: float = 1.0,
):
    if len(task_spec.subtasks) != 1:
        raise NotImplementedError("ROCKET-2 fixed-goal rollout currently supports single-subtask tasks only.")
    subtask = task_spec.subtasks[0]
    if subtask.interaction_type != session.goal_segment_type:
        raise ValueError(
            f"Goal segment_type '{session.goal_segment_type}' does not match task interaction '{subtask.interaction_type}'."
        )

    reset_started_at = time.perf_counter()
    session.reset(task_spec.task_config_name, seed, warmup_noop_steps=warmup_noop_steps)
    reset_wall_time = time.perf_counter() - reset_started_at
    reset_metadata = getattr(session, "debug_assets", {}).get("metadata", {}) if hasattr(session, "debug_assets") else {}
    reset_env_reused = bool(reset_metadata.get("rollout_env_reused", False))
    reset_kind = "fast_reuse" if reset_env_reused else "initial_env"
    initial_info = copy.deepcopy(session.info)
    reward_tracker = InteractionPostTrainRewardTracker(task_spec, initial_info)
    success_tracker = InteractionBenchmarkSuccessTracker(task_spec, initial_info)
    reward_supported = reward_tracker.success_result(session.info).supported
    total_step_budget = int(sum(item.step_budget for item in task_spec.subtasks))
    if step_budget_override is not None and int(step_budget_override) > 0:
        total_step_budget = int(step_budget_override)
    episode_reward = 0.0
    trajectory_rows: List[Dict] = []
    fragment_steps: List[Dict] = []
    auto_result = success_tracker.update(session.info)
    stop_reason = ""

    for _ in range(total_step_budget):
        session.step()
        reward_event = reward_tracker.evaluate(
            session.info,
            env_reward=float(env_reward_scale) * float(session.last_reward),
        )
        auto_result = success_tracker.update(session.info)
        episode_reward += float(reward_event.reward)
        trajectory_rows.append(
            capture_step_snapshot(
                session=session,
                global_step=session.num_steps,
                subtask_index=0,
                target_text=subtask.target_text,
                prompt_text=subtask.prompt_text or subtask.target_text,
                interaction_type=subtask.interaction_type,
                rl_reward=reward_event.reward,
                reward_reason=reward_event.reason,
                reward_metrics=reward_event.metrics,
                initial_info=initial_info,
            )
        )
        fragment_step = capture_fragment_step(
            session=session,
            global_step=session.num_steps,
            subtask_index=0,
            interaction_type=subtask.interaction_type,
            prompt_text=subtask.prompt_text or subtask.target_text,
            rl_reward=reward_event.reward,
        )
        if fragment_step is not None:
            fragment_steps.append(fragment_step)

        if session.last_terminated:
            stop_reason = "terminated"
            break
        if session.last_truncated:
            stop_reason = "truncated"
            break
        if reward_event.failure:
            stop_reason = "constraint_failure"
            break
        if stop_on_success and reward_event.success:
            stop_reason = "success"
            break

    if not stop_reason:
        stop_reason = "max_steps"

    bootstrap_value_raw, bootstrap_value_norm, bootstrap_valid, bootstrap_reason = compute_fragment_bootstrap(
        session=session,
        stop_reason=stop_reason,
    )

    return {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "category": task_spec.category,
        "seed": int(seed),
        "warmup_noop_steps": int(warmup_noop_steps),
        "reset_kind": reset_kind,
        "reset_env_reused": bool(reset_env_reused),
        "reset_wall_time_sec": round(float(reset_wall_time), 4),
        "active_step_budget": int(total_step_budget),
        "num_steps": int(session.num_steps),
        "episode_reward": round(float(episode_reward), 6),
        "reward_supported": bool(reward_supported),
        "trainable": bool(reward_supported),
        "goal_segment_type": session.goal_segment_type,
        "auto_eval_supported": bool(auto_result.supported),
        "auto_success": auto_result.success,
        "auto_metric": auto_result.metric,
        "auto_progress": auto_result.progress,
        "auto_reason": auto_result.reason,
        "stop_reason": stop_reason,
        "last_action_summary": session.last_action_summary,
        "trajectory_rows": trajectory_rows,
        "ppo_fragment": build_episode_fragment(
            task_spec=task_spec,
            seed=seed,
            fragment_steps=fragment_steps,
            trainable=bool(reward_supported),
            stop_reason=stop_reason,
            bootstrap_value_raw=bootstrap_value_raw,
            bootstrap_value_norm=bootstrap_value_norm,
            bootstrap_valid=bootstrap_valid,
            bootstrap_reason=bootstrap_reason,
        ),
    }


def summarize(rows: List[Dict]):
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark_name"], []).append(row)
    summary = []
    for benchmark_name, benchmark_rows in grouped.items():
        auto_supported_rows = [row for row in benchmark_rows if row["auto_eval_supported"]]
        auto_success_rate = None
        if auto_supported_rows:
            auto_success_rate = round(
                sum(1 for row in auto_supported_rows if bool(row["auto_success"])) / len(auto_supported_rows),
                4,
            )
        summary.append(
            {
                "benchmark_name": benchmark_name,
                "episodes": len(benchmark_rows),
                "reward_supported_episodes": sum(1 for row in benchmark_rows if row["reward_supported"]),
                "trainable_episodes": sum(1 for row in benchmark_rows if row["trainable"]),
                "auto_eval_supported_episodes": len(auto_supported_rows),
                "auto_success_rate": auto_success_rate,
                "mean_reward": round(sum(float(row["episode_reward"]) for row in benchmark_rows) / len(benchmark_rows), 4),
                "mean_steps": round(sum(int(row["num_steps"]) for row in benchmark_rows) / len(benchmark_rows), 2),
                "mean_core_episode_wall_time_sec": round(
                    sum(float(row.get("core_episode_wall_time_sec", 0.0) or 0.0) for row in benchmark_rows) / len(benchmark_rows),
                    4,
                ),
                "mean_episode_wall_time_sec": round(
                    sum(float(row.get("episode_wall_time_sec", 0.0) or 0.0) for row in benchmark_rows) / len(benchmark_rows),
                    4,
                ),
                "mean_video_write_wall_time_sec": round(
                    sum(float(row.get("video_write_wall_time_sec", 0.0) or 0.0) for row in benchmark_rows) / len(benchmark_rows),
                    4,
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-group", type=str, default="rocket2_official")
    parser.add_argument("--task-group-path", type=str, default="/home/gyulab/envgen2/ROCKET-2/env_conf")
    parser.add_argument("--env-source", type=str, default="rocket2_official", choices=["rocket2_official"])
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--tasks", type=str, required=True, help="Single semantic task key, e.g. mine_emerald.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--warmup-noop-steps", type=int, default=None)
    parser.add_argument("--step-budget-override", type=int, default=0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--save-ppo-fragments", action="store_true")
    parser.add_argument("--save-debug-assets", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/interaction_crossview_rollout")
    parser.add_argument("--refresh-task-configs", action="store_true")
    parser.add_argument("--episode-retries", type=int, default=2)
    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal-spec", type=str)
    goal_group.add_argument("--auto-goal", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--sampling-base-seed", type=int, default=None)
    parser.add_argument("--sampling-seed-step", type=int, default=None)
    parser.add_argument("--cfg-coef", type=float, default=None)
    parser.add_argument("--env-reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg-policy-mode",
        type=str,
        default="full",
        choices=["full", "frozen_base"],
    )
    parser.add_argument("--cfg-base-ref-model-path", type=str, default="")
    return parser


def _task_yaml_path(task_group_path: str, task_config_name: str) -> Path:
    return Path(task_group_path) / f"{task_config_name}.yaml"


def resolve_goal_spec_for_task(
    args,
    task_spec: InteractionBenchmarkTaskSpec,
    explicit_goal_spec: Optional[Dict],
) -> tuple[Optional[Dict], str]:
    if explicit_goal_spec is not None:
        return explicit_goal_spec, "fixed"
    if not args.auto_goal:
        return None, "none"
    task_yaml_path = _task_yaml_path(args.task_group_path, task_spec.task_config_name)
    if not task_yaml_path.exists():
        return None, "auto"
    try:
        task_config = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, "auto"
    baked_goal_spec_path = str(task_config.get("baked_goal_spec_path") or "").strip()
    if not baked_goal_spec_path:
        return None, "auto"
    goal_spec_path = Path(baked_goal_spec_path)
    if not goal_spec_path.exists():
        return None, "auto"
    baked_goal_spec = dict(load_goal_spec(goal_spec_path))
    baked_goal_spec.setdefault("goal_source", "baked_task_group")
    baked_goal_spec.setdefault("baked_goal_spec_path", str(goal_spec_path.resolve()))
    return baked_goal_spec, "baked"


def _apply_fixed_goal_metadata(session: CrossViewSession, goal_spec: Dict) -> None:
    fixed_metadata = {
        "goal_mode": "fixed_asset",
        "goal_image_path": str(goal_spec["goal_image_path"]),
        "goal_mask_path": str(goal_spec["goal_mask_path"]),
        "segment_type": str(goal_spec["segment_type"]),
    }
    for key, value in goal_spec.items():
        if str(key) == "occupied_voxel_keys":
            continue
        fixed_metadata[str(key)] = sanitize_mapping(value)
    session.current_goal_metadata = fixed_metadata
    session._fixed_goal_metadata = sanitize_mapping(fixed_metadata)


def build_session(
    args,
    name_file_mapping,
    task_spec: InteractionBenchmarkTaskSpec,
    goal_spec: Optional[Dict],
    goal_mode: str,
):
    if goal_mode == "auto":
        model_path = args.model_path or "hf:phython96/ROCKET-2-1x-22w"
        cfg_coef = float(1.5 if args.cfg_coef is None else args.cfg_coef)
        return CrossViewSession(
            model_path=model_path,
            name_file_mapping=name_file_mapping,
            goal_segment_type=task_spec.subtasks[0].interaction_type,
            cfg_coef=cfg_coef,
            cfg_policy_mode=args.cfg_policy_mode,
            cfg_base_ref_model_path=args.cfg_base_ref_model_path,
            obs_size=(224, 224),
            auto_goal_task_spec=task_spec,
        )
    assert goal_spec is not None
    model_path = args.model_path or goal_spec.get("model_uri") or "hf:phython96/ROCKET-2-1x-22w"
    cfg_coef = float(goal_spec.get("cfg_coef", 1.5) if args.cfg_coef is None else args.cfg_coef)
    obs_size = tuple(goal_spec.get("obs_size", [224, 224]))
    session = CrossViewSession(
        model_path=model_path,
        name_file_mapping=name_file_mapping,
        goal_image_path=goal_spec["goal_image_path"],
        goal_mask_path=goal_spec["goal_mask_path"],
        goal_segment_type=goal_spec["segment_type"],
        cfg_coef=cfg_coef,
        cfg_policy_mode=args.cfg_policy_mode,
        cfg_base_ref_model_path=args.cfg_base_ref_model_path,
        obs_size=obs_size,
    )
    _apply_fixed_goal_metadata(session, goal_spec)
    return session


def main():
    args = build_argparser().parse_args()
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if len(task_names) != 1:
        raise ValueError("interaction_crossview_rollout currently supports exactly one task via --tasks.")
    explicit_goal_spec = load_goal_spec(args.goal_spec) if args.goal_spec else None
    protocol = resolve_interaction_protocol(args.protocol)
    task_specs = resolve_interaction_task_specs(task_names, env_source=args.env_source)
    task_specs = apply_protocol_to_task_specs(task_specs, protocol_name=args.protocol)
    warmup_noop_steps = protocol.warmup_noop_steps if args.warmup_noop_steps is None else int(args.warmup_noop_steps)

    refresh_task_configs = bool(args.refresh_task_configs or Path(args.task_group_path).is_dir())
    file_list = prepare_task_configs(args.task_group, path=args.task_group_path, refresh=refresh_task_configs)
    name_file_mapping = {name: file for name, file in file_list.items()}

    goal_spec, resolved_goal_mode = resolve_goal_spec_for_task(args, task_specs[0], explicit_goal_spec)
    if args.auto_goal and resolved_goal_mode == "auto":
        task_key = task_specs[0].task_key or task_specs[0].task_config_name
        if task_key not in AUTO_GOAL_SUPPORTED_TASKS:
            supported = ", ".join(sorted(AUTO_GOAL_SUPPORTED_TASKS))
            raise NotImplementedError(f"--auto-goal is only implemented for: {supported}. Got {task_key}.")
    session = build_session(args, name_file_mapping, task_specs[0], goal_spec, resolved_goal_mode)

    out_dir = Path(args.out_dir)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "env_source": args.env_source,
                "task_group": args.task_group,
                "task_group_path": args.task_group_path,
                "protocol": asdict(protocol),
                "episodes_per_task": int(args.episodes_per_task),
                "base_seed": int(args.base_seed),
                "seed_step": int(args.seed_step),
                "sampling_base_seed": (int(args.sampling_base_seed) if args.sampling_base_seed is not None else None),
                "sampling_seed_step": resolve_sampling_seed_step(args),
                "warmup_noop_steps": int(warmup_noop_steps),
                "step_budget_override": int(args.step_budget_override),
                "tasks": [spec.task_key or spec.task_config_name for spec in task_specs],
                "goal_mode_requested": "auto" if args.auto_goal else "fixed",
                "goal_mode_resolved": resolved_goal_mode,
                "goal_spec": goal_spec,
                "cfg_coef": float(session.cfg_coef),
                "env_reward_scale": float(args.env_reward_scale),
                "cfg_policy_mode": str(args.cfg_policy_mode),
                "cfg_base_ref_model_path": str(args.cfg_base_ref_model_path or ""),
                "model_path": args.model_path or (goal_spec.get("model_uri") if goal_spec else "") or "hf:phython96/ROCKET-2-1x-22w",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    all_rows: List[Dict] = []
    total_episodes = int(sum(max(1, int(args.episodes_per_task)) for _ in task_specs))
    completed_episodes = 0
    run_started_at = time.perf_counter()
    try:
        for task_spec in task_specs:
            if task_spec.task_config_name not in name_file_mapping:
                raise KeyError(f"Task config not found: {task_spec.task_config_name}")
            for episode_idx in range(args.episodes_per_task):
                episode_started_at = time.perf_counter()
                core_episode_wall_time = 0.0
                goal_save_wall_time = 0.0
                debug_save_wall_time = 0.0
                video_write_wall_time = 0.0
                seed = int(args.base_seed + episode_idx * int(args.seed_step))
                episode_sampling_seed = compute_episode_sampling_seed(args, episode_idx)
                episode_tag = f"seed_{seed}_ep_{episode_idx:03d}"
                goal_spec, resolved_goal_mode = resolve_goal_spec_for_task(args, task_spec, explicit_goal_spec)
                print(
                    "[interaction-crossview][progress]",
                    json.dumps(
                        {
                            "event": "episode_start",
                            "episode_progress": f"{completed_episodes + 1}/{total_episodes}",
                            "task_config_name": task_spec.task_config_name,
                            "seed": int(seed),
                            "sampling_seed": (int(episode_sampling_seed) if episode_sampling_seed is not None else None),
                            "episode_index": int(episode_idx),
                            "goal_mode": resolved_goal_mode,
                            "skip_video": bool(args.skip_video),
                            "episode_retries": int(max(1, args.episode_retries)),
                            "step_budget_override": int(args.step_budget_override),
                        },
                        ensure_ascii=False,
                    ),
                )
                result = None
                last_exc = None
                attempt_count = 0
                sampling_seed_used = episode_sampling_seed
                for attempt in range(int(max(1, args.episode_retries))):
                    attempt_count = int(attempt + 1)
                    sampling_seed_used = None if episode_sampling_seed is None else int(episode_sampling_seed + attempt)
                    set_sampling_seed(sampling_seed_used)
                    try:
                        core_started_at = time.perf_counter()
                        result = run_crossview_episode(
                            task_spec=task_spec,
                            seed=seed,
                            session=session,
                            warmup_noop_steps=warmup_noop_steps,
                            stop_on_success=args.stop_on_success,
                            step_budget_override=(int(args.step_budget_override) if int(args.step_budget_override) > 0 else None),
                            env_reward_scale=float(args.env_reward_scale),
                        )
                        core_episode_wall_time = time.perf_counter() - core_started_at
                        break
                    except Exception as exc:
                        last_exc = exc
                        print(
                            f"[interaction-crossview][episode-error] task={task_spec.task_config_name} "
                            f"seed={seed} episode={episode_idx} attempt={attempt + 1} {type(exc).__name__}: {exc}"
                        )
                        if args.save_debug_assets:
                            failure_dir = run_dir / task_spec.task_config_name / episode_tag / f"attempt_{attempt + 1:02d}_failed"
                            failure_dir.mkdir(parents=True, exist_ok=True)
                            debug_assets = session.save_debug_assets(failure_dir)
                            (failure_dir / "failure.json").write_text(
                                json.dumps(
                                    {
                                        "task_config_name": task_spec.task_config_name,
                                        "seed": int(seed),
                                        "episode_index": int(episode_idx),
                                        "attempt": int(attempt + 1),
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc),
                                        "goal_metadata": session.current_goal_metadata,
                                        "debug_assets": debug_assets,
                                    },
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                encoding="utf-8",
                            )
                        try:
                            session.close()
                        except Exception:
                            pass
                        goal_spec, resolved_goal_mode = resolve_goal_spec_for_task(args, task_spec, explicit_goal_spec)
                        session = build_session(args, name_file_mapping, task_spec, goal_spec, resolved_goal_mode)
                if result is None:
                    raise last_exc or RuntimeError("Unknown cross-view episode failure")

                episode_dir = run_dir / task_spec.task_config_name / episode_tag
                episode_dir.mkdir(parents=True, exist_ok=True)
                debug_assets = None
                if args.save_debug_assets:
                    debug_save_started_at = time.perf_counter()
                    debug_assets = session.save_debug_assets(episode_dir)
                    debug_save_wall_time = time.perf_counter() - debug_save_started_at
                goal_save_started_at = time.perf_counter()
                goal_assets = session.save_goal_assets(episode_dir)
                goal_save_wall_time = time.perf_counter() - goal_save_started_at
                write_jsonl(episode_dir / "trajectory.jsonl", result["trajectory_rows"])
                if args.save_ppo_fragments and result.get("ppo_fragment") is not None:
                    torch.save(result["ppo_fragment"], episode_dir / "episode_fragment.pt")

                if not args.skip_video:
                    video_started_at = time.perf_counter()
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_base = f"{task_spec.task_config_name}_seed_{seed}_ep_{episode_idx:03d}_{timestamp}"
                    raw_video_path = episode_dir / f"{video_base}.mp4"
                    annotated_video_path = episode_dir / f"{video_base}_annotated.mp4"
                    final_frame_path = episode_dir / f"{video_base}_final_frame.png"
                    final_frame_annotated_path = episode_dir / f"{video_base}_final_frame_annotated.png"
                    write_video(raw_video_path, session.image_history)
                    write_annotated_video(
                        annotated_video_path,
                        session.image_history,
                        benchmark_name=task_spec.benchmark_name,
                        goal_image=session.goal_image_224,
                        goal_mask=session.goal_mask_224,
                        goal_segment_type=session.goal_segment_type,
                    )
                    if session.image_history:
                        final_frame = session.image_history[-1]
                        cv2.imwrite(str(final_frame_path), cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
                        final_annotated = annotate_frame(
                            final_frame,
                            benchmark_name=task_spec.benchmark_name,
                            goal_image=session.goal_image_224,
                            goal_mask=session.goal_mask_224,
                            goal_segment_type=session.goal_segment_type,
                            global_step=int(result["num_steps"]),
                        )
                        cv2.imwrite(str(final_frame_annotated_path), cv2.cvtColor(final_annotated, cv2.COLOR_RGB2BGR))
                    video_write_wall_time = time.perf_counter() - video_started_at

                episode_wall_time = time.perf_counter() - episode_started_at
                result_payload = {
                    **{k: v for k, v in result.items() if k not in {"trajectory_rows", "ppo_fragment"}},
                    "episode_index": int(episode_idx),
                    "attempt_count": int(attempt_count),
                    "episode_sampling_seed": (int(episode_sampling_seed) if episode_sampling_seed is not None else None),
                    "sampling_seed": (int(sampling_seed_used) if sampling_seed_used is not None else None),
                    "core_episode_wall_time_sec": round(float(core_episode_wall_time), 4),
                    "goal_save_wall_time_sec": round(float(goal_save_wall_time), 4),
                    "debug_save_wall_time_sec": round(float(debug_save_wall_time), 4),
                    "video_write_wall_time_sec": round(float(video_write_wall_time), 4),
                    "episode_wall_time_sec": round(float(episode_wall_time), 4),
                    "goal_assets": goal_assets,
                    "goal_metadata": session.current_goal_metadata,
                    "debug_assets": debug_assets,
                }
                (episode_dir / "result.json").write_text(
                    json.dumps(result_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                row = {k: v for k, v in result.items() if k not in {"trajectory_rows", "ppo_fragment"}}
                row["episode_index"] = int(episode_idx)
                row["attempt_count"] = int(attempt_count)
                row["episode_sampling_seed"] = (int(episode_sampling_seed) if episode_sampling_seed is not None else None)
                row["sampling_seed"] = (int(sampling_seed_used) if sampling_seed_used is not None else None)
                row["goal_mode"] = resolved_goal_mode
                row["goal_metadata"] = session.current_goal_metadata
                row["core_episode_wall_time_sec"] = round(float(core_episode_wall_time), 4)
                row["goal_save_wall_time_sec"] = round(float(goal_save_wall_time), 4)
                row["debug_save_wall_time_sec"] = round(float(debug_save_wall_time), 4)
                row["video_write_wall_time_sec"] = round(float(video_write_wall_time), 4)
                row["episode_wall_time_sec"] = round(float(episode_wall_time), 4)
                print("[interaction-crossview]", json.dumps(row, ensure_ascii=False))
                completed_episodes += 1
                print(
                    "[interaction-crossview][progress]",
                    json.dumps(
                        {
                            "event": "episode_done",
                            "episode_progress": f"{completed_episodes}/{total_episodes}",
                            "task_config_name": task_spec.task_config_name,
                            "seed": int(seed),
                            "sampling_seed": (int(sampling_seed_used) if sampling_seed_used is not None else None),
                            "episode_index": int(episode_idx),
                            "attempt_count": int(attempt_count),
                            "num_steps": int(result["num_steps"]),
                            "stop_reason": result["stop_reason"],
                            "episode_wall_time_sec": round(float(episode_wall_time), 4),
                            "episode_wall_time_human": format_seconds(episode_wall_time),
                            "video_write_wall_time_sec": round(float(video_write_wall_time), 4),
                            "video_write_wall_time_human": format_seconds(video_write_wall_time),
                            "remaining_episodes": int(total_episodes - completed_episodes),
                        },
                        ensure_ascii=False,
                    ),
                )
                all_rows.append(row)

        summary_rows = summarize(all_rows)
        (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(run_dir / "summary.csv", summary_rows)
        write_csv(run_dir / "episodes.csv", all_rows)
        total_run_wall_time = time.perf_counter() - run_started_at
        print(
            "[interaction-crossview][run-summary]",
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "episodes": int(len(all_rows)),
                    "total_run_wall_time_sec": round(float(total_run_wall_time), 4),
                    "total_run_wall_time_human": format_seconds(total_run_wall_time),
                    "mean_episode_wall_time_sec": round(
                        sum(float(row.get("episode_wall_time_sec", 0.0) or 0.0) for row in all_rows) / max(1, len(all_rows)),
                        4,
                    ),
                    "mean_video_write_wall_time_sec": round(
                        sum(float(row.get("video_write_wall_time_sec", 0.0) or 0.0) for row in all_rows) / max(1, len(all_rows)),
                        4,
                    ),
                },
                ensure_ascii=False,
            ),
        )
        for item in summary_rows:
            print("[interaction-crossview] summary")
            print(json.dumps(item, ensure_ascii=False))
    finally:
        session.close()


if __name__ == "__main__":
    main()
