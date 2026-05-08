import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import torch

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_protocol import (
    apply_protocol_to_task_specs,
    resolve_interaction_protocol,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    InteractionBenchmarkTaskSpec,
    ScriptedSubtask,
    resolve_interaction_task_specs,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_posttrain_rewards import (
    InteractionPostTrainRewardTracker,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_verifier import (
    PointVerificationResult,
    verify_subtask_point,
)
from minestudio.tutorials.inference.evaluate_rocket.success_checker import inventory_totals
from minestudio.tutorials.inference.evaluate_rocket.utils import Pointer, Session


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
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _draw_text_block(frame, lines: List[str]):
    overlay = frame.copy()
    line_height = 18
    width = max(220, min(frame.shape[1] - 20, 12 * max((len(line) for line in lines), default=10)))
    height = max(28, 14 + line_height * len(lines))
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for idx, line in enumerate(lines):
        y = 28 + idx * line_height
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def annotate_frame(frame, event: Optional[Dict], benchmark_name: str, global_step: Optional[int] = None):
    annotated = frame.copy()
    lines = [benchmark_name]
    if global_step is not None:
        lines.append(f"step={global_step}")
    if event is not None:
        lines.append(f"target={event.get('target_text', 'n/a')}")
        lines.append(f"prompt={event.get('prompt_text', 'n/a')}")
        lines.append(f"interaction={event.get('interaction_type', 'n/a')}")
        lines.append(f"reprompt@step={event.get('step', 'n/a')}")
        accepted_point = event.get("accepted_point")
        if accepted_point is not None:
            x, y = int(accepted_point[0]), int(accepted_point[1])
            lines.append(f"point=({x}, {y})")
            cv2.circle(annotated, (x, y), 8, (255, 64, 64), -1)
            cv2.circle(annotated, (x, y), 14, (255, 255, 255), 2)
        else:
            lines.append("point=None")
        lines.append(f"pointer_valid={event.get('pointer_valid', False)}")
    else:
        lines.append("reprompt=None")
    return _draw_text_block(annotated, lines)


def event_for_step(reprompt_events: List[Dict], global_step: int) -> Optional[Dict]:
    active_event = None
    for event in reprompt_events:
        if int(event.get("step", 0)) < int(global_step):
            active_event = event
        else:
            break
    return active_event


def write_annotated_video(path: Path, frames: List, reprompt_events: List[Dict], benchmark_name: str):
    annotated_frames = []
    for frame_idx, frame in enumerate(frames):
        global_step = frame_idx + 1
        annotated_frames.append(
            annotate_frame(
                frame=frame,
                event=event_for_step(reprompt_events, global_step),
                benchmark_name=benchmark_name,
                global_step=global_step,
            )
        )
    write_video(path, annotated_frames)


def write_reprompt_debug_frames(
    episode_dir: Path,
    reprompt_debug_frames: List[Dict],
    benchmark_name: str,
):
    for item in reprompt_debug_frames:
        frame = annotate_frame(
            frame=item["frame"],
            event=item["event"],
            benchmark_name=benchmark_name,
            global_step=int(item["event"].get("step", 0)),
        )
        filename = (
            f"reprompt_{int(item['event'].get('reprompt_index', 0)):03d}"
            f"_step_{int(item['event'].get('step', 0)):03d}.png"
        )
        cv2.imwrite(str(episode_dir / filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def copy_for_json(data):
    return json.loads(json.dumps(sanitize_mapping(data), ensure_ascii=False))


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
    bootstrap_value: float = 0.0,
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
        "bootstrap_value": float(bootstrap_value),
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


def compute_fragment_bootstrap(session: Session, stop_reason: str):
    if bool(session.last_terminated):
        return 0.0, False, "terminated"
    if bool(session.last_truncated) or str(stop_reason) == "max_steps":
        value = session.estimate_current_policy_value()
        if value is not None:
            reason = "truncated" if bool(session.last_truncated) else "max_steps"
            return float(value), True, reason
    return 0.0, False, str(stop_reason)


def build_error_result(
    task_spec: InteractionBenchmarkTaskSpec,
    seed: int,
    warmup_noop_steps: int,
    exc: Exception,
    attempt: int,
):
    total_step_budget = sum(subtask.step_budget for subtask in task_spec.subtasks)
    return {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "category": task_spec.category,
        "seed": int(seed),
        "warmup_noop_steps": int(warmup_noop_steps),
        "active_step_budget": int(total_step_budget),
        "num_steps": 0,
        "episode_reward": 0.0,
        "pointer_calls": 0,
        "pointer_failures": 0,
        "reward_supported": False,
        "trainable": False,
        "stop_reason": "exception",
        "last_action_summary": "none",
        "reprompt_events": [],
        "reprompt_debug_frames": [],
        "trajectory_rows": [],
        "ppo_fragment": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "attempt": int(attempt),
    }


def capture_fragment_step(
    session: Session,
    global_step: int,
    subtask_index: int,
    subtask: ScriptedSubtask,
    rl_reward: float,
):
    if session.last_model_input is None or session.last_policy_action is None:
        return None
    if session.last_policy_logprob is None or session.last_policy_value is None:
        return None

    result = {
        "global_step": int(global_step),
        "subtask_index": torch.tensor(int(subtask_index), dtype=torch.long),
        "interaction_type": subtask.interaction_type,
        "prompt_text": subtask.prompt_text or subtask.target_text,
        "image": torch.from_numpy(session.last_model_input["image"]).to(dtype=torch.uint8),
        "obj_mask": session.last_model_input["obj_mask"].to(dtype=torch.uint8),
        "obj_id": torch.tensor(int(session.last_model_input["obj_id"]), dtype=torch.long),
        "action": _clone_tensor_tree(session.last_policy_action),
        "old_logprob": torch.tensor(float(session.last_policy_logprob), dtype=torch.float32),
        "old_value": torch.tensor(float(session.last_policy_value), dtype=torch.float32),
        "reward": torch.tensor(float(rl_reward), dtype=torch.float32),
        "done": torch.tensor(bool(session.last_terminated or session.last_truncated), dtype=torch.bool),
        "env_done": torch.tensor(bool(session.last_terminated or session.last_truncated), dtype=torch.bool),
        "terminated": torch.tensor(bool(session.last_terminated), dtype=torch.bool),
        "truncated": torch.tensor(bool(session.last_truncated), dtype=torch.bool),
        "first": torch.tensor(bool(global_step == 1), dtype=torch.bool),
        "segment_area": torch.tensor(int(session.last_segment_area), dtype=torch.long),
    }
    if "cross_view_image" in session.last_model_input:
        result["cross_view_image"] = session.last_model_input["cross_view_image"].to(dtype=torch.uint8)
    if "cross_view_obj_mask" in session.last_model_input:
        result["cross_view_obj_mask"] = session.last_model_input["cross_view_obj_mask"].to(dtype=torch.uint8)
    if "cross_view_obj_id" in session.last_model_input:
        result["cross_view_obj_id"] = torch.tensor(int(session.last_model_input["cross_view_obj_id"]), dtype=torch.long)
    if "env_prev_action" in session.last_model_input:
        result["env_prev_action"] = _clone_tensor_tree(session.last_model_input["env_prev_action"])
    return result


def capture_step_snapshot(
    session: Session,
    global_step: int,
    subtask_index: int,
    subtask: ScriptedSubtask,
    rl_reward: float,
    reward_reason: str,
    reward_metrics: Dict[str, float],
):
    info = session.info
    player_pos = sanitize_mapping(info.get("player_pos") or info.get("location_stats") or {})
    return {
        "global_step": global_step,
        "subtask_index": subtask_index,
        "target_text": subtask.target_text,
        "prompt_text": subtask.prompt_text or subtask.target_text,
        "interaction_type": subtask.interaction_type,
        "env_reward": float(session.last_reward),
        "rl_reward": float(rl_reward),
        "reward_reason": reward_reason,
        "reward_metrics": reward_metrics,
        "terminated": bool(session.last_terminated),
        "truncated": bool(session.last_truncated),
        "last_action_summary": session.last_action_summary,
        "policy_logprob": session.last_policy_logprob,
        "policy_value": session.last_policy_value,
        "segment_area": int(session.last_segment_area),
        "player_pos": player_pos,
        "inventory_totals": inventory_totals(info),
        "use_item": sanitize_mapping(info.get("use_item", {})),
        "pickup": sanitize_mapping(info.get("pickup", {})),
        "craft_item": sanitize_mapping(info.get("craft_item", {})),
        "mine_block": sanitize_mapping(info.get("mine_block", {})),
        "kill_entity": sanitize_mapping(info.get("kill_entity", {})),
        "equipped_items": sanitize_mapping(info.get("equipped_items", {})),
    }


def _verification_payload(result: PointVerificationResult) -> Dict:
    return {
        "valid": bool(result.valid),
        "reason": result.reason,
        "normalized_point": result.normalized_point,
    }


def request_verified_segment(
    session: Session,
    pointer: Pointer,
    subtask: ScriptedSubtask,
) -> Dict:
    accepted_point: Optional[Tuple[int, int]] = None
    accepted_verification: Optional[PointVerificationResult] = None
    attempts: List[Dict] = []
    prompt_text = subtask.prompt_text or subtask.target_text

    for attempt_idx in range(subtask.max_point_retries + 1):
        points = pointer.gen_point(image=session.current_image.copy(), prompt=prompt_text)
        point_entries = []
        for point in points:
            point_tuple = (int(point[0]), int(point[1]))
            verification = verify_subtask_point(subtask, point_tuple, session.current_image.shape)
            point_entries.append(
                {
                    "point": [point_tuple[0], point_tuple[1]],
                    "verification": _verification_payload(verification),
                }
            )
            if accepted_point is None and verification.valid:
                accepted_point = point_tuple
                accepted_verification = verification
        attempts.append(
            {
                "attempt_index": attempt_idx,
                "prompt_text": prompt_text,
                "points": point_entries,
                "molmo_text": pointer.molmo_result,
            }
        )
        if accepted_point is not None:
            break

    session.segment_type = subtask.interaction_type
    session.clear_points()
    if accepted_point is not None:
        session.points.append([accepted_point[0], accepted_point[1]])
        session.points_label.append(1)
        session.segment()

    return {
        "target_text": subtask.target_text,
        "prompt_text": prompt_text,
        "interaction_type": subtask.interaction_type,
        "accepted_point": list(accepted_point) if accepted_point is not None else None,
        "accepted_verification": _verification_payload(accepted_verification) if accepted_verification is not None else None,
        "attempts": attempts,
        "pointer_valid": accepted_point is not None,
        "pointer_calls": len(attempts),
    }


def run_posttrain_episode(
    task_spec: InteractionBenchmarkTaskSpec,
    seed: int,
    session: Session,
    pointer: Pointer,
    warmup_noop_steps: int,
    stop_on_success: bool,
    refresh_on_tracker_loss: bool,
    min_tracker_area: int,
):
    session.reset(task_spec.task_config_name, seed, warmup_noop_steps=warmup_noop_steps)
    reward_tracker = InteractionPostTrainRewardTracker(task_spec, session.info)
    reprompt_events: List[Dict] = []
    reprompt_debug_frames: List[Dict] = []
    trajectory_rows: List[Dict] = []
    fragment_steps: List[Dict] = []
    pointer_failures = 0
    pointer_calls = 0
    episode_reward = 0.0
    trainable = True
    stop_reason = ""
    total_step_budget = sum(subtask.step_budget for subtask in task_spec.subtasks)
    reward_supported = reward_tracker.success_result(session.info).supported

    for subtask_index, subtask in enumerate(task_spec.subtasks):
        event = request_verified_segment(session=session, pointer=pointer, subtask=subtask)
        event_with_meta = {"subtask_index": subtask_index, "reprompt_index": len(reprompt_events), "step": session.num_steps, **event}
        reprompt_events.append(event_with_meta)
        reprompt_debug_frames.append({"event": copy_for_json(event_with_meta), "frame": session.current_image.copy()})
        pointer_calls += int(event["pointer_calls"])
        if not event["pointer_valid"]:
            pointer_failures += 1
            trainable = False
            stop_reason = "pointer_fail_initial"
            break

        for local_step in range(subtask.step_budget):
            need_reprompt = False
            if local_step > 0 and local_step % subtask.reprompt_interval == 0:
                need_reprompt = True
            if refresh_on_tracker_loss and session.able_to_track and session.last_segment_area < int(min_tracker_area):
                need_reprompt = True

            if need_reprompt:
                event = request_verified_segment(session=session, pointer=pointer, subtask=subtask)
                event_with_meta = {"subtask_index": subtask_index, "reprompt_index": len(reprompt_events), "step": session.num_steps, **event}
                reprompt_events.append(event_with_meta)
                reprompt_debug_frames.append({"event": copy_for_json(event_with_meta), "frame": session.current_image.copy()})
                pointer_calls += int(event["pointer_calls"])
                if not event["pointer_valid"]:
                    pointer_failures += 1
                    trainable = False
                    stop_reason = "pointer_fail_reprompt"
                    break

            session.step()
            reward_event = reward_tracker.evaluate(session.info, env_reward=session.last_reward)
            episode_reward += float(reward_event.reward)
            trajectory_rows.append(
                capture_step_snapshot(
                    session=session,
                    global_step=session.num_steps,
                    subtask_index=subtask_index,
                    subtask=subtask,
                    rl_reward=reward_event.reward,
                    reward_reason=reward_event.reason,
                    reward_metrics=reward_event.metrics,
                )
            )
            fragment_step = capture_fragment_step(
                session=session,
                global_step=session.num_steps,
                subtask_index=subtask_index,
                subtask=subtask,
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

        if stop_reason:
            break

    if not stop_reason:
        stop_reason = "max_steps"

    bootstrap_value, bootstrap_valid, bootstrap_reason = compute_fragment_bootstrap(
        session=session,
        stop_reason=stop_reason,
    )

    return {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "category": task_spec.category,
        "seed": seed,
        "warmup_noop_steps": int(warmup_noop_steps),
        "active_step_budget": int(total_step_budget),
        "num_steps": session.num_steps,
        "episode_reward": round(float(episode_reward), 6),
        "pointer_calls": pointer_calls,
        "pointer_failures": pointer_failures,
        "reward_supported": bool(reward_supported),
        "trainable": bool(trainable and reward_supported),
        "stop_reason": stop_reason,
        "last_action_summary": session.last_action_summary,
        "reprompt_events": reprompt_events,
        "reprompt_debug_frames": reprompt_debug_frames,
        "trajectory_rows": trajectory_rows,
        "ppo_fragment": build_episode_fragment(
            task_spec=task_spec,
            seed=seed,
            fragment_steps=fragment_steps,
            trainable=bool(trainable and reward_supported),
            stop_reason=stop_reason,
            bootstrap_value=bootstrap_value,
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
        summary.append(
            {
                "benchmark_name": benchmark_name,
                "episodes": len(benchmark_rows),
                "reward_supported_episodes": sum(1 for row in benchmark_rows if row["reward_supported"]),
                "trainable_episodes": sum(1 for row in benchmark_rows if row["trainable"]),
                "mean_reward": round(sum(float(row["episode_reward"]) for row in benchmark_rows) / len(benchmark_rows), 4),
                "mean_steps": round(sum(int(row["num_steps"]) for row in benchmark_rows) / len(benchmark_rows), 2),
                "mean_pointer_calls": round(sum(int(row["pointer_calls"]) for row in benchmark_rows) / len(benchmark_rows), 2),
                "pointer_fail_rate": round(sum(1 for row in benchmark_rows if row["pointer_failures"] > 0) / len(benchmark_rows), 4),
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
    parser.add_argument("--task-group", type=str, default="rocket_interaction")
    parser.add_argument(
        "--task-group-path",
        type=str,
        default="/home/gyulab/envgen2/Minestudio/minestudio/benchmark/task_configs/rocket_interaction",
    )
    parser.add_argument(
        "--env-source",
        type=str,
        default="local",
        choices=["local", "rocket2_official"],
        help="Which environment config naming scheme to use for task lookup.",
    )
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--warmup-noop-steps", type=int, default=None)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--refresh-on-tracker-loss", action="store_true")
    parser.add_argument("--min-tracker-area", type=int, default=64)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--save-ppo-fragments", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/interaction_posttrain")

    parser.add_argument("--model-path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--sam-path", type=str, required=True)
    parser.add_argument("--molmo-id", type=str, default="allenai/MolmoE-1B-0924")
    parser.add_argument("--molmo-url", type=str, default="http://127.0.0.1:9163")
    parser.add_argument("--molmo-api-key", type=str, default="EMPTY")
    parser.add_argument("--molmo-managed-env", type=str, default="molmo-pointing")
    parser.add_argument("--molmo-managed-host", type=str, default="127.0.0.1")
    parser.add_argument("--molmo-managed-port", type=int, default=0)
    parser.add_argument("--molmo-loader", type=str, default="manual")
    parser.add_argument("--molmo-torch-dtype", type=str, default="float16")
    parser.add_argument("--molmo-autocast-dtype", type=str, default="float16")
    parser.add_argument("--refresh-task-configs", action="store_true")
    parser.add_argument("--episode-retries", type=int, default=2)
    return parser


def build_session(args, name_file_mapping):
    return Session(
        model_path=args.model_path,
        sam_path=args.sam_path,
        name_file_mapping=name_file_mapping,
    )


def main():
    args = build_argparser().parse_args()
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()] or None
    protocol = resolve_interaction_protocol(args.protocol)
    task_specs = resolve_interaction_task_specs(task_names, env_source=args.env_source)
    task_specs = apply_protocol_to_task_specs(task_specs, protocol_name=args.protocol)
    warmup_noop_steps = protocol.warmup_noop_steps if args.warmup_noop_steps is None else int(args.warmup_noop_steps)

    refresh_task_configs = bool(args.refresh_task_configs or Path(args.task_group_path).is_dir())
    file_list = prepare_task_configs(args.task_group, path=args.task_group_path, refresh=refresh_task_configs)
    name_file_mapping = {name: file for name, file in file_list.items()}

    session = build_session(args, name_file_mapping)
    pointer = Pointer(
        model_id=args.molmo_id,
        model_url=args.molmo_url,
        api_key=args.molmo_api_key,
        managed_env_name=args.molmo_managed_env,
        managed_host=args.molmo_managed_host,
        managed_port=args.molmo_managed_port,
        molmo_loader=args.molmo_loader,
        molmo_torch_dtype=args.molmo_torch_dtype,
        molmo_autocast_dtype=args.molmo_autocast_dtype,
    )

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
                "warmup_noop_steps": int(warmup_noop_steps),
                "tasks": [spec.task_key or spec.task_config_name for spec in task_specs],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    all_rows: List[Dict] = []
    try:
        for task_spec in task_specs:
            if task_spec.task_config_name not in name_file_mapping:
                raise KeyError(f"Task config not found: {task_spec.task_config_name}")
            for episode_idx in range(args.episodes_per_task):
                seed = int(args.base_seed + episode_idx)
                result = None
                last_exc = None
                for attempt in range(int(max(1, args.episode_retries))):
                    try:
                        result = run_posttrain_episode(
                            task_spec=task_spec,
                            seed=seed,
                            session=session,
                            pointer=pointer,
                            warmup_noop_steps=warmup_noop_steps,
                            stop_on_success=args.stop_on_success,
                            refresh_on_tracker_loss=args.refresh_on_tracker_loss,
                            min_tracker_area=args.min_tracker_area,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        print(
                            f"[interaction-posttrain][episode-error] "
                            f"task={task_spec.task_config_name} seed={seed} attempt={attempt + 1} "
                            f"{type(exc).__name__}: {exc}"
                        )
                        try:
                            session.close()
                        except Exception:
                            pass
                        session = build_session(args, name_file_mapping)
                if result is None:
                    result = build_error_result(
                        task_spec=task_spec,
                        seed=seed,
                        warmup_noop_steps=warmup_noop_steps,
                        exc=last_exc or RuntimeError("Unknown episode failure"),
                        attempt=int(max(1, args.episode_retries)),
                    )
                episode_dir = run_dir / task_spec.task_config_name / f"seed_{seed}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "result.json").write_text(
                    json.dumps(
                        {k: v for k, v in result.items() if k not in {"trajectory_rows", "reprompt_debug_frames", "ppo_fragment"}},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                write_jsonl(episode_dir / "trajectory.jsonl", result["trajectory_rows"])
                if args.save_ppo_fragments and result.get("ppo_fragment") is not None:
                    torch.save(result["ppo_fragment"], episode_dir / "episode_fragment.pt")
                file_stem = f"{task_spec.task_config_name}_seed_{seed}_{run_dir.name}"
                if session.image_history:
                    final_event = event_for_step(result["reprompt_events"], len(session.image_history))
                    annotated_final = annotate_frame(
                        frame=session.image_history[-1],
                        event=final_event,
                        benchmark_name=task_spec.benchmark_name,
                        global_step=len(session.image_history),
                    )
                    cv2.imwrite(
                        str(episode_dir / f"{file_stem}_final_frame.png"),
                        cv2.cvtColor(session.image_history[-1], cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(episode_dir / f"{file_stem}_final_frame_annotated.png"),
                        cv2.cvtColor(annotated_final, cv2.COLOR_RGB2BGR),
                    )
                if not args.skip_video:
                    write_video(episode_dir / f"{file_stem}.mp4", session.image_history)
                    write_annotated_video(
                        episode_dir / f"{file_stem}_annotated.mp4",
                        session.image_history,
                        result["reprompt_events"],
                        task_spec.benchmark_name,
                    )
                write_reprompt_debug_frames(
                    episode_dir=episode_dir,
                    reprompt_debug_frames=result["reprompt_debug_frames"],
                    benchmark_name=task_spec.benchmark_name,
                )

                row = {
                    "benchmark_name": result["benchmark_name"],
                    "task_config_name": result["task_config_name"],
                    "category": result["category"],
                    "seed": result["seed"],
                    "warmup_noop_steps": result["warmup_noop_steps"],
                    "active_step_budget": result["active_step_budget"],
                    "num_steps": result["num_steps"],
                    "episode_reward": result["episode_reward"],
                    "pointer_calls": result["pointer_calls"],
                    "pointer_failures": result["pointer_failures"],
                    "reward_supported": result["reward_supported"],
                    "trainable": result["trainable"],
                    "stop_reason": result["stop_reason"],
                }
                all_rows.append(row)
                print("[interaction-posttrain]", json.dumps(row, ensure_ascii=False))
                session.image_history = []
    finally:
        pointer.close()
        session.close()

    summary_rows = summarize(all_rows)
    write_jsonl(run_dir / "episodes.jsonl", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[interaction-posttrain] summary")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
