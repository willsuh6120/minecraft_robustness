import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import cv2

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_protocol import (
    apply_protocol_to_task_specs,
    resolve_interaction_protocol,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    InteractionBenchmarkTaskSpec,
    resolve_interaction_task_specs,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_failure_analysis import (
    analyze_interaction_episode,
    summarize_failure_buckets,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_success_checker import (
    InteractionBenchmarkSuccessTracker,
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


def _event_points(event: Dict) -> List[Tuple[int, int]]:
    return [tuple(map(int, point)) for point in event.get("points", [])]


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
        points = _event_points(event)
        if points:
            lines.append(f"point={points[0]}")
        else:
            lines.append("point=None")
        for point_idx, (x, y) in enumerate(points):
            cv2.circle(annotated, (x, y), 8, (255, 64, 64), -1)
            cv2.circle(annotated, (x, y), 14, (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"p{point_idx}",
                (x + 10, max(18, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
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


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def capture_step_snapshot(
    session: Session,
    global_step: int,
    subtask_index: int,
    target_text: str,
    interaction_type: str,
):
    info = session.info
    player_pos = sanitize_mapping(info.get("player_pos") or info.get("location_stats") or {})
    return {
        "global_step": global_step,
        "subtask_index": subtask_index,
        "target_text": target_text,
        "interaction_type": interaction_type,
        "reward": float(session.last_reward),
        "terminated": bool(session.last_terminated),
        "truncated": bool(session.last_truncated),
        "last_action_summary": session.last_action_summary,
        "player_pos": player_pos,
        "inventory_totals": inventory_totals(info),
        "use_item": sanitize_mapping(info.get("use_item", {})),
        "pickup": sanitize_mapping(info.get("pickup", {})),
        "craft_item": sanitize_mapping(info.get("craft_item", {})),
        "mine_block": sanitize_mapping(info.get("mine_block", {})),
        "kill_entity": sanitize_mapping(info.get("kill_entity", {})),
        "equipped_items": sanitize_mapping(info.get("equipped_items", {})),
    }


def reprompt_and_segment(
    session: Session,
    pointer: Pointer,
    subtask,
    reprompt_idx: int,
):
    session.segment_type = subtask.interaction_type
    prompt_text = subtask.prompt_text or subtask.target_text
    points = pointer.gen_point(image=session.current_image.copy(), prompt=prompt_text)
    session.clear_points()
    for x, y in points:
        session.points.append([int(x), int(y)])
        session.points_label.append(1)
    if points:
        session.segment()
    return {
        "reprompt_index": reprompt_idx,
        "step": session.num_steps,
        "target_text": subtask.target_text,
        "prompt_text": prompt_text,
        "interaction_type": subtask.interaction_type,
        "points": [(int(x), int(y)) for x, y in points],
        "molmo_text": pointer.molmo_result,
        "segment_area": int(getattr(session, "last_segment_area", 0)),
        "segment_raw_area": int(getattr(session, "last_segment_raw_area", 0)),
        "segment_fallback_area": int(getattr(session, "last_segment_fallback_area", 0)),
        "segment_used_fallback": bool(getattr(session, "last_segment_used_fallback", False)),
        "segment_point_count": int(getattr(session, "last_segment_point_count", 0)),
    }


def run_interaction_episode(
    task_spec: InteractionBenchmarkTaskSpec,
    seed: int,
    session: Session,
    pointer: Pointer,
    stop_on_auto_success: bool,
    warmup_noop_steps: int,
):
    session.reset(task_spec.task_config_name, seed, warmup_noop_steps=warmup_noop_steps)
    tracker = InteractionBenchmarkSuccessTracker(task_spec, session.info)
    reprompt_events = []
    reprompt_debug_frames = []
    trajectory_rows = []
    initial_inventory = inventory_totals(session.info)
    total_step_budget = sum(subtask.step_budget for subtask in task_spec.subtasks)

    auto_success = None
    auto_supported = False
    auto_metric = "manual_review_required"
    auto_progress = {}
    auto_reason = ""
    stop_reason = ""
    pointer_calls = 0

    for subtask_index, subtask in enumerate(task_spec.subtasks):
        last_event = reprompt_and_segment(
            session=session,
            pointer=pointer,
            subtask=subtask,
            reprompt_idx=len(reprompt_events),
        )
        reprompt_events.append(last_event)
        reprompt_debug_frames.append({"event": copy_for_json(last_event), "frame": session.current_image.copy()})
        pointer_calls += 1

        for local_step in range(subtask.step_budget):
            if local_step > 0 and local_step % subtask.reprompt_interval == 0:
                last_event = reprompt_and_segment(
                    session=session,
                    pointer=pointer,
                    subtask=subtask,
                    reprompt_idx=len(reprompt_events),
                )
                reprompt_events.append(last_event)
                reprompt_debug_frames.append({"event": copy_for_json(last_event), "frame": session.current_image.copy()})
                pointer_calls += 1

            session.step()
            success_result = tracker.update(session.info)
            auto_supported = success_result.supported
            auto_success = success_result.success
            auto_metric = success_result.metric
            auto_progress = success_result.progress
            auto_reason = success_result.reason

            trajectory_rows.append(
                capture_step_snapshot(
                    session=session,
                    global_step=session.num_steps,
                    subtask_index=subtask_index,
                    target_text=subtask.target_text,
                    interaction_type=subtask.interaction_type,
                )
            )

            if session.last_terminated:
                stop_reason = "terminated"
                break
            if session.last_truncated:
                stop_reason = "truncated"
                break
            if stop_on_auto_success and success_result.supported and success_result.success:
                stop_reason = "auto_success"
                break

        if stop_reason:
            break

    if not stop_reason:
        stop_reason = "max_steps"

    return {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "category": task_spec.category,
        "seed": seed,
        "warmup_noop_steps": int(warmup_noop_steps),
        "active_step_budget": int(total_step_budget),
        "num_steps": session.num_steps,
        "pointer_calls": pointer_calls,
        "auto_eval_supported": auto_supported,
        "auto_success": auto_success,
        "auto_metric": auto_metric,
        "auto_progress": auto_progress,
        "auto_reason": auto_reason,
        "stop_reason": stop_reason,
        "initial_inventory": initial_inventory,
        "final_inventory": inventory_totals(session.info),
        "last_action_summary": session.last_action_summary,
        "reprompt_events": reprompt_events,
        "reprompt_debug_frames": reprompt_debug_frames,
        "trajectory_rows": trajectory_rows,
    }


def copy_for_json(data):
    return json.loads(json.dumps(sanitize_mapping(data), ensure_ascii=False))


def summarize(rows: List[Dict]):
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark_name"], []).append(row)

    summary = []
    for benchmark_name, benchmark_rows in grouped.items():
        auto_supported_rows = [row for row in benchmark_rows if row["auto_eval_supported"]]
        auto_success_values = [1.0 if row["auto_success"] else 0.0 for row in auto_supported_rows if row["auto_success"] is not None]
        summary.append(
            {
                "benchmark_name": benchmark_name,
                "episodes": len(benchmark_rows),
                "auto_eval_supported_episodes": len(auto_supported_rows),
                "auto_success_rate": round(sum(auto_success_values) / len(auto_success_values), 4) if auto_success_values else None,
                "mean_steps": round(sum(int(row["num_steps"]) for row in benchmark_rows) / len(benchmark_rows), 2),
                "mean_pointer_calls": round(sum(int(row["pointer_calls"]) for row in benchmark_rows) / len(benchmark_rows), 2),
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
    parser.add_argument("--stop-on-auto-success", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/interaction_benchmark")

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


def build_error_result(task_spec: InteractionBenchmarkTaskSpec, seed: int, warmup_noop_steps: int, exc: Exception, attempt: int):
    return {
        "benchmark_name": task_spec.benchmark_name,
        "task_config_name": task_spec.task_config_name,
        "category": task_spec.category,
        "seed": seed,
        "warmup_noop_steps": int(warmup_noop_steps),
        "active_step_budget": int(sum(subtask.step_budget for subtask in task_spec.subtasks)),
        "num_steps": 0,
        "pointer_calls": 0,
        "auto_eval_supported": False,
        "auto_success": None,
        "auto_metric": "runner_exception",
        "auto_progress": {},
        "auto_reason": f"{type(exc).__name__}: {exc}",
        "stop_reason": "exception",
        "initial_inventory": {},
        "final_inventory": {},
        "last_action_summary": "none",
        "reprompt_events": [],
        "reprompt_debug_frames": [],
        "trajectory_rows": [],
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "attempt": int(attempt),
    }


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

    all_rows = []
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
                        result = run_interaction_episode(
                            task_spec=task_spec,
                            seed=seed,
                            session=session,
                            pointer=pointer,
                            stop_on_auto_success=args.stop_on_auto_success,
                            warmup_noop_steps=warmup_noop_steps,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        print(
                            f"[interaction-benchmark][episode-error] "
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
                        exc=last_exc if last_exc is not None else RuntimeError("unknown episode failure"),
                        attempt=int(max(1, args.episode_retries)),
                    )
                failure_analysis = analyze_interaction_episode(result)
                result["failure_analysis"] = failure_analysis
                episode_dir = run_dir / task_spec.task_config_name / f"seed_{seed}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "result.json").write_text(
                    json.dumps(
                        {k: v for k, v in result.items() if k not in {"trajectory_rows", "reprompt_debug_frames"}},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                write_jsonl(episode_dir / "trajectory.jsonl", result["trajectory_rows"])
                file_stem = f"{task_spec.task_config_name}_seed_{seed}_{run_dir.name}"
                image_history = getattr(session, "image_history", [])
                if image_history:
                    final_event = event_for_step(result["reprompt_events"], len(session.image_history))
                    annotated_final = annotate_frame(
                        frame=image_history[-1],
                        event=final_event,
                        benchmark_name=task_spec.benchmark_name,
                        global_step=len(image_history),
                    )
                    cv2.imwrite(
                        str(episode_dir / f"{file_stem}_final_frame.png"),
                        cv2.cvtColor(image_history[-1], cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(episode_dir / f"{file_stem}_final_frame_annotated.png"),
                        cv2.cvtColor(annotated_final, cv2.COLOR_RGB2BGR),
                    )
                if not args.skip_video:
                    write_video(episode_dir / f"{file_stem}.mp4", image_history)
                    write_annotated_video(
                        episode_dir / f"{file_stem}_annotated.mp4",
                        image_history,
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
                    "pointer_calls": result["pointer_calls"],
                    "auto_eval_supported": result["auto_eval_supported"],
                    "auto_success": result["auto_success"],
                    "auto_metric": result["auto_metric"],
                    "primary_failure_bucket": failure_analysis["primary_bucket"],
                    "failure_tags": ",".join(failure_analysis["tags"]),
                    "stop_reason": result["stop_reason"],
                }
                all_rows.append(row)
                print("[interaction-benchmark]", json.dumps(row, ensure_ascii=False))
                session.image_history = []
    finally:
        session.close()

    summary_rows = summarize(all_rows)
    failure_summary_rows = summarize_failure_buckets(all_rows)
    write_jsonl(run_dir / "episodes.jsonl", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "failure_summary.csv", failure_summary_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "failure_summary.json").write_text(
        json.dumps(failure_summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[interaction-benchmark] summary")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))
    print("[interaction-benchmark] failure-summary")
    for row in failure_summary_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
