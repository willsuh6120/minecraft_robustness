import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.benchmark_spec import (
    EpisodeSpec,
    build_episode_specs,
    episode_spec_to_dict,
    resolve_task_specs,
)
from minestudio.tutorials.inference.evaluate_rocket.success_checker import TaskSuccessTracker
from minestudio.tutorials.inference.evaluate_rocket.utils import Planner, Pointer, Session


def load_task_text(task_config_path: str) -> str:
    with open(task_config_path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("text") or Path(task_config_path).stem.replace("_", " ")


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


def apply_planner_and_segment(
    planner_task: str,
    planner_session: Planner,
    molmo_session: Pointer,
    session: Session,
    counters: Dict[str, int],
    reset_memory_on_replan: bool,
    is_replan: bool,
):
    history = json.dumps(session.plan_history[-4:], ensure_ascii=False)
    plan = planner_session.plan(
        task_text=planner_task,
        image=session.current_image.copy(),
        current_interaction=session.segment_type,
        last_action_summary=session.last_action_summary,
        plan_count=session.plan_count,
        history=history,
        last_plan=session.last_plan,
    )
    counters["planner_calls"] += 1
    session.plan_count += 1
    session.last_plan = plan
    session.plan_history.append(plan)

    target_text = plan.get("target_text", "")
    session.segment_type = plan.get("interaction_type", session.segment_type)

    points = []
    if target_text:
        points = molmo_session.gen_point(image=session.current_image.copy(), prompt=target_text)
        counters["pointer_calls"] += 1

    session.clear_points()
    for x, y in points:
        session.points.append([int(x), int(y)])
        session.points_label.append(1)
    if points:
        session.segment()
        if is_replan and reset_memory_on_replan:
            session.clear_agent_memory(reset_counters=False)

    return {
        "plan": plan,
        "target_text": target_text,
        "interaction_type": session.segment_type,
        "points": [(int(x), int(y)) for x, y in points],
    }


def run_episode(
    episode_spec: EpisodeSpec,
    session: Session,
    planner_session: Planner,
    molmo_session: Pointer,
    stop_on_success: bool,
    reset_memory_on_replan: bool,
    planned_steps_override: Optional[int] = None,
    replan_interval_override: Optional[int] = None,
):
    session.reset(episode_spec.task_name, episode_spec.seed)
    tracker = TaskSuccessTracker(episode_spec.task_name, session.info)
    counters = {"planner_calls": 0, "pointer_calls": 0, "replans": 0}

    planned_steps = planned_steps_override or episode_spec.planned_steps
    replan_interval = max(1, replan_interval_override or episode_spec.replan_interval)

    initial_plan = apply_planner_and_segment(
        planner_task=episode_spec.planner_task,
        planner_session=planner_session,
        molmo_session=molmo_session,
        session=session,
        counters=counters,
        reset_memory_on_replan=reset_memory_on_replan,
        is_replan=False,
    )

    stop_reason = ""
    success_result = tracker.update(session.info)
    replan_events = []
    if initial_plan["plan"]:
        replan_events.append(
            {
                "step": 0,
                "planner_phase": initial_plan["plan"].get("planner_phase", "initial_plan"),
                "current_subtask_index": initial_plan["plan"].get("current_subtask_index", 0),
                "num_subtasks": len(initial_plan["plan"].get("subtasks", [])),
                "target_text": initial_plan["target_text"],
                "interaction_type": initial_plan["interaction_type"],
                "subgoal_status": initial_plan["plan"].get("subgoal_status", ""),
                "reasoning": initial_plan["plan"].get("reasoning", ""),
                "active_subtask_reasoning": initial_plan["plan"].get("active_subtask_reasoning", ""),
                "points": initial_plan["points"],
            }
        )

    if not initial_plan["points"]:
        stop_reason = "no_initial_points"

    for step_idx in range(int(planned_steps)):
        if stop_reason:
            break

        if step_idx > 0 and step_idx % replan_interval == 0:
            replanned = apply_planner_and_segment(
                planner_task=episode_spec.planner_task,
                planner_session=planner_session,
                molmo_session=molmo_session,
                session=session,
                counters=counters,
                reset_memory_on_replan=reset_memory_on_replan,
                is_replan=True,
            )
            counters["replans"] += 1
            replan_events.append(
                {
                    "step": step_idx,
                    "planner_phase": replanned["plan"].get("planner_phase", "review"),
                    "current_subtask_index": replanned["plan"].get("current_subtask_index", 0),
                    "num_subtasks": len(replanned["plan"].get("subtasks", [])),
                    "target_text": replanned["target_text"],
                    "interaction_type": replanned["interaction_type"],
                    "subgoal_status": replanned["plan"].get("subgoal_status", ""),
                    "reasoning": replanned["plan"].get("reasoning", ""),
                    "active_subtask_reasoning": replanned["plan"].get("active_subtask_reasoning", ""),
                    "points": replanned["points"],
                }
            )

        session.step()
        success_result = tracker.update(session.info)

        if stop_on_success and success_result.success:
            stop_reason = "success"
        elif session.last_terminated:
            stop_reason = "terminated"
        elif session.last_truncated:
            stop_reason = "truncated"
        elif session.last_plan.get("done", False):
            stop_reason = "planner_done"

    if not stop_reason:
        stop_reason = "max_steps"

    final_plan = session.last_plan or {}
    return {
        "task_name": episode_spec.task_name,
        "planner_task": episode_spec.planner_task,
        "split": episode_spec.split,
        "seed": episode_spec.seed,
        "factors": episode_spec_to_dict(episode_spec)["factors"],
        "success": success_result.success,
        "success_metric": success_result.metric,
        "success_progress": success_result.progress,
        "success_reason": success_result.reason,
        "stop_reason": stop_reason,
        "num_steps": session.num_steps,
        "planner_calls": counters["planner_calls"],
        "pointer_calls": counters["pointer_calls"],
        "replans": counters["replans"],
        "last_reward": session.last_reward,
        "last_action_summary": session.last_action_summary,
        "current_subtask_index": final_plan.get("current_subtask_index", 0),
        "num_subtasks": len(final_plan.get("subtasks", [])),
        "target_text": final_plan.get("target_text", ""),
        "interaction_type": final_plan.get("interaction_type", ""),
        "subgoal_status": final_plan.get("subgoal_status", ""),
        "planner_phase": final_plan.get("planner_phase", ""),
        "reasoning": final_plan.get("reasoning", ""),
        "active_subtask_reasoning": final_plan.get("active_subtask_reasoning", ""),
        "replan_events": replan_events,
    }


def summarize_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["task_name"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (split, task_name), task_rows in sorted(grouped.items()):
        success_values = [1.0 if row["success"] else 0.0 for row in task_rows]
        step_values = [int(row["num_steps"]) for row in task_rows]
        planner_calls = [int(row["planner_calls"]) for row in task_rows]
        pointer_calls = [int(row["pointer_calls"]) for row in task_rows]
        summary_rows.append(
            {
                "split": split,
                "task_name": task_name,
                "episodes": len(task_rows),
                "success_rate": round(sum(success_values) / len(success_values), 4),
                "mean_steps": round(statistics.mean(step_values), 2),
                "mean_planner_calls": round(statistics.mean(planner_calls), 2),
                "mean_pointer_calls": round(statistics.mean(pointer_calls), 2),
            }
        )
    return summary_rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-group", type=str, default="simple")
    parser.add_argument("--task-group-path", type=str, default="CraftJarvis/MineStudio_task_group.simple")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--split", type=str, default="id", choices=["id", "ood", "stress", "all"])
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--planned-steps", type=int, default=0)
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--reset-memory-on-replan", action="store_true")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/headless_eval")

    parser.add_argument("--model-path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--sam-path", type=str, required=True)
    parser.add_argument("--molmo-id", type=str, default="allenai/MolmoE-1B-0924")
    parser.add_argument("--molmo-url", type=str, default="http://127.0.0.1:9163")
    parser.add_argument("--molmo-api-key", type=str, default="EMPTY")
    parser.add_argument("--planner-id", type=str, default="gpt-4o-mini")
    parser.add_argument("--planner-url", type=str, default="http://127.0.0.1:9164")
    parser.add_argument("--planner-api-key", type=str, default="EMPTY")
    return parser


def main():
    args = build_argparser().parse_args()

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()] or None
    task_specs = resolve_task_specs(task_names)

    split_names = ["id", "ood", "stress"] if args.split == "all" else [args.split]

    file_list = prepare_task_configs(args.task_group, path=args.task_group_path)
    name_file_mapping = {name: file for name, file in file_list.items()}

    session = Session(
        model_path=args.model_path,
        sam_path=args.sam_path,
        name_file_mapping=name_file_mapping,
    )
    molmo_session = Pointer(
        model_id=args.molmo_id,
        model_url=args.molmo_url,
        api_key=args.molmo_api_key,
    )
    planner_session = Planner(
        model_id=args.planner_id,
        model_url=args.planner_url,
        api_key=args.planner_api_key,
    )

    out_dir = Path(args.out_dir)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task_group": args.task_group,
        "task_group_path": args.task_group_path,
        "tasks": [spec.task_name for spec in task_specs],
        "split": args.split,
        "episodes_per_task": args.episodes_per_task,
        "base_seed": args.base_seed,
        "planned_steps": args.planned_steps,
        "replan_interval": args.replan_interval,
        "reset_memory_on_replan": args.reset_memory_on_replan,
        "stop_on_success": args.stop_on_success,
        "save_video": args.save_video,
        "molmo_id": args.molmo_id,
        "molmo_url": args.molmo_url,
        "planner_id": args.planner_id,
        "planner_url": args.planner_url,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows: List[Dict[str, object]] = []
    try:
        for split in split_names:
            episode_specs = build_episode_specs(
                task_specs=task_specs,
                split=split,
                count=args.episodes_per_task,
                base_seed=args.base_seed,
            )
            for episode_spec in episode_specs:
                if episode_spec.task_name not in name_file_mapping:
                    raise KeyError(f"Task config not found for {episode_spec.task_name}")
                task_path = name_file_mapping[episode_spec.task_name]
                if not episode_spec.planner_task:
                    episode_spec = EpisodeSpec(
                        task_name=episode_spec.task_name,
                        planner_task=load_task_text(task_path),
                        split=episode_spec.split,
                        seed=episode_spec.seed,
                        factors=episode_spec.factors,
                        planned_steps=episode_spec.planned_steps,
                        replan_interval=episode_spec.replan_interval,
                    )

                row = run_episode(
                    episode_spec=episode_spec,
                    session=session,
                    planner_session=planner_session,
                    molmo_session=molmo_session,
                    stop_on_success=args.stop_on_success,
                    reset_memory_on_replan=args.reset_memory_on_replan,
                    planned_steps_override=args.planned_steps or None,
                    replan_interval_override=args.replan_interval or None,
                )
                all_rows.append(row)
                print(
                    "[headless-eval]",
                    json.dumps(
                        {
                            "task_name": row["task_name"],
                            "split": row["split"],
                            "seed": row["seed"],
                            "success": row["success"],
                            "stop_reason": row["stop_reason"],
                            "num_steps": row["num_steps"],
                            "planner_calls": row["planner_calls"],
                            "pointer_calls": row["pointer_calls"],
                            "subgoal_status": row["subgoal_status"],
                            "target_text": row["target_text"],
                        },
                        ensure_ascii=False,
                    ),
                )

                if args.save_video:
                    video_path = run_dir / "videos" / row["split"] / f"{row['task_name']}_seed{row['seed']}.mp4"
                    write_video(video_path, session.image_history)

                session.image_history = []
    finally:
        session.close()

    summary_rows = summarize_results(all_rows)
    write_jsonl(run_dir / "episode_results.jsonl", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[headless-eval] summary")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
