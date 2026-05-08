import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from minestudio.benchmark import prepare_task_configs
from minestudio.tutorials.inference.evaluate_rocket.crossview_utils import (
    AUTO_GOAL_SUPPORTED_TASKS,
    CrossViewSession,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_protocol import (
    apply_protocol_to_task_specs,
    resolve_interaction_protocol,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    InteractionBenchmarkTaskSpec,
    resolve_interaction_task_specs,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-source", type=str, default="rocket2_official", choices=["rocket2_official"])
    parser.add_argument("--task-group", type=str, required=True)
    parser.add_argument("--task-group-path", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="ours_v1")
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--warmup-noop-steps", type=int, default=None)
    parser.add_argument("--episode-retries", type=int, default=2)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--cfg-coef", type=float, default=None)
    parser.add_argument("--goal-subdir", type=str, default="_baked_goals")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-assets", action="store_true")
    parser.add_argument("--refresh-task-configs", action="store_true")
    return parser


def parse_task_names(raw: str) -> List[str]:
    return [task.strip() for task in str(raw).split(",") if task.strip()]


def _load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: Dict) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _task_yaml_path(task_group_path: Path, task_config_name: str) -> Path:
    return task_group_path / f"{task_config_name}.yaml"


def _resolved_existing_baked_goal(task_yaml_path: Path) -> Optional[Path]:
    if not task_yaml_path.exists():
        return None
    data = _load_yaml(task_yaml_path)
    baked_goal_spec_path = str(data.get("baked_goal_spec_path") or "").strip()
    if not baked_goal_spec_path:
        return None
    goal_spec_path = Path(baked_goal_spec_path)
    if goal_spec_path.exists():
        return goal_spec_path.resolve()
    return None


def _goal_output_dir(task_group_path: Path, goal_subdir: str, task_config_name: str, protocol_name: str, base_seed: int) -> Path:
    return (
        task_group_path
        / str(goal_subdir).strip()
        / task_config_name
        / f"protocol_{protocol_name}"
        / f"seed_{int(base_seed):06d}"
    )


def _update_task_yaml(
    task_group_path: Path,
    task_config_name: str,
    goal_assets: Dict,
    *,
    protocol_name: str,
    base_seed: int,
) -> Path:
    task_yaml_path = _task_yaml_path(task_group_path, task_config_name)
    if not task_yaml_path.exists():
        raise FileNotFoundError(f"Task YAML not found for baked goal update: {task_yaml_path}")
    data = _load_yaml(task_yaml_path)
    data["baked_goal_spec_path"] = str(goal_assets["goal_spec_path"])
    data["baked_goal_dir"] = str(Path(goal_assets["goal_spec_path"]).parent.resolve())
    data["baked_goal_protocol"] = str(protocol_name)
    data["baked_goal_seed"] = int(base_seed)
    data["baked_goal_mode"] = "fixed_from_auto"
    _dump_yaml(task_yaml_path, data)
    return task_yaml_path


def _update_worldgen_manifest(
    task_group_path: Path,
    task_config_name: str,
    goal_assets: Dict,
    *,
    protocol_name: str,
    base_seed: int,
    debug_assets: Optional[Dict],
) -> Optional[Path]:
    manifest_path = task_group_path / "worldgen_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_rows = list(manifest.get("tasks") or [])
    for row in task_rows:
        if str(row.get("task_config_name", "")).strip() != str(task_config_name).strip():
            continue
        baked_goal = {
            "goal_spec_path": str(goal_assets["goal_spec_path"]),
            "goal_image_path": str(goal_assets["goal_image_path"]),
            "goal_mask_path": str(goal_assets["goal_mask_path"]),
            "goal_bbox_overlay_path": str(goal_assets.get("goal_bbox_overlay_path") or ""),
            "goal_mask_overlay_path": str(goal_assets.get("goal_mask_overlay_path") or ""),
            "protocol": str(protocol_name),
            "seed": int(base_seed),
        }
        if debug_assets:
            baked_goal["debug_assets"] = debug_assets
        row["baked_goal"] = baked_goal
        row["baked_goal_spec_path"] = baked_goal["goal_spec_path"]
        row["baked_goal_protocol"] = str(protocol_name)
        row["baked_goal_seed"] = int(base_seed)
        break
    manifest["goal_baked"] = True
    manifest["goal_bake_protocol"] = str(protocol_name)
    manifest["goal_bake_seed"] = int(base_seed)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _build_auto_goal_session(
    task_spec: InteractionBenchmarkTaskSpec,
    name_file_mapping: Dict[str, str],
    *,
    model_path: str,
    cfg_coef: float,
) -> CrossViewSession:
    return CrossViewSession(
        model_path=model_path,
        name_file_mapping=name_file_mapping,
        goal_segment_type=task_spec.subtasks[0].interaction_type,
        cfg_coef=cfg_coef,
        obs_size=(224, 224),
        auto_goal_task_spec=task_spec,
    )


def main():
    args = build_argparser().parse_args()
    task_names = parse_task_names(args.tasks)
    if not task_names:
        raise ValueError("No tasks provided for goal baking.")

    protocol = resolve_interaction_protocol(args.protocol)
    warmup_noop_steps = protocol.warmup_noop_steps if args.warmup_noop_steps is None else int(args.warmup_noop_steps)
    task_specs = resolve_interaction_task_specs(task_names, env_source=args.env_source)
    task_specs = apply_protocol_to_task_specs(task_specs, protocol_name=args.protocol)
    task_group_path = Path(args.task_group_path)
    refresh_task_configs = bool(args.refresh_task_configs or task_group_path.is_dir())
    name_file_mapping = prepare_task_configs(args.task_group, path=str(task_group_path), refresh=refresh_task_configs)
    model_path = args.model_path or "hf:phython96/ROCKET-2-1x-22w"
    cfg_coef = float(1.5 if args.cfg_coef is None else args.cfg_coef)

    baked_root = task_group_path / str(args.goal_subdir).strip()
    baked_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for task_spec in task_specs:
        task_key = str(task_spec.task_key or task_spec.task_config_name)
        if task_key not in AUTO_GOAL_SUPPORTED_TASKS:
            supported = ", ".join(sorted(AUTO_GOAL_SUPPORTED_TASKS))
            raise NotImplementedError(
                f"Goal baking is only implemented for auto-goal tasks: {supported}. Got {task_key}."
            )

        task_yaml_path = _task_yaml_path(task_group_path, task_spec.task_config_name)
        existing_goal_spec_path = _resolved_existing_baked_goal(task_yaml_path)
        if existing_goal_spec_path is not None and not args.overwrite:
            summary_rows.append(
                {
                    "task_config_name": task_spec.task_config_name,
                    "task_key": task_key,
                    "goal_spec_path": str(existing_goal_spec_path),
                    "status": "reused_existing",
                    "task_yaml_path": str(task_yaml_path.resolve()),
                }
            )
            print(json.dumps(summary_rows[-1], ensure_ascii=False))
            continue

        output_dir = _goal_output_dir(
            task_group_path=task_group_path,
            goal_subdir=args.goal_subdir,
            task_config_name=task_spec.task_config_name,
            protocol_name=args.protocol,
            base_seed=int(args.base_seed),
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        session: Optional[CrossViewSession] = None
        last_exc = None
        goal_assets: Optional[Dict] = None
        debug_assets: Optional[Dict] = None
        try:
            for attempt in range(int(max(1, args.episode_retries))):
                try:
                    session = _build_auto_goal_session(
                        task_spec=task_spec,
                        name_file_mapping=name_file_mapping,
                        model_path=model_path,
                        cfg_coef=cfg_coef,
                    )
                    session.reset(
                        task_spec.task_config_name,
                        int(args.base_seed),
                        warmup_noop_steps=int(warmup_noop_steps),
                    )
                    goal_assets = session.save_goal_assets(output_dir)
                    if args.save_debug_assets:
                        debug_assets = session.save_debug_assets(output_dir)
                    break
                except Exception as exc:
                    last_exc = exc
                    if args.save_debug_assets and session is not None:
                        failure_dir = output_dir / f"attempt_{attempt + 1:02d}_failed"
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        debug_assets = session.save_debug_assets(failure_dir)
                        (failure_dir / "failure.json").write_text(
                            json.dumps(
                                {
                                    "task_config_name": task_spec.task_config_name,
                                    "seed": int(args.base_seed),
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
                        if session is not None:
                            session.close()
                    except Exception:
                        pass
                    session = None
                    print(
                        f"[interaction-bake-goal-assets][task-error] task={task_spec.task_config_name} "
                        f"seed={int(args.base_seed)} attempt={attempt + 1} {type(exc).__name__}: {exc}"
                    )
            if goal_assets is None:
                raise last_exc or RuntimeError(f"Failed to bake goal assets for {task_spec.task_config_name}")

            updated_yaml_path = _update_task_yaml(
                task_group_path=task_group_path,
                task_config_name=task_spec.task_config_name,
                goal_assets=goal_assets,
                protocol_name=args.protocol,
                base_seed=int(args.base_seed),
            )
            manifest_path = _update_worldgen_manifest(
                task_group_path=task_group_path,
                task_config_name=task_spec.task_config_name,
                goal_assets=goal_assets,
                protocol_name=args.protocol,
                base_seed=int(args.base_seed),
                debug_assets=debug_assets,
            )
            row = {
                "task_config_name": task_spec.task_config_name,
                "task_key": task_key,
                "status": "baked",
                "goal_spec_path": str(goal_assets["goal_spec_path"]),
                "goal_dir": str(Path(goal_assets["goal_spec_path"]).parent.resolve()),
                "task_yaml_path": str(updated_yaml_path.resolve()),
                "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else "",
                "protocol": str(args.protocol),
                "seed": int(args.base_seed),
            }
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        finally:
            try:
                if session is not None:
                    session.close()
            except Exception:
                pass

    summary_path = baked_root / "bake_manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_group": str(args.task_group),
                "task_group_path": str(task_group_path.resolve()),
                "protocol": str(args.protocol),
                "base_seed": int(args.base_seed),
                "tasks": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved baked goals to {summary_path}")


if __name__ == "__main__":
    main()
