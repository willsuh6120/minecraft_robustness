import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import VoxelsCallback, load_callbacks_from_config
from minestudio.tutorials.inference.evaluate_rocket.crossview_utils import (
    AUTO_GOAL_BLOCK_TYPES,
    _configured_target_entries,
    _normalize_name,
    _player_pose,
    _voxel_world_center_from_player_pose,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import canonical_task_key


DEFAULT_VOXEL_BOUNDS = [-20, 20, -10, 10, -20, 20]
DEFAULT_TARGET_TOLERANCE = 0.75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/evaluate_rocket/validated_template_banks")
    parser.add_argument("--copy-valid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument("--repair-nearest-within", type=float, default=0.0)
    parser.add_argument("--post-reset-noops", type=int, default=2)
    return parser.parse_args()


def parse_task_names(tasks: str) -> List[str]:
    return [task.strip() for task in str(tasks).split(",") if task.strip()]


def find_candidate_yamls(source_dir: Path, task_config_name: str) -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    for path in sorted(source_dir.rglob(f"{task_config_name}.yaml")):
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(path)
    return candidates


def build_probe_env(task_config: Dict):
    callbacks = list(load_callbacks_from_config(task_config))
    if not any(isinstance(callback, VoxelsCallback) for callback in callbacks):
        callbacks.append(VoxelsCallback(DEFAULT_VOXEL_BOUNDS))
    return MinecraftSim(seed=0, preferred_spawn_biome="plains", callbacks=callbacks)


def matching_target_voxels(task_key: str, info: Dict) -> List[Dict]:
    target_names = AUTO_GOAL_BLOCK_TYPES.get(task_key, ())
    player_pose = _player_pose(info)
    voxels = info.get("voxels") or []
    matches: List[Dict] = []
    if not isinstance(voxels, list):
        return matches
    for voxel in voxels:
        if not isinstance(voxel, dict):
            continue
        voxel_type = _normalize_name(voxel.get("type"))
        if not voxel_type or not any(name in voxel_type for name in target_names):
            continue
        raw_offset = (
            float(voxel.get("x", 0.0)),
            float(voxel.get("y", 0.0)),
            float(voxel.get("z", 0.0)),
        )
        world_center = _voxel_world_center_from_player_pose(player_pose, raw_offset)
        matches.append(
            {
                "type": voxel_type,
                "world_center": [float(world_center[0]), float(world_center[1]), float(world_center[2])],
            }
        )
    return matches


def l2_distance(a: Tuple[float, float, float] | List[float], b: Tuple[float, float, float] | List[float]) -> float:
    return (
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    ) ** 0.5


def safe_distance(value: float | None, fallback: float = 999.0) -> float:
    return float(fallback if value is None else value)


def build_repaired_task_config(task_config: Dict, entry_reports: List[Dict]) -> Dict:
    repaired = copy.deepcopy(task_config)
    repaired_blocks: List[Dict] = []
    repair_distances: List[float] = []
    for entry in entry_reports:
        nearest = entry.get("nearest_actual_target") or {}
        actual_center = list(nearest.get("world_center") or [])
        if len(actual_center) != 3:
            continue
        configured = entry.get("configured_target") or {}
        repaired_blocks.append(
            {
                "type": str(configured.get("target_type") or nearest.get("type") or ""),
                "world_center": [float(actual_center[0]), float(actual_center[1]), float(actual_center[2])],
            }
        )
        repair_distances.append(float(entry.get("distance") or 0.0))
    if repaired_blocks:
        repaired["target_blocks"] = repaired_blocks
        if len(repaired_blocks) == 1:
            repaired["target_world_center"] = list(repaired_blocks[0]["world_center"])
            repaired["target_type"] = str(repaired_blocks[0]["type"])
    metadata = repaired.get("target_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        repaired["target_metadata"] = metadata
    metadata["validator_repair_applied"] = True
    metadata["validator_repair_distances"] = repair_distances
    metadata["target_source"] = "validated_bank"
    return repaired


def validate_template(task_config_name: str, yaml_path: Path, target_tolerance: float, post_reset_noops: int) -> Dict:
    task_key = canonical_task_key(task_config_name)
    task_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    configured_targets = _configured_target_entries(task_key, task_config)
    if not configured_targets:
        return {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "yaml_path": str(yaml_path.resolve()),
            "valid": False,
            "reason": "missing_configured_target_metadata",
        }

    env = None
    try:
        env = build_probe_env(task_config)
        _, info = env.reset()
        for _ in range(max(0, int(post_reset_noops))):
            _, _, terminated, truncated, info = env.step(env.noop_action())
            if terminated or truncated:
                break
        actual_targets = matching_target_voxels(task_key, info)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    if not actual_targets:
        return {
            "task_config_name": task_config_name,
            "task_key": task_key,
            "yaml_path": str(yaml_path.resolve()),
            "valid": False,
            "reason": "no_matching_target_voxels_found",
            "configured_targets": configured_targets,
            "actual_target_count": 0,
            "actual_targets_preview": [],
        }

    entry_reports: List[Dict] = []
    all_valid = True
    for configured in configured_targets:
        configured_center = configured["world_center"]
        best = None
        best_distance = None
        for actual in actual_targets:
            distance = l2_distance(configured_center, actual["world_center"])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = actual
        entry_valid = best is not None and safe_distance(best_distance) <= float(target_tolerance)
        all_valid = all_valid and bool(entry_valid)
        entry_reports.append(
            {
                "configured_target": configured,
                "nearest_actual_target": best,
                "distance": round(safe_distance(best_distance), 4),
                "valid": bool(entry_valid),
            }
        )

    return {
        "task_config_name": task_config_name,
        "task_key": task_key,
        "yaml_path": str(yaml_path.resolve()),
        "valid": bool(all_valid),
        "reason": "ok" if all_valid else "configured_target_mismatch",
        "configured_targets": configured_targets,
        "actual_target_count": len(actual_targets),
        "actual_targets_preview": actual_targets[:16],
        "entry_reports": entry_reports,
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_root = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    task_names = parse_task_names(args.tasks)

    summary: Dict[str, object] = {
        "source_dir": str(source_dir.resolve()),
        "tasks": {},
        "validated_bank_dir": "",
    }

    validated_bank_dir = out_root / "validated_bank"
    if args.copy_valid:
        validated_bank_dir.mkdir(parents=True, exist_ok=True)
        summary["validated_bank_dir"] = str(validated_bank_dir.resolve())

    for task_config_name in task_names:
        candidates = find_candidate_yamls(source_dir, task_config_name)
        reports: List[Dict] = []
        valid_reports: List[Dict] = []
        for candidate_idx, yaml_path in enumerate(candidates):
            task_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            report = validate_template(
                task_config_name=task_config_name,
                yaml_path=yaml_path,
                target_tolerance=float(args.target_tolerance),
                post_reset_noops=int(args.post_reset_noops),
            )
            report["candidate_index"] = int(candidate_idx)
            report["repair_applied"] = False
            report["repaired_yaml_path"] = ""
            if not bool(report.get("valid")) and float(args.repair_nearest_within) > 0.0:
                entry_reports = report.get("entry_reports") or []
                if entry_reports and all(
                    bool(entry.get("nearest_actual_target"))
                    and safe_distance(entry.get("distance")) <= float(args.repair_nearest_within)
                    for entry in entry_reports
                ):
                    repaired_dir = out_root / "repaired_templates" / task_config_name
                    repaired_dir.mkdir(parents=True, exist_ok=True)
                    repaired_yaml_path = repaired_dir / f"{yaml_path.parent.name}_{task_config_name}.yaml"
                    repaired_config = build_repaired_task_config(task_config, entry_reports)
                    repaired_yaml_path.write_text(
                        yaml.safe_dump(repaired_config, sort_keys=False, allow_unicode=True),
                        encoding="utf-8",
                    )
                    report["repair_applied"] = True
                    report["repaired_yaml_path"] = str(repaired_yaml_path.resolve())
                    report["reason"] = "repaired_to_nearest_actual_target"
                    report["valid"] = True
            reports.append(report)
            if bool(report.get("valid")):
                valid_reports.append(report)

        if args.copy_valid:
            task_bank_dir = validated_bank_dir / task_config_name
            if task_bank_dir.exists() and args.overwrite:
                shutil.rmtree(task_bank_dir)
            task_bank_dir.mkdir(parents=True, exist_ok=True)
            for template_idx, report in enumerate(valid_reports):
                dst_dir = task_bank_dir / f"template_{template_idx:03d}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                src_yaml = str(report.get("repaired_yaml_path") or report["yaml_path"])
                shutil.copy2(src_yaml, dst_dir / f"{task_config_name}.yaml")

        summary["tasks"][task_config_name] = {
            "num_candidates": len(candidates),
            "num_valid": len(valid_reports),
            "reports": reports,
        }

    summary_path = out_root / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
