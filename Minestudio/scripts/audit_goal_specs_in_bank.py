#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten_probe_pose(spec: dict[str, Any]) -> dict[str, Any]:
    pose = spec.get("probe_player_pose")
    if not isinstance(pose, dict):
        pose = {}
    return {
        "pose_x": pose.get("x", ""),
        "pose_y": pose.get("y", ""),
        "pose_z": pose.get("z", ""),
        "pose_yaw": pose.get("yaw", ""),
        "pose_pitch": pose.get("pitch", ""),
    }


def _find_specs(root: Path) -> list[Path]:
    if root.is_file() and root.name == "goal_spec.json":
        return [root]
    return sorted(root.rglob("goal_spec.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Asset bank, generated task group, or any directory containing goal_spec.json files.")
    parser.add_argument("--out-tsv", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec_path in _find_specs(args.root):
        spec = _load_json(spec_path)
        row = {
            "goal_spec_path": str(spec_path),
            "goal_image_path": spec.get("goal_image_path", ""),
            "goal_mode": spec.get("goal_mode", ""),
            "pose_family": spec.get("pose_family", ""),
            "hint_name": spec.get("hint_name", ""),
            "sampled_distance": spec.get("sampled_distance", ""),
            "sampled_center_distance": spec.get("sampled_center_distance", ""),
            "sampled_face_distance": spec.get("sampled_face_distance", ""),
            "clearance_backoff": spec.get("clearance_backoff", ""),
            "num_pose_candidates": spec.get("num_pose_candidates", ""),
            "pose_success_count": spec.get("pose_success_count", ""),
            "pose_rejected_collision_count": spec.get("pose_rejected_collision_count", ""),
            "sweep_success_count": spec.get("sweep_success_count", ""),
            "sweep_rejected_collision_count": spec.get("sweep_rejected_collision_count", ""),
            "target_world_center": json.dumps(spec.get("target_world_center", ""), ensure_ascii=False),
        }
        row.update(_flatten_probe_pose(spec))
        rows.append(row)

    rows.sort(key=lambda item: str(item["goal_spec_path"]))
    fields = [
        "goal_spec_path",
        "goal_image_path",
        "goal_mode",
        "pose_family",
        "hint_name",
        "sampled_distance",
        "sampled_center_distance",
        "sampled_face_distance",
        "clearance_backoff",
        "pose_pitch",
        "pose_yaw",
        "pose_x",
        "pose_y",
        "pose_z",
        "target_world_center",
        "num_pose_candidates",
        "pose_success_count",
        "pose_rejected_collision_count",
        "sweep_success_count",
        "sweep_rejected_collision_count",
    ]
    if args.out_tsv:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"goal_specs={len(rows)}")
    for row in rows:
        print(
            "\t".join(
                str(row.get(key, ""))
                for key in (
                    "pose_family",
                    "hint_name",
                    "sampled_distance",
                    "sampled_center_distance",
                    "clearance_backoff",
                    "pose_pitch",
                    "goal_spec_path",
                )
            )
        )


if __name__ == "__main__":
    main()
