#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True, help="assets/<timestamp> dir that contains eval_bank/")
    parser.add_argument("--task-config-name", default="mine_coal")
    parser.add_argument("--decision-steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--tool-type", default="diamond_pickaxe")
    return parser.parse_args()


def iter_task_yamls(asset_dir: Path, task_config_name: str):
    pattern = f"eval_bank/instance_*/generated_task_groups/*/{task_config_name}.yaml"
    yield from sorted(asset_dir.glob(pattern))


def patch_task_yaml(path: Path, *, effective_time_limit: int, tool_type: str) -> bool:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    changed = False

    if int(payload.get("time_limit") or 0) != int(effective_time_limit):
        payload["time_limit"] = int(effective_time_limit)
        changed = True

    desired_inventory = [{"slot": 0, "type": str(tool_type), "quantity": 1}]
    if payload.get("init_inventory") != desired_inventory:
        payload["init_inventory"] = desired_inventory
        changed = True

    if changed:
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return changed


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir).resolve()
    if not (asset_dir / "eval_bank").is_dir():
        raise FileNotFoundError(f"invalid asset dir: {asset_dir}")

    effective_time_limit = int(args.decision_steps) + int(args.warmup_steps)
    touched = 0
    changed = 0
    for yaml_path in iter_task_yamls(asset_dir, args.task_config_name):
        touched += 1
        if patch_task_yaml(
            yaml_path,
            effective_time_limit=effective_time_limit,
            tool_type=str(args.tool_type),
        ):
            changed += 1

    print(
        {
            "asset_dir": str(asset_dir),
            "task_config_name": str(args.task_config_name),
            "decision_steps": int(args.decision_steps),
            "warmup_steps": int(args.warmup_steps),
            "effective_time_limit": int(effective_time_limit),
            "tool_type": str(args.tool_type),
            "task_yaml_count": int(touched),
            "changed_count": int(changed),
        }
    )


if __name__ == "__main__":
    main()
