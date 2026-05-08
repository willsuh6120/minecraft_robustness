#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_blocks(raw: str) -> list[int]:
    blocks: list[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"Invalid block index: {value}")
        blocks.append(value)
    if not blocks:
        raise ValueError("At least one block index is required.")
    return blocks


def latest_subdir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No generated task group directory found under {root}")
    return sorted(candidates)[-1]


def run_command(cmd: list[str]) -> None:
    print("[prepare-o2-mixed-bank] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--env-conf-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--blocks", default="1,2,3,4,5")
    parser.add_argument("--mine-layout-backend", default="procedural")
    parser.add_argument("--mine-anchor-mode", default="source_or_fallback")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir).expanduser().resolve()
    env_conf_dir = Path(args.env_conf_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    plan_dir = asset_dir / "collect_block_plans"
    manifest_path = out_dir / "bank_manifest.json"

    if manifest_path.exists() and not args.overwrite:
        print(f"[prepare-o2-mixed-bank] reusing existing bank: {manifest_path}")
        print(out_dir)
        return

    if not plan_dir.is_dir():
        raise FileNotFoundError(f"Missing collect_block_plans directory: {plan_dir}")
    if not env_conf_dir.is_dir():
        raise FileNotFoundError(f"Missing env conf dir: {env_conf_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = parse_blocks(args.blocks)
    instance_worlds = []

    for instance_idx, block_idx in enumerate(blocks):
        block_tag = f"block_{block_idx:03d}"
        plan_path = plan_dir / f"{block_tag}_collect_plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing O2 collect plan: {plan_path}")

        instance_dir = out_dir / f"instance_{instance_idx:03d}_{block_tag}"
        generated_root = instance_dir / "generated_task_groups"
        generated_root.mkdir(parents=True, exist_ok=True)

        if args.overwrite:
            for child in generated_root.iterdir():
                if child.is_dir():
                    subprocess.run(["rm", "-rf", str(child)], check=True)
                else:
                    child.unlink()

        if not any(path.is_dir() for path in generated_root.iterdir()):
            run_command(
                [
                    str(args.python_bin),
                    "-m",
                    "minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen",
                    "--plan-json",
                    str(plan_path),
                    "--env-conf-dir",
                    str(env_conf_dir),
                    "--mine-layout-backend",
                    str(args.mine_layout_backend),
                    "--mine-anchor-mode",
                    str(args.mine_anchor_mode),
                    "--out-dir",
                    str(generated_root),
                ]
            )

        generated_dir = latest_subdir(generated_root)
        plan_rows = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest = generated_dir / "worldgen_manifest.json"
        instance_worlds.append(
            {
                "instance_idx": int(instance_idx),
                "source_block_idx": int(block_idx),
                "source_block_tag": block_tag,
                "task_group": f"o2_mixed_collect_{block_tag}",
                "task_group_path": str(generated_dir),
                "generated_task_group_dir": str(generated_dir),
                "manifest_path": str(manifest) if manifest.exists() else "",
                "plan_json": str(plan_path),
                "plan_rows": plan_rows,
                "split_label": "train_id",
            }
        )

    bank_manifest = {
        "split_label": "o2_mixed_collect",
        "tasks": ["mine_coal"],
        "bank_dir": str(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "asset_dir": str(asset_dir),
        "blocks": blocks,
        "instances_per_split": len(instance_worlds),
        "instance_worlds": instance_worlds,
    }
    manifest_path.write_text(json.dumps(bank_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare-o2-mixed-bank] wrote {manifest_path}")
    print(out_dir)


if __name__ == "__main__":
    main()
