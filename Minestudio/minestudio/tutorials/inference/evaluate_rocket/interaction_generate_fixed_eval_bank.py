import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    canonical_task_key,
    classify_factor_split,
    factor_levels_to_worldgen_suggestions,
    hard_factor_count,
    sample_factor_levels_for_split,
)


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_OUT_DIR = "outputs/evaluate_rocket/fixed_eval_banks"
SPLIT_ALIASES = {
    "id": "train_id",
    "train": "train_id",
    "train_id": "train_id",
    "ood": "ood",
    "stress": "stress",
    "clean": "clean",
}

MINE_LAYOUT_CASE_SPECS: Dict[str, Dict[str, str]] = {
    "straight": {"blueprint_id": "straight_tunnel", "target_sign": "any"},
    "offset_chamber_left": {"blueprint_id": "offset_chamber", "target_sign": "neg"},
    "offset_chamber_right": {"blueprint_id": "offset_chamber", "target_sign": "pos"},
    "turn_left": {"blueprint_id": "turn_left", "target_sign": "neg"},
    "turn_right": {"blueprint_id": "turn_right", "target_sign": "pos"},
    "side_alcove_left": {"blueprint_id": "side_alcove_left", "target_sign": "neg"},
    "side_alcove_right": {"blueprint_id": "side_alcove_right", "target_sign": "pos"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train_id,ood,stress")
    parser.add_argument("--instances-per-split", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--env-conf-dir", type=str, default=DEFAULT_ENV_CONF_DIR)
    parser.add_argument("--protocol", type=str, default="ours_v1")
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
    parser.add_argument(
        "--mine-layout-case",
        type=str,
        default="",
        choices=[""] + sorted(MINE_LAYOUT_CASE_SPECS.keys()),
        help="Lock mine fixed-bank generation to a single layout family/sign.",
    )
    parser.add_argument("--bake-goals", action="store_true")
    parser.add_argument("--bake-goal-base-seed", type=int, default=1)
    parser.add_argument("--bake-goal-episode-retries", type=int, default=2)
    parser.add_argument("--bake-goal-save-debug-assets", action="store_true")
    parser.add_argument(
        "--max-instance-attempts",
        type=int,
        default=6,
        help="How many times to regenerate/resample a fixed-bank instance before giving up.",
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_task_names(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_split_names(raw: str) -> List[str]:
    names: List[str] = []
    for item in str(raw).split(","):
        key = SPLIT_ALIASES.get(str(item).strip().lower())
        if key and key not in names:
            names.append(key)
    return names


def latest_subdir(path: Path) -> Path:
    children = sorted([child for child in path.iterdir() if child.is_dir()])
    if not children:
        raise RuntimeError(f"No generated directories found under {path}")
    return children[-1]


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    print("[interaction-generate-fixed-eval-bank] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def list_child_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([child for child in path.iterdir() if child.is_dir()])


def detect_generated_dir(path: Path, before_dirs: List[Path]) -> Path:
    after_dirs = list_child_dirs(path)
    before_set = {item.resolve() for item in before_dirs}
    new_dirs = [item for item in after_dirs if item.resolve() not in before_set]
    if new_dirs:
        return new_dirs[-1]
    if after_dirs:
        return after_dirs[-1]
    raise RuntimeError(f"No generated directories found under {path}")


def build_worldgen_cmd(
    plan_json_path: Path,
    env_conf_dir: Path,
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
        str(env_conf_dir),
        "--mine-layout-backend",
        str(mine_layout_backend),
        "--mine-anchor-mode",
        str(mine_anchor_mode),
        "--out-dir",
        str(out_dir),
    ]


def build_bake_goal_cmd(
    *,
    task_group: str,
    task_group_path: Path,
    tasks: List[str],
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
        ",".join(tasks),
        "--base-seed",
        str(base_seed),
        "--episode-retries",
        str(episode_retries),
    ]
    if save_debug_assets:
        cmd.append("--save-debug-assets")
    return cmd


def stable_layout_seed(
    task_config_name: str,
    split_label: str,
    instance_idx: int,
    base_seed: int,
    factor_levels: Dict[str, int],
    layout_case: str = "",
) -> int:
    payload = {
        "task_config_name": str(task_config_name),
        "split_label": str(split_label),
        "instance_idx": int(instance_idx),
        "base_seed": int(base_seed),
        "factor_levels": factor_levels,
        "layout_case": str(layout_case or ""),
    }
    return abs(hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))) % (2**31 - 1)


def resample_layout_seed(raw_seed: int, attempt_idx: int, row_idx: int) -> int:
    modulus = 2**31 - 1
    try:
        base_seed = int(raw_seed)
    except Exception:
        base_seed = 0
    resampled = (base_seed + int(attempt_idx) * 104_729 + int(row_idx) * 1_009) % modulus
    return resampled or (104_729 + int(row_idx) + 1)


def resample_plan_rows(plan_rows: List[Dict], attempt_idx: int) -> List[Dict]:
    rows: List[Dict] = []
    for row_idx, row in enumerate(plan_rows):
        new_row = dict(row)
        new_row["layout_seed"] = resample_layout_seed(row.get("layout_seed", 0), attempt_idx=attempt_idx, row_idx=row_idx)
        new_row["resample_attempt"] = int(attempt_idx)
        rows.append(new_row)
    return rows


def write_attempt_failure(instance_dir: Path, attempt_idx: int, payload: Dict) -> None:
    failure_path = instance_dir / f"attempt_{int(attempt_idx) + 1:02d}_failure.json"
    failure_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _progress_payload(*, split_label: str, split_idx: int, total_splits: int, instance_idx: int, instances_per_split: int, attempt_idx: int, max_attempts: int) -> Dict[str, object]:
    overall_total = int(total_splits) * int(instances_per_split)
    overall_index = int(split_idx) * int(instances_per_split) + int(instance_idx) + 1
    return {
        "split_label": str(split_label),
        "split_progress": f"{int(split_idx) + 1}/{int(total_splits)}",
        "instance_progress": f"{int(instance_idx) + 1}/{int(instances_per_split)}",
        "overall_progress": f"{overall_index}/{overall_total}",
        "attempt_progress": f"{int(attempt_idx) + 1}/{int(max_attempts)}",
    }


def _mine_fixed_recipe(task_key: str, split_label: str, instance_idx: int) -> Dict | None:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return None
    recipes = {
        "clean": [
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
        ],
        "train_id": [
            {"factor_levels": {"R": 2, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 2, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 1, "H": 0, "O": 1, "C": 0, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 2, "A": 0}, "blueprint_id": "side_alcove_left"},
            {"factor_levels": {"R": 0, "H": 1, "O": 0, "C": 0, "P": 2, "A": 0}, "blueprint_id": "turn_right"},
            {"factor_levels": {"R": 1, "H": 1, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 0, "H": 2, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 2, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
        ],
        "ood": [
            {"factor_levels": {"R": 2, "H": 0, "O": 1, "C": 0, "P": 2, "A": 0}, "blueprint_id": "turn_left"},
            {"factor_levels": {"R": 1, "H": 2, "O": 0, "C": 0, "P": 2, "A": 0}, "blueprint_id": "side_alcove_right"},
            {"factor_levels": {"R": 0, "H": 0, "O": 2, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 2, "H": 0, "O": 0, "C": 2, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 1, "H": 0, "O": 0, "C": 0, "P": 2, "A": 2}, "blueprint_id": "side_alcove_left"},
            {"factor_levels": {"R": 2, "H": 1, "O": 1, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 0, "H": 2, "O": 0, "C": 2, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 1, "H": 0, "O": 2, "C": 0, "P": 2, "A": 0}, "blueprint_id": "turn_right"},
        ],
        "stress": [
            {"factor_levels": {"R": 3, "H": 0, "O": 1, "C": 0, "P": 2, "A": 0}, "blueprint_id": "turn_left"},
            {"factor_levels": {"R": 2, "H": 0, "O": 3, "C": 0, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 1, "H": 0, "O": 0, "C": 0, "P": 3, "A": 0}, "blueprint_id": "side_alcove_right"},
            # Keep this as a stress-level action/path case, but avoid the
            # turn_right+A3 combination which repeatedly produced unbakeable
            # mine_coal goals under the stricter goal-visibility validator.
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 2, "A": 3}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 2, "H": 3, "O": 0, "C": 0, "P": 0, "A": 0}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 0, "H": 0, "O": 2, "C": 3, "P": 0, "A": 0}, "blueprint_id": "offset_chamber"},
            {"factor_levels": {"R": 3, "H": 0, "O": 0, "C": 0, "P": 0, "A": 3}, "blueprint_id": "straight_tunnel"},
            {"factor_levels": {"R": 0, "H": 3, "O": 0, "C": 0, "P": 3, "A": 0}, "blueprint_id": "side_alcove_left"},
        ],
    }
    options = recipes.get(split_label)
    if not options:
        return None
    return dict(options[int(instance_idx) % len(options)])


def _mine_layout_fixed_recipe(task_key: str, split_label: str, instance_idx: int, layout_case: str) -> Dict | None:
    if task_key not in {"mine_coal", "mine_emerald"}:
        return None
    if layout_case not in MINE_LAYOUT_CASE_SPECS:
        return None
    recipes = {
        "clean": [
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}},
        ],
        "train_id": [
            {"factor_levels": {"R": 2, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 2, "O": 0, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 2, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 2, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 1, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 2, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 1}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 2}},
        ],
        "ood": [
            {"factor_levels": {"R": 2, "H": 0, "O": 1, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 2, "O": 0, "C": 1, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 2, "C": 2, "P": 0, "A": 0}},
            {"factor_levels": {"R": 1, "H": 0, "O": 0, "C": 0, "P": 2, "A": 0}},
            {"factor_levels": {"R": 0, "H": 1, "O": 0, "C": 0, "P": 0, "A": 2}},
            {"factor_levels": {"R": 1, "H": 0, "O": 0, "C": 0, "P": 2, "A": 2}},
            {"factor_levels": {"R": 2, "H": 1, "O": 1, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 1, "O": 1, "C": 0, "P": 1, "A": 0}},
        ],
        "stress": [
            {"factor_levels": {"R": 3, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 3, "O": 0, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 3, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 3, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 3, "A": 0}},
            {"factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 3}},
            {"factor_levels": {"R": 2, "H": 0, "O": 2, "C": 0, "P": 0, "A": 0}},
            {"factor_levels": {"R": 0, "H": 2, "O": 0, "C": 0, "P": 3, "A": 0}},
        ],
    }
    options = recipes.get(split_label)
    if not options:
        return None
    recipe = dict(options[int(instance_idx) % len(options)])
    recipe.update(MINE_LAYOUT_CASE_SPECS[layout_case])
    return recipe


def build_plan_row(
    task_config_name: str,
    split_label: str,
    instance_idx: int,
    base_seed: int,
    rng: random.Random,
    mine_layout_case: str = "",
) -> Dict:
    task_key = canonical_task_key(task_config_name)
    recipe = _mine_layout_fixed_recipe(task_key, split_label, instance_idx, mine_layout_case) or _mine_fixed_recipe(task_key, split_label, instance_idx)
    factor_levels = dict(recipe["factor_levels"]) if recipe else sample_factor_levels_for_split(task_key, split_label, rng)
    suggestions = {
        **factor_levels_to_worldgen_suggestions(task_key, factor_levels),
        "notes": (
            f"Fixed held-out eval bank row for split={split_label}, instance={instance_idx}."
            + (f" layout_case={mine_layout_case}." if mine_layout_case else "")
        ),
    }
    if recipe and recipe.get("blueprint_id"):
        suggestions["mine_blueprint_id"] = str(recipe["blueprint_id"])
    if recipe and recipe.get("target_sign") and str(recipe["target_sign"]) != "any":
        suggestions["mine_target_sign"] = str(recipe["target_sign"])
    return {
        "task_config_name": str(task_config_name),
        "task_key": str(task_key),
        "primary_failure_mode_majority": f"fixed_eval_{split_label}",
        "primary_factors_majority": [code for code, value in factor_levels.items() if int(value) > 0],
        "severity_majority": max([int(value) for value in factor_levels.values()] or [0]),
        "factor_levels": factor_levels,
        "layout_seed": stable_layout_seed(
            task_config_name,
            split_label,
            instance_idx,
            base_seed,
            factor_levels,
            layout_case=mine_layout_case,
        ),
        "template_index": int(instance_idx),
        "requested_split_label": str(split_label),
        "computed_split_label": classify_factor_split(task_key, factor_levels),
        "hard_factor_count": int(hard_factor_count(task_key, factor_levels)),
        "requested_layout_case": str(mine_layout_case or ""),
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": suggestions,
    }


def main():
    args = parse_args()
    task_names = parse_task_names(args.tasks)
    split_names = parse_split_names(args.splits)
    if not task_names:
        raise ValueError("No tasks provided.")
    if not split_names:
        raise ValueError("No valid split labels provided.")

    out_root = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    env_conf_dir = Path(args.env_conf_dir)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    root_manifest = {
        "config": {
            "tasks": task_names,
            "splits": split_names,
            "instances_per_split": int(args.instances_per_split),
            "base_seed": int(args.base_seed),
            "env_conf_dir": str(env_conf_dir.resolve()),
            "mine_layout_backend": str(args.mine_layout_backend),
            "mine_layout_case": str(args.mine_layout_case),
            "protocol": str(args.protocol),
            "bake_goals": bool(args.bake_goals),
            "bake_goal_base_seed": int(args.bake_goal_base_seed),
            "max_instance_attempts": int(args.max_instance_attempts),
        },
        "root_dir": str(out_root.resolve()),
        "split_dirs": {},
    }

    for split_idx, split_label in enumerate(split_names):
        split_dir = out_root / split_label
        split_dir.mkdir(parents=True, exist_ok=True)
        instance_worlds: List[Dict] = []
        instances_per_split = max(1, int(args.instances_per_split))
        for instance_idx in range(instances_per_split):
            instance_dir = split_dir / f"instance_{instance_idx:03d}"
            instance_dir.mkdir(parents=True, exist_ok=True)
            rng = random.Random(int(args.base_seed) * 100_003 + split_idx * 10_007 + instance_idx * 997)
            base_plan_rows = [
                build_plan_row(
                    task_config_name=task_name,
                    split_label=split_label,
                    instance_idx=instance_idx,
                    base_seed=int(args.base_seed),
                    rng=rng,
                    mine_layout_case=str(args.mine_layout_case or ""),
                )
                for task_name in task_names
            ]
            plan_json_path = instance_dir / "worldgen_plan.json"
            generated_groups_root = instance_dir / "generated_task_groups"
            generated_groups_root.mkdir(parents=True, exist_ok=True)
            attempt_records: List[Dict] = []
            selected_entry = None
            last_exc = None
            max_attempts = max(1, int(args.max_instance_attempts))
            for attempt_idx in range(max_attempts):
                print(
                    json.dumps(
                        {
                            "event": "fixed_eval_bank_instance_start",
                            **_progress_payload(
                                split_label=split_label,
                                split_idx=split_idx,
                                total_splits=len(split_names),
                                instance_idx=instance_idx,
                                instances_per_split=instances_per_split,
                                attempt_idx=attempt_idx,
                                max_attempts=max_attempts,
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                plan_rows = [dict(row) for row in base_plan_rows] if attempt_idx == 0 else resample_plan_rows(base_plan_rows, attempt_idx)
                plan_json_path.write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
                generated_dir = None
                try:
                    before_dirs = list_child_dirs(generated_groups_root)
                    run_command(
                        build_worldgen_cmd(
                            plan_json_path=plan_json_path,
                            env_conf_dir=env_conf_dir,
                            out_dir=generated_groups_root,
                            mine_layout_backend=args.mine_layout_backend,
                            mine_anchor_mode=args.mine_anchor_mode,
                        ),
                        env,
                    )
                    generated_dir = detect_generated_dir(generated_groups_root, before_dirs)
                    if args.bake_goals:
                        run_command(
                            build_bake_goal_cmd(
                                task_group=f"fixed_eval_{split_label}_instance_{instance_idx:03d}",
                                task_group_path=generated_dir,
                                tasks=task_names,
                                protocol_name=args.protocol,
                                base_seed=int(args.bake_goal_base_seed),
                                episode_retries=int(args.bake_goal_episode_retries),
                                save_debug_assets=bool(args.bake_goal_save_debug_assets),
                            ),
                            env,
                        )
                    manifest_path = generated_dir / "worldgen_manifest.json"
                    selected_entry = {
                        "instance_idx": int(instance_idx),
                        "split_label": str(split_label),
                        "plan_json": str(plan_json_path.resolve()),
                        "generated_task_group_dir": str(generated_dir.resolve()),
                        "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
                        "plan_rows": plan_rows,
                        "attempt_idx": int(attempt_idx),
                        "attempt_number": int(attempt_idx + 1),
                        "attempt_count": int(attempt_idx + 1),
                    }
                    attempt_records.append(
                        {
                            "attempt_idx": int(attempt_idx),
                            "attempt_number": int(attempt_idx + 1),
                            "status": "success",
                            "plan_json": str(plan_json_path.resolve()),
                            "generated_task_group_dir": str(generated_dir.resolve()),
                            "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "event": "fixed_eval_bank_instance_done",
                                **_progress_payload(
                                    split_label=split_label,
                                    split_idx=split_idx,
                                    total_splits=len(split_names),
                                    instance_idx=instance_idx,
                                    instances_per_split=instances_per_split,
                                    attempt_idx=attempt_idx,
                                    max_attempts=max_attempts,
                                ),
                                "status": "success",
                                "generated_task_group_dir": str(generated_dir.resolve()),
                            },
                            ensure_ascii=False,
                        )
                    )
                    break
                except subprocess.CalledProcessError as exc:
                    last_exc = exc
                    failure_payload = {
                        "instance_idx": int(instance_idx),
                        "split_label": str(split_label),
                        "attempt_idx": int(attempt_idx),
                        "attempt_number": int(attempt_idx + 1),
                        "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                        "plan_json": str(plan_json_path.resolve()),
                        "generated_task_group_dir": str(generated_dir.resolve()) if generated_dir is not None else "",
                        "returncode": int(exc.returncode),
                        "cmd": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                        "error_message": str(exc),
                    }
                    attempt_records.append({**failure_payload, "status": "failed"})
                    write_attempt_failure(instance_dir, attempt_idx, failure_payload)
                    print(
                        json.dumps(
                            {
                                "event": "fixed_eval_bank_instance_retry",
                                **_progress_payload(
                                    split_label=split_label,
                                    split_idx=split_idx,
                                    total_splits=len(split_names),
                                    instance_idx=instance_idx,
                                    instances_per_split=instances_per_split,
                                    attempt_idx=attempt_idx,
                                    max_attempts=max_attempts,
                                ),
                                "status": "resampling_after_failure",
                                "error_message": str(exc),
                            },
                            ensure_ascii=False,
                        )
                    )
            if selected_entry is None:
                raise RuntimeError(
                    f"Failed to build fixed bank instance after {max_attempts} attempts: "
                    f"split={split_label} instance={instance_idx}"
                ) from last_exc
            selected_entry["attempt_records"] = attempt_records
            instance_worlds.append(selected_entry)

        split_manifest = {
            "split_label": str(split_label),
            "tasks": task_names,
            "instances_per_split": int(args.instances_per_split),
            "bank_dir": str(split_dir.resolve()),
            "protocol": str(args.protocol),
            "bake_goals": bool(args.bake_goals),
            "bake_goal_base_seed": int(args.bake_goal_base_seed),
            "instance_worlds": instance_worlds,
        }
        (split_dir / "bank_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        root_manifest["split_dirs"][split_label] = str(split_dir.resolve())
        print(json.dumps({"split_label": split_label, "bank_dir": str(split_dir.resolve())}, ensure_ascii=False))

    (out_root / "root_manifest.json").write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved fixed eval banks to {out_root}")


if __name__ == "__main__":
    main()
