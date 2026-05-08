import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
    normalize_factor_levels,
)


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_OUT_DIR = "outputs/evaluate_rocket/mine_o2_generalization_assets"
DEFAULT_FACTOR_LEVELS = {"R": 0, "H": 0, "O": 2, "C": 0, "P": 0, "A": 0}
DEFAULT_LAYOUT_CASES = [
    "straight",
    "offset_chamber_left",
    "offset_chamber_right",
    "turn_left",
    "turn_right",
    "side_alcove_left",
    "side_alcove_right",
]

MINE_LAYOUT_CASE_SPECS: Dict[str, Dict[str, str]] = {
    "straight": {"blueprint_id": "straight_tunnel", "target_sign": "any"},
    "offset_chamber_left": {"blueprint_id": "offset_chamber", "target_sign": "neg"},
    "offset_chamber_right": {"blueprint_id": "offset_chamber", "target_sign": "pos"},
    "turn_left": {"blueprint_id": "turn_left", "target_sign": "neg"},
    "turn_right": {"blueprint_id": "turn_right", "target_sign": "pos"},
    "side_alcove_left": {"blueprint_id": "side_alcove_left", "target_sign": "neg"},
    "side_alcove_right": {"blueprint_id": "side_alcove_right", "target_sign": "pos"},
}

DEFAULT_LEAF_MATERIAL = "minecraft:oak_leaves[persistent=true]"
FINAL_BANK_LEAF_MATERIAL = "minecraft:azalea_leaves[persistent=true]"

STRAIGHT_O2_BANK_PATTERNS: List[Dict[str, object]] = [
    {"variant_id": "pattern_01", "top_row": ["glass", "glass", "glass"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_02", "top_row": ["leaves", "glass", "glass"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_03", "top_row": ["glass", "leaves", "glass"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_04", "top_row": ["glass", "glass", "leaves"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_05", "top_row": ["glass", "glass", "glass"], "bottom_row": ["leaves", "glass", "glass"]},
    {"variant_id": "pattern_06", "top_row": ["glass", "glass", "glass"], "bottom_row": ["glass", "glass", "leaves"]},
    {"variant_id": "pattern_07", "top_row": ["leaves", "leaves", "glass"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_08", "top_row": ["leaves", "glass", "leaves"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_09", "top_row": ["leaves", "glass", "glass"], "bottom_row": ["leaves", "glass", "glass"]},
    {"variant_id": "pattern_10", "top_row": ["leaves", "glass", "glass"], "bottom_row": ["glass", "glass", "leaves"]},
    {"variant_id": "pattern_11", "top_row": ["glass", "leaves", "leaves"], "bottom_row": ["glass", "glass", "glass"]},
    {"variant_id": "pattern_12", "top_row": ["glass", "leaves", "glass"], "bottom_row": ["leaves", "glass", "glass"]},
    {"variant_id": "pattern_13", "top_row": ["glass", "leaves", "glass"], "bottom_row": ["glass", "glass", "leaves"]},
    {"variant_id": "pattern_14", "top_row": ["glass", "glass", "leaves"], "bottom_row": ["leaves", "glass", "glass"]},
    {"variant_id": "pattern_15", "top_row": ["glass", "glass", "leaves"], "bottom_row": ["glass", "glass", "leaves"]},
    {"variant_id": "pattern_16", "top_row": ["glass", "glass", "glass"], "bottom_row": ["leaves", "glass", "leaves"]},
]

STRAIGHT_O2_COLLECT_BLOCK_PATTERNS: List[Dict[str, object]] = [
    {"block_index": 1, "variant_id": "collect_block_01_all_glass", "top_row": ["glass", "glass", "glass"], "bottom_row": ["glass", "glass", "glass"]},
    {"block_index": 2, "variant_id": "collect_block_02_g_l_g__l_g_l", "top_row": ["glass", "leaves", "glass"], "bottom_row": ["leaves", "glass", "leaves"]},
    {"block_index": 3, "variant_id": "collect_block_03_l_l_l__g_g_g", "top_row": ["leaves", "leaves", "leaves"], "bottom_row": ["glass", "glass", "glass"]},
    {"block_index": 4, "variant_id": "collect_block_04_l_l_g__g_g_l", "top_row": ["leaves", "leaves", "glass"], "bottom_row": ["glass", "glass", "leaves"]},
    {"block_index": 5, "variant_id": "collect_block_05_l_g_l__g_g_l", "top_row": ["leaves", "glass", "leaves"], "bottom_row": ["glass", "glass", "leaves"]},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--eval-instances", type=int, default=16)
    parser.add_argument("--final-instances", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=1)
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
    parser.add_argument("--layout-cases", type=str, default=",".join(DEFAULT_LAYOUT_CASES))
    parser.add_argument("--factor-levels-json", type=str, default=json.dumps(DEFAULT_FACTOR_LEVELS))
    parser.add_argument("--bake-goals", action="store_true")
    parser.add_argument("--bake-goal-base-seed", type=int, default=1)
    parser.add_argument("--bake-goal-episode-retries", type=int, default=2)
    parser.add_argument("--bake-goal-save-debug-assets", action="store_true")
    parser.add_argument("--max-instance-attempts", type=int, default=6)
    parser.add_argument("--bank-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_task_names(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_layout_cases(raw: str) -> List[str]:
    cases = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not cases:
        raise ValueError("No layout cases provided.")
    invalid = [case for case in cases if case not in MINE_LAYOUT_CASE_SPECS]
    if invalid:
        raise ValueError(f"Unsupported layout cases: {invalid}")
    return cases


def parse_factor_levels(raw: str, task_name: str) -> Dict[str, int]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--factor-levels-json must decode to a JSON object.")
    return normalize_factor_levels(task_name, data)


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    print("[interaction-prepare-mine-o2-generalization-assets] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_worldgen_cmd(
    plan_json_path: Path,
    env_conf_dir: Path,
    out_dir: Path,
    mine_layout_backend: str,
    mine_anchor_mode: str,
    path_progress_reward_per_zone: float | None = None,
) -> List[str]:
    cmd = [
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
    if path_progress_reward_per_zone is not None:
        cmd.extend(["--path-progress-reward-per-zone", str(float(path_progress_reward_per_zone))])
    return cmd


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


def stable_layout_seed(
    task_config_name: str,
    bank_label: str,
    instance_idx: int,
    base_seed: int,
    factor_levels: Dict[str, int],
    layout_case: str,
) -> int:
    payload = {
        "task_config_name": str(task_config_name),
        "bank_label": str(bank_label),
        "instance_idx": int(instance_idx),
        "base_seed": int(base_seed),
        "factor_levels": factor_levels,
        "layout_case": str(layout_case),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


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


def _supports_straight_o2_exact_patterns(
    *,
    task_key: str,
    layout_case: str,
    factor_levels: Dict[str, int],
    blueprint_id: str,
) -> bool:
    if str(layout_case) != "straight":
        return False
    normalized = normalize_factor_levels(task_key, factor_levels)
    if any(int(normalized.get(code, 0)) != 0 for code in ("R", "H", "C", "P", "A")):
        return False
    if int(normalized.get("O", 0)) != 2:
        return False
    return str(blueprint_id) == "straight_tunnel"


def attach_exact_occluder_pattern(
    *,
    suggestions: Dict[str, object],
    pattern_spec: Dict[str, object],
    leaf_material: str,
) -> Dict[str, object]:
    merged = dict(suggestions)
    merged.update(
        {
            "mine_occluder_variant_id": str(pattern_spec["variant_id"]),
            "mine_occluder_front_depth": 1,
            "mine_occluder_side_span": 1,
            "mine_occluder_top_row": list(pattern_spec["top_row"]),
            "mine_occluder_bottom_row": list(pattern_spec["bottom_row"]),
            "mine_occluder_leaf_material": str(leaf_material),
        }
    )
    return merged


def write_attempt_failure(instance_dir: Path, attempt_idx: int, payload: Dict) -> None:
    failure_path = instance_dir / f"attempt_{int(attempt_idx) + 1:02d}_failure.json"
    failure_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_collect_plan_rows(
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    layout_cases: List[str],
) -> List[Dict]:
    rows: List[Dict] = []
    locked_case = layout_cases[0] if len(layout_cases) == 1 else ""
    case_spec = MINE_LAYOUT_CASE_SPECS.get(locked_case, {}) if locked_case else {}
    for task_name in task_names:
        task_key = canonical_task_key(task_name)
        normalized_levels = normalize_factor_levels(task_key, factor_levels)
        split_label = classify_factor_split(task_key, normalized_levels)
        blueprint_id = str(case_spec.get("blueprint_id", ""))
        suggestions = {
            **factor_levels_to_worldgen_suggestions(task_key, normalized_levels),
            "notes": (
                "O2 collect plan for generalization training. "
                "Layout family is fixed when exactly one layout_case is requested; "
                "layout_seed resampling then changes only unseen world instances within that family."
            ),
        }
        if case_spec:
            suggestions["mine_blueprint_id"] = blueprint_id
            target_sign = str(case_spec.get("target_sign", "any")).strip()
            if target_sign and target_sign != "any":
                suggestions["mine_target_sign"] = target_sign
        if _supports_straight_o2_exact_patterns(
            task_key=task_key,
            layout_case=locked_case,
            factor_levels=normalized_levels,
            blueprint_id=blueprint_id,
        ):
            suggestions = attach_exact_occluder_pattern(
                suggestions=suggestions,
                pattern_spec=STRAIGHT_O2_COLLECT_BLOCK_PATTERNS[0],
                leaf_material=DEFAULT_LEAF_MATERIAL,
            )
        rows.append(
            {
                "task_config_name": str(task_name),
                "task_key": str(task_key),
                "primary_failure_mode_majority": "o2_collect_generalization",
                "primary_factors_majority": [code for code, value in normalized_levels.items() if int(value) > 0],
                "severity_majority": max([int(value) for value in normalized_levels.values()] or [0]),
                "factor_levels": normalized_levels,
                "layout_seed": stable_layout_seed(
                    task_config_name=task_name,
                    bank_label="collect_plan",
                    instance_idx=0,
                    base_seed=base_seed,
                    factor_levels=normalized_levels,
                    layout_case=locked_case or "collect_plan",
                ),
                "template_index": 0,
                "requested_split_label": str(split_label),
                "computed_split_label": str(split_label),
                "hard_factor_count": int(hard_factor_count(task_key, normalized_levels)),
                "trainable_with_rl_majority": True,
                "requested_layout_case": str(locked_case) if locked_case else "",
                "world_generation_suggestions": suggestions,
            }
        )
    return rows


def build_collect_block_plan_rows(
    *,
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    layout_cases: List[str],
    block_index: int,
) -> List[Dict]:
    if len(layout_cases) != 1 or str(layout_cases[0]) != "straight":
        raise ValueError("Collect block plans currently require --layout-cases straight.")
    pattern_spec = STRAIGHT_O2_COLLECT_BLOCK_PATTERNS[int(block_index) - 1]
    rows: List[Dict] = []
    case_spec = MINE_LAYOUT_CASE_SPECS["straight"]
    for task_name in task_names:
        task_key = canonical_task_key(task_name)
        normalized_levels = normalize_factor_levels(task_key, factor_levels)
        split_label = classify_factor_split(task_key, normalized_levels)
        suggestions = {
            **factor_levels_to_worldgen_suggestions(task_key, normalized_levels),
            "mine_blueprint_id": str(case_spec["blueprint_id"]),
            "notes": (
                f"Straight O2 collect block {int(block_index)}. "
                f"Exact occluder pattern {pattern_spec['variant_id']}."
            ),
        }
        suggestions = attach_exact_occluder_pattern(
            suggestions=suggestions,
            pattern_spec=pattern_spec,
            leaf_material=DEFAULT_LEAF_MATERIAL,
        )
        rows.append(
            {
                "task_config_name": str(task_name),
                "task_key": str(task_key),
                "primary_failure_mode_majority": f"o2_collect_block_{int(block_index):02d}",
                "primary_factors_majority": [code for code, value in normalized_levels.items() if int(value) > 0],
                "severity_majority": max([int(value) for value in normalized_levels.values()] or [0]),
                "factor_levels": normalized_levels,
                "layout_seed": stable_layout_seed(
                    task_config_name=task_name,
                    bank_label=f"collect_block_{int(block_index):02d}",
                    instance_idx=0,
                    base_seed=base_seed,
                    factor_levels=normalized_levels,
                    layout_case="straight",
                ),
                "template_index": 0,
                "requested_split_label": str(split_label),
                "computed_split_label": str(split_label),
                "hard_factor_count": int(hard_factor_count(task_key, normalized_levels)),
                "requested_layout_case": "straight",
                "trainable_with_rl_majority": True,
                "world_generation_suggestions": suggestions,
            }
        )
    return rows


def build_bank_plan_row(
    *,
    task_name: str,
    factor_levels: Dict[str, int],
    layout_case: str,
    instance_idx: int,
    bank_label: str,
    base_seed: int,
    leaf_material: str,
) -> Dict:
    task_key = canonical_task_key(task_name)
    normalized_levels = normalize_factor_levels(task_key, factor_levels)
    split_label = classify_factor_split(task_key, normalized_levels)
    case_spec = MINE_LAYOUT_CASE_SPECS[layout_case]
    suggestions = {
        **factor_levels_to_worldgen_suggestions(task_key, normalized_levels),
        "mine_blueprint_id": str(case_spec["blueprint_id"]),
        "notes": (
            f"O2 fixed bank row for {bank_label}, layout_case={layout_case}, instance={int(instance_idx)}."
        ),
    }
    if _supports_straight_o2_exact_patterns(
        task_key=task_key,
        layout_case=layout_case,
        factor_levels=normalized_levels,
        blueprint_id=str(case_spec["blueprint_id"]),
    ):
        pattern_spec = STRAIGHT_O2_BANK_PATTERNS[int(instance_idx) % len(STRAIGHT_O2_BANK_PATTERNS)]
        suggestions = attach_exact_occluder_pattern(
            suggestions=suggestions,
            pattern_spec=pattern_spec,
            leaf_material=leaf_material,
        )
    target_sign = str(case_spec.get("target_sign", "any"))
    if target_sign and target_sign != "any":
        suggestions["mine_target_sign"] = target_sign
    return {
        "task_config_name": str(task_name),
        "task_key": str(task_key),
        "primary_failure_mode_majority": f"o2_fixed_bank_{bank_label}",
        "primary_factors_majority": [code for code, value in normalized_levels.items() if int(value) > 0],
        "severity_majority": max([int(value) for value in normalized_levels.values()] or [0]),
        "factor_levels": normalized_levels,
        "layout_seed": stable_layout_seed(
            task_config_name=task_name,
            bank_label=bank_label,
            instance_idx=instance_idx,
            base_seed=base_seed,
            factor_levels=normalized_levels,
            layout_case=layout_case,
        ),
        "template_index": int(instance_idx),
        "requested_split_label": str(split_label),
        "computed_split_label": str(split_label),
        "hard_factor_count": int(hard_factor_count(task_key, normalized_levels)),
        "requested_layout_case": str(layout_case),
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": suggestions,
    }


def generate_bank(
    *,
    bank_dir: Path,
    bank_label: str,
    instances: int,
    task_names: List[str],
    factor_levels: Dict[str, int],
    layout_cases: List[str],
    leaf_material: str,
    args,
    env: Dict[str, str],
) -> None:
    bank_dir.mkdir(parents=True, exist_ok=True)
    env_conf_dir = Path(args.env_conf_dir)
    instance_worlds: List[Dict] = []
    max_attempts = max(1, int(args.max_instance_attempts))
    worker_count = max(1, int(getattr(args, "bank_workers", 1) or 1))

    def build_instance(instance_idx: int) -> Dict:
        layout_case = layout_cases[int(instance_idx) % len(layout_cases)]
        instance_dir = bank_dir / f"instance_{instance_idx:03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        base_plan_rows = [
            build_bank_plan_row(
                task_name=task_name,
                factor_levels=factor_levels,
                layout_case=layout_case,
                instance_idx=instance_idx,
                bank_label=bank_label,
                base_seed=int(args.base_seed),
                leaf_material=leaf_material,
            )
            for task_name in task_names
        ]
        plan_json_path = instance_dir / "worldgen_plan.json"
        generated_groups_root = instance_dir / "generated_task_groups"
        generated_groups_root.mkdir(parents=True, exist_ok=True)
        attempt_records: List[Dict] = []
        selected_entry = None
        last_exc = None
        for attempt_idx in range(max_attempts):
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
                            task_group=f"{bank_label}_instance_{instance_idx:03d}",
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
                    "split_label": str(bank_label),
                    "layout_case": str(layout_case),
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
                        "layout_case": str(layout_case),
                        "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                    }
                )
                break
            except subprocess.CalledProcessError as exc:
                last_exc = exc
                failure_payload = {
                    "instance_idx": int(instance_idx),
                    "split_label": str(bank_label),
                    "layout_case": str(layout_case),
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
        if selected_entry is None:
            raise RuntimeError(
                f"Failed to build bank instance after {max_attempts} attempts: bank={bank_label} instance={instance_idx}"
            ) from last_exc
        selected_entry["attempt_records"] = attempt_records
        return selected_entry

    total_instances = max(1, int(instances))
    if worker_count == 1:
        for instance_idx in range(total_instances):
            instance_worlds.append(build_instance(instance_idx))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(build_instance, instance_idx): int(instance_idx)
                for instance_idx in range(total_instances)
            }
            results: Dict[int, Dict] = {}
            for future in as_completed(future_map):
                instance_idx = future_map[future]
                results[instance_idx] = future.result()
            for instance_idx in range(total_instances):
                instance_worlds.append(results[instance_idx])

    bank_manifest = {
        "split_label": str(bank_label),
        "tasks": task_names,
        "instances_per_split": int(instances),
        "bank_dir": str(bank_dir.resolve()),
        "bank_workers": int(worker_count),
        "protocol": str(args.protocol),
        "bake_goals": bool(args.bake_goals),
        "bake_goal_base_seed": int(args.bake_goal_base_seed),
        "factor_levels": factor_levels,
        "layout_cases": list(layout_cases),
        "instance_worlds": instance_worlds,
    }
    (bank_dir / "bank_manifest.json").write_text(
        json.dumps(bank_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    task_names = parse_task_names(args.tasks)
    if not task_names:
        raise ValueError("No tasks provided.")
    layout_cases = parse_layout_cases(args.layout_cases)
    factor_levels = parse_factor_levels(args.factor_levels_json, task_names[0])
    out_root = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    collect_plan_rows = build_collect_plan_rows(task_names, factor_levels, int(args.base_seed), layout_cases)
    collect_plan_path = out_root / "collect_plan.json"
    collect_plan_path.write_text(json.dumps(collect_plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    collect_block_plan_dir = out_root / "collect_block_plans"
    collect_block_plan_dir.mkdir(parents=True, exist_ok=True)
    collect_block_plan_paths: List[str] = []
    for block_index in range(1, len(STRAIGHT_O2_COLLECT_BLOCK_PATTERNS) + 1):
        block_rows = build_collect_block_plan_rows(
            task_names=task_names,
            factor_levels=factor_levels,
            base_seed=int(args.base_seed),
            layout_cases=layout_cases,
            block_index=block_index,
        )
        block_plan_path = collect_block_plan_dir / f"block_{int(block_index):03d}_collect_plan.json"
        block_plan_path.write_text(json.dumps(block_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        collect_block_plan_paths.append(str(block_plan_path.resolve()))

    eval_bank_dir = out_root / "eval_bank"
    final_eval_bank_dir = out_root / "final_eval_bank"
    generate_bank(
        bank_dir=eval_bank_dir,
        bank_label="eval_bank",
        instances=int(args.eval_instances),
        task_names=task_names,
        factor_levels=factor_levels,
        layout_cases=layout_cases,
        leaf_material=DEFAULT_LEAF_MATERIAL,
        args=args,
        env=env,
    )
    generate_bank(
        bank_dir=final_eval_bank_dir,
        bank_label="final_eval_bank",
        instances=int(args.final_instances),
        task_names=task_names,
        factor_levels=factor_levels,
        layout_cases=layout_cases,
        leaf_material=FINAL_BANK_LEAF_MATERIAL,
        args=args,
        env=env,
    )

    manifest = {
        "root_dir": str(out_root.resolve()),
        "tasks": task_names,
        "factor_levels": factor_levels,
        "layout_cases": layout_cases,
        "collect_plan_json": str(collect_plan_path.resolve()),
        "collect_block_plan_dir": str(collect_block_plan_dir.resolve()),
        "collect_block_plan_paths": collect_block_plan_paths,
        "eval_bank_dir": str(eval_bank_dir.resolve()),
        "final_eval_bank_dir": str(final_eval_bank_dir.resolve()),
        "eval_instances": int(args.eval_instances),
        "final_instances": int(args.final_instances),
        "bake_goals": bool(args.bake_goals),
        "eval_leaf_material": DEFAULT_LEAF_MATERIAL,
        "final_eval_leaf_material": FINAL_BANK_LEAF_MATERIAL,
        "config": {
            "env_conf_dir": str(Path(args.env_conf_dir).resolve()),
            "mine_layout_backend": str(args.mine_layout_backend),
            "mine_anchor_mode": str(args.mine_anchor_mode),
            "protocol": str(args.protocol),
            "base_seed": int(args.base_seed),
        },
    }
    (out_root / "assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
