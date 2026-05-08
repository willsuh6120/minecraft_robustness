import argparse
import copy
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_o2_generalization_assets import (
    build_bake_goal_cmd,
    build_worldgen_cmd,
    detect_generated_dir,
    list_child_dirs,
    resample_plan_rows,
    stable_layout_seed,
    write_attempt_failure,
)
from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    canonical_task_key,
    classify_factor_split,
    factor_levels_to_worldgen_suggestions,
    hard_factor_count,
    normalize_factor_levels,
)


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_OUT_DIR = "outputs/evaluate_rocket/mine_h_heading_assets"
DEFAULT_FACTOR_LEVELS = {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}
DEFAULT_LAYOUT_CASES = ["straight"]
DEFAULT_COLLECT_HEADINGS = [0, 90, 120, 150, 180]
DEFAULT_EVAL_HEADINGS = [0, 90, 120, 150, 180]

LAYOUT_CASE_SPECS: Dict[str, Dict[str, str]] = {
    "straight": {"blueprint_id": "straight_tunnel", "target_sign": "any"},
    "offset_chamber_left": {"blueprint_id": "offset_chamber", "target_sign": "neg"},
    "offset_chamber_right": {"blueprint_id": "offset_chamber", "target_sign": "pos"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="mine_coal")
    parser.add_argument("--layout-cases", type=str, default=",".join(DEFAULT_LAYOUT_CASES))
    parser.add_argument("--collect-headings", type=str, default=",".join(str(v) for v in DEFAULT_COLLECT_HEADINGS))
    parser.add_argument("--eval-headings", type=str, default=",".join(str(v) for v in DEFAULT_EVAL_HEADINGS))
    parser.add_argument("--final-headings", type=str, default="")
    parser.add_argument("--eval-instances", type=int, default=0)
    parser.add_argument("--final-instances", type=int, default=0)
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
    parser.add_argument("--factor-levels-json", type=str, default=json.dumps(DEFAULT_FACTOR_LEVELS))
    parser.add_argument("--bake-goals", action="store_true")
    parser.add_argument("--goal-bake-world-mode", type=str, default="same", choices=["same"])
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
    invalid = [case for case in cases if case not in LAYOUT_CASE_SPECS]
    if invalid:
        raise ValueError(f"Unsupported layout cases: {invalid}")
    return cases


def parse_heading_values(raw: str, *, default: List[int]) -> List[int]:
    text = str(raw).strip()
    if not text:
        values = list(default)
    else:
        values = []
        for item in text.split(","):
            token = item.strip()
            if not token:
                continue
            value = int(token)
            if value < 0:
                raise ValueError(f"Heading values must be non-negative degrees: {value}")
            values.append(value)
    deduped: List[int] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(int(value))
    if not deduped:
        raise ValueError("At least one heading must be provided.")
    return deduped


def parse_factor_levels(raw: str, task_name: str) -> Dict[str, int]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--factor-levels-json must decode to a JSON object.")
    return normalize_factor_levels(task_name, data)


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    print("[interaction-prepare-mine-h-heading-assets] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def heading_variant_id(layout_case: str, heading_deg: int, sign_label: str) -> str:
    return f"{layout_case}_H{int(heading_deg):03d}" if sign_label == "" else f"{layout_case}_H{int(heading_deg):03d}_{sign_label}"


def build_heading_variants(layout_cases: List[str], headings: List[int]) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    for layout_case in layout_cases:
        case_spec = LAYOUT_CASE_SPECS[layout_case]
        for heading_deg in headings:
            if int(heading_deg) == 0:
                variant_id = heading_variant_id(layout_case, int(heading_deg), "")
                variants.append(
                    {
                        "variant_id": variant_id,
                        "layout_case": str(layout_case),
                        "blueprint_id": str(case_spec["blueprint_id"]),
                        "target_sign": str(case_spec["target_sign"]),
                        "heading_deg": int(heading_deg),
                        "heading_sign": 0,
                        "heading_sign_label": "zero",
                        "heading_offset_deg": 0.0,
                    }
                )
                continue
            if int(heading_deg) == 180:
                variant_id = heading_variant_id(layout_case, int(heading_deg), "")
                variants.append(
                    {
                        "variant_id": variant_id,
                        "layout_case": str(layout_case),
                        "blueprint_id": str(case_spec["blueprint_id"]),
                        "target_sign": str(case_spec["target_sign"]),
                        "heading_deg": int(heading_deg),
                        "heading_sign": 0,
                        "heading_sign_label": "opp",
                        "spawn_yaw_deg": 180.0,
                    }
                )
                continue
            for heading_sign, sign_label in ((1, "pos"), (-1, "neg")):
                variant_id = heading_variant_id(layout_case, int(heading_deg), sign_label)
                variants.append(
                    {
                        "variant_id": variant_id,
                        "layout_case": str(layout_case),
                        "blueprint_id": str(case_spec["blueprint_id"]),
                        "target_sign": str(case_spec["target_sign"]),
                        "heading_deg": int(heading_deg),
                        "heading_sign": int(heading_sign),
                        "heading_sign_label": str(sign_label),
                        "heading_offset_deg": float(int(heading_sign) * int(heading_deg)),
                    }
                )
    return variants


def build_plan_row(
    *,
    task_name: str,
    factor_levels: Dict[str, int],
    variant: Dict[str, object],
    bank_label: str,
    instance_idx: int,
    base_seed: int,
) -> Dict:
    task_key = canonical_task_key(task_name)
    normalized_levels = normalize_factor_levels(task_key, factor_levels)
    split_label = classify_factor_split(task_key, normalized_levels)
    suggestions = {
        **factor_levels_to_worldgen_suggestions(task_key, normalized_levels),
        "mine_blueprint_id": str(variant["blueprint_id"]),
        "mine_heading_variant_id": str(variant["variant_id"]),
        "mine_spawn_pitch_deg": 0.0,
        "notes": (
            f"H-only heading recovery row for {bank_label}, variant={variant['variant_id']}. "
            "No occluder or path obstacle is injected; the only intended perturbation is initial heading."
        ),
    }
    target_sign = str(variant.get("target_sign") or "any")
    if target_sign != "any":
        suggestions["mine_target_sign"] = target_sign
    if "spawn_yaw_deg" in variant:
        suggestions["mine_spawn_yaw_deg"] = float(variant["spawn_yaw_deg"])
    elif "heading_offset_deg" in variant:
        suggestions["mine_heading_offset_deg"] = float(variant["heading_offset_deg"])
    else:
        raise ValueError(f"Invalid heading variant: {variant}")
    return {
        "task_config_name": str(task_name),
        "task_key": str(task_key),
        "primary_failure_mode_majority": f"h_heading_{bank_label}",
        "primary_factors_majority": ["H"],
        "severity_majority": 0,
        "factor_levels": normalized_levels,
        "layout_seed": stable_layout_seed(
            task_config_name=task_name,
            bank_label=bank_label,
            instance_idx=instance_idx,
            base_seed=base_seed,
            factor_levels=normalized_levels,
            layout_case=str(variant["variant_id"]),
        ),
        "template_index": int(instance_idx),
        "requested_split_label": str(split_label),
        "computed_split_label": str(split_label),
        "hard_factor_count": int(hard_factor_count(task_key, normalized_levels)),
        "requested_layout_case": str(variant["layout_case"]),
        "trainable_with_rl_majority": True,
        "world_generation_suggestions": suggestions,
    }


def build_collect_plan_rows(
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    collect_variants: List[Dict[str, object]],
) -> List[Dict]:
    variant = collect_variants[0]
    return [
        build_plan_row(
            task_name=task_name,
            factor_levels=factor_levels,
            variant=variant,
            bank_label="collect_plan",
            instance_idx=0,
            base_seed=base_seed,
        )
        for task_name in task_names
    ]


def build_collect_block_plan_rows(
    *,
    task_names: List[str],
    factor_levels: Dict[str, int],
    base_seed: int,
    collect_variants: List[Dict[str, object]],
    block_index: int,
) -> List[Dict]:
    variant = collect_variants[int(block_index) - 1]
    return [
        build_plan_row(
            task_name=task_name,
            factor_levels=factor_levels,
            variant=variant,
            bank_label=f"collect_block_{int(block_index):02d}",
            instance_idx=0,
            base_seed=base_seed,
        )
        for task_name in task_names
    ]


def resolve_variant_sequence(variants: List[Dict[str, object]], instances: int) -> List[Dict[str, object]]:
    if not variants:
        raise ValueError("No variants provided.")
    total_instances = len(variants) if int(instances) <= 0 else max(1, int(instances))
    return [copy.deepcopy(variants[idx % len(variants)]) for idx in range(total_instances)]


def generate_bank(
    *,
    bank_dir: Path,
    bank_label: str,
    instances: int,
    task_names: List[str],
    factor_levels: Dict[str, int],
    variants: List[Dict[str, object]],
    args,
    env: Dict[str, str],
) -> None:
    bank_dir.mkdir(parents=True, exist_ok=True)
    env_conf_dir = Path(args.env_conf_dir)
    instance_worlds: List[Dict] = []
    max_attempts = max(1, int(args.max_instance_attempts))
    worker_count = max(1, int(args.bank_workers))
    variant_sequence = resolve_variant_sequence(variants, instances)

    def build_instance(instance_idx: int) -> Dict:
        variant = variant_sequence[int(instance_idx)]
        instance_dir = bank_dir / f"instance_{int(instance_idx):03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        base_plan_rows = [
            build_plan_row(
                task_name=task_name,
                factor_levels=factor_levels,
                variant=variant,
                bank_label=bank_label,
                instance_idx=instance_idx,
                base_seed=int(args.base_seed),
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
            plan_rows = copy.deepcopy(base_plan_rows) if attempt_idx == 0 else resample_plan_rows(base_plan_rows, attempt_idx)
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
                    "layout_case": str(variant["layout_case"]),
                    "heading_variant_id": str(variant["variant_id"]),
                    "heading_deg": int(variant["heading_deg"]),
                    "heading_sign": int(variant["heading_sign"]),
                    "heading_sign_label": str(variant["heading_sign_label"]),
                    "heading_offset_deg": variant.get("heading_offset_deg"),
                    "spawn_yaw_deg": variant.get("spawn_yaw_deg"),
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
                        "layout_case": str(variant["layout_case"]),
                        "heading_variant_id": str(variant["variant_id"]),
                        "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                    }
                )
                break
            except Exception as exc:
                last_exc = exc
                failure_payload = {
                    "instance_idx": int(instance_idx),
                    "split_label": str(bank_label),
                    "layout_case": str(variant["layout_case"]),
                    "heading_variant_id": str(variant["variant_id"]),
                    "attempt_idx": int(attempt_idx),
                    "attempt_number": int(attempt_idx + 1),
                    "layout_seeds": [int(row.get("layout_seed", 0) or 0) for row in plan_rows],
                    "plan_json": str(plan_json_path.resolve()),
                    "generated_task_group_dir": str(generated_dir.resolve()) if generated_dir is not None else "",
                    "error_message": str(exc),
                }
                attempt_records.append({**failure_payload, "status": "failed"})
                write_attempt_failure(instance_dir, attempt_idx, failure_payload)
        if selected_entry is None:
            raise RuntimeError(
                f"Failed to build H heading bank instance after {max_attempts} attempts: "
                f"bank={bank_label} instance={instance_idx}"
            ) from last_exc
        selected_entry["attempt_records"] = attempt_records
        return selected_entry

    total_instances = len(variant_sequence)
    if worker_count == 1:
        for instance_idx in range(total_instances):
            instance_worlds.append(build_instance(instance_idx))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(build_instance, instance_idx): int(instance_idx) for instance_idx in range(total_instances)}
            results: Dict[int, Dict] = {}
            for future in as_completed(future_map):
                instance_idx = future_map[future]
                results[instance_idx] = future.result()
            for instance_idx in range(total_instances):
                instance_worlds.append(results[instance_idx])

    bank_manifest = {
        "split_label": str(bank_label),
        "tasks": task_names,
        "instances_per_split": int(total_instances),
        "bank_dir": str(bank_dir.resolve()),
        "bank_workers": int(worker_count),
        "protocol": str(args.protocol),
        "bake_goals": bool(args.bake_goals),
        "goal_bake_world_mode": str(args.goal_bake_world_mode),
        "bake_goal_base_seed": int(args.bake_goal_base_seed),
        "factor_levels": factor_levels,
        "layout_cases": sorted({str(item["layout_case"]) for item in variants}),
        "heading_variants": variants,
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
    collect_headings = parse_heading_values(args.collect_headings, default=DEFAULT_COLLECT_HEADINGS)
    eval_headings = parse_heading_values(args.eval_headings, default=DEFAULT_EVAL_HEADINGS)
    final_headings = parse_heading_values(args.final_headings, default=eval_headings)
    factor_levels = parse_factor_levels(args.factor_levels_json, task_names[0])

    collect_variants = build_heading_variants(layout_cases, collect_headings)
    eval_variants = build_heading_variants(layout_cases, eval_headings)
    final_variants = build_heading_variants(layout_cases, final_headings)

    out_root = (Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MINESTUDIO_DIR", str(Path.home() / ".minestudio"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    collect_plan_rows = build_collect_plan_rows(task_names, factor_levels, int(args.base_seed), collect_variants)
    collect_plan_path = out_root / "collect_plan.json"
    collect_plan_path.write_text(json.dumps(collect_plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    collect_block_plan_dir = out_root / "collect_block_plans"
    collect_block_plan_dir.mkdir(parents=True, exist_ok=True)
    collect_block_plan_paths: List[str] = []
    collect_block_variant_map: List[Dict[str, object]] = []
    for block_index in range(1, len(collect_variants) + 1):
        block_rows = build_collect_block_plan_rows(
            task_names=task_names,
            factor_levels=factor_levels,
            base_seed=int(args.base_seed),
            collect_variants=collect_variants,
            block_index=block_index,
        )
        block_plan_path = collect_block_plan_dir / f"block_{int(block_index):03d}_collect_plan.json"
        block_plan_path.write_text(json.dumps(block_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        collect_block_plan_paths.append(str(block_plan_path.resolve()))
        collect_block_variant_map.append(
            {
                "block_index": int(block_index),
                "variant_id": str(collect_variants[int(block_index) - 1]["variant_id"]),
                "layout_case": str(collect_variants[int(block_index) - 1]["layout_case"]),
                "heading_deg": int(collect_variants[int(block_index) - 1]["heading_deg"]),
                "heading_sign_label": str(collect_variants[int(block_index) - 1]["heading_sign_label"]),
            }
        )

    eval_bank_dir = out_root / "eval_bank"
    final_eval_bank_dir = out_root / "final_eval_bank"
    generate_bank(
        bank_dir=eval_bank_dir,
        bank_label="eval_bank",
        instances=int(args.eval_instances),
        task_names=task_names,
        factor_levels=factor_levels,
        variants=eval_variants,
        args=args,
        env=env,
    )
    generate_bank(
        bank_dir=final_eval_bank_dir,
        bank_label="final_eval_bank",
        instances=int(args.final_instances),
        task_names=task_names,
        factor_levels=factor_levels,
        variants=final_variants,
        args=args,
        env=env,
    )

    manifest = {
        "asset_type": "mine_h_heading_offset_assets",
        "root_dir": str(out_root.resolve()),
        "tasks": task_names,
        "factor_levels": factor_levels,
        "layout_cases": layout_cases,
        "collect_headings": collect_headings,
        "eval_headings": eval_headings,
        "final_headings": final_headings,
        "collect_variants": collect_variants,
        "collect_plan_path": str(collect_plan_path.resolve()),
        "collect_block_plan_dir": str(collect_block_plan_dir.resolve()),
        "collect_block_plan_paths": collect_block_plan_paths,
        "collect_block_variant_map": collect_block_variant_map,
        "eval_bank_dir": str(eval_bank_dir.resolve()),
        "final_eval_bank_dir": str(final_eval_bank_dir.resolve()),
        "eval_instances": int(len(resolve_variant_sequence(eval_variants, int(args.eval_instances)))),
        "final_instances": int(len(resolve_variant_sequence(final_variants, int(args.final_instances)))),
        "bake_goals": bool(args.bake_goals),
        "goal_bake_world_mode": str(args.goal_bake_world_mode),
        "config": {
            "env_conf_dir": str(Path(args.env_conf_dir).resolve()),
            "mine_layout_backend": str(args.mine_layout_backend),
            "mine_anchor_mode": str(args.mine_anchor_mode),
            "protocol": str(args.protocol),
            "base_seed": int(args.base_seed),
        },
    }
    (out_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
