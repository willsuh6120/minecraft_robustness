#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_VARIANTS = [
    "left_lane_z1_t6",
    "right_lane_z2_t6",
    "left_chicane_t6",
    "right_s_curve_t6",
    "left_funnel_t6",
    "right_gate_entrance_t6",
    "left_outer_detour_t6",
    "right_narrow_door_t6",
]

HELDOUT_VARIANTS = [
    "right_lane_z1_t6",
    "left_lane_z2_t6",
    "right_chicane_t6",
    "left_s_curve_t6",
    "right_funnel_t6",
    "left_gate_entrance_t6",
    "right_outer_detour_t6",
    "left_narrow_door_t6",
]

ROUND_ROBIN_SCHEDULE = [
    "left_lane_z1_t6",
    "right_lane_z2_t6",
    "left_outer_detour_t6",
    "left_chicane_t6",
    "left_funnel_t6",
    "right_s_curve_t6",
    "right_narrow_door_t6",
    "right_gate_entrance_t6",
    "left_lane_z1_t6",
    "right_lane_z2_t6",
    "left_outer_detour_t6",
    "left_chicane_t6",
    "left_funnel_t6",
    "right_s_curve_t6",
    "right_narrow_door_t6",
    "right_gate_entrance_t6",
]

FAMILY_VARIANTS = {
    "lane_z1": ["left_lane_z1_t6", "right_lane_z1_t6"],
    "lane_z2": ["left_lane_z2_t6", "right_lane_z2_t6"],
    "chicane": ["left_chicane_t6", "right_chicane_t6"],
    "s_curve": ["left_s_curve_t6", "right_s_curve_t6"],
    "funnel": ["left_funnel_t6", "right_funnel_t6"],
    "gate_entrance": ["left_gate_entrance_t6", "right_gate_entrance_t6"],
    "outer_detour": ["left_outer_detour_t6", "right_outer_detour_t6"],
    "narrow_door": ["left_narrow_door_t6", "right_narrow_door_t6"],
}

# Difficulty-aware family CV proposal:
# - Keep both left/right directions of each family together.
# - Use 6 families for train, 1 for val, 1 for test.
# - Pair a relatively easier family with a relatively harder family across folds.
CV_FOLDS = {
    "fold_a": {
        "val_families": ["lane_z2"],
        "test_families": ["gate_entrance"],
    },
    "fold_b": {
        "val_families": ["chicane"],
        "test_families": ["narrow_door"],
    },
    "fold_c": {
        "val_families": ["funnel"],
        "test_families": ["s_curve"],
    },
    "fold_d": {
        "val_families": ["lane_z1"],
        "test_families": ["outer_detour"],
    },
}


def load_manifest(asset_dir: Path) -> Dict:
    manifest_path = asset_dir / "eval_bank" / "bank_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_manifest_path(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_summary(summary_path: Path) -> Dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def normalize_variant_rows(summary_payload: Dict) -> List[Dict]:
    rows = list(summary_payload.get("variants") or [])
    by_variant = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip()
        if variant_id:
            by_variant[variant_id] = row
    return list(by_variant.values())


def aggregate_group(rows: Iterable[Dict], allowed_variants: List[str]) -> Dict[str, object]:
    allowed = set(allowed_variants)
    selected = [row for row in rows if str(row.get("variant_id") or "") in allowed]
    episodes = sum(int(row.get("episodes", 0) or 0) for row in selected)
    successes = sum(int(row.get("successful_episodes", 0) or 0) for row in selected)
    weighted_steps = sum(float(row.get("mean_steps", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0)) for row in selected)
    return {
        "variant_count": len(selected),
        "episodes": int(episodes),
        "successful_episodes": int(successes),
        "success_rate": (float(successes) / float(episodes)) if episodes > 0 else None,
        "mean_steps": (weighted_steps / float(episodes)) if episodes > 0 else None,
        "variants": sorted(str(row.get("variant_id") or "") for row in selected),
    }


def aggregate_full(rows: Iterable[Dict]) -> Dict[str, object]:
    selected = list(rows)
    episodes = sum(int(row.get("episodes", 0) or 0) for row in selected)
    successes = sum(int(row.get("successful_episodes", 0) or 0) for row in selected)
    weighted_steps = sum(float(row.get("mean_steps", 0.0) or 0.0) * float(int(row.get("episodes", 0) or 0)) for row in selected)
    return {
        "variant_count": len(selected),
        "episodes": int(episodes),
        "successful_episodes": int(successes),
        "success_rate": (float(successes) / float(episodes)) if episodes > 0 else None,
        "mean_steps": (weighted_steps / float(episodes)) if episodes > 0 else None,
        "variants": sorted(str(row.get("variant_id") or "") for row in selected),
    }


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def compute_report(summary_payload: Dict, baseline_payload: Dict | None, label: str) -> Dict:
    rows = normalize_variant_rows(summary_payload)
    groups = {
        "train8": aggregate_group(rows, TRAIN_VARIANTS),
        "heldout8": aggregate_group(rows, HELDOUT_VARIANTS),
        "full16": aggregate_full(rows),
    }
    report = {
        "label": label,
        "groups": groups,
        "train_variants": TRAIN_VARIANTS,
        "heldout_variants": HELDOUT_VARIANTS,
        "round_robin_schedule": ROUND_ROBIN_SCHEDULE,
        "per_variant": sorted(
            [
                {
                    "variant_id": str(row.get("variant_id") or ""),
                    "episodes": int(row.get("episodes", 0) or 0),
                    "successful_episodes": int(row.get("successful_episodes", 0) or 0),
                    "success_rate": row.get("success_rate"),
                    "mean_steps": row.get("mean_steps"),
                    "instance_idx": row.get("instance_idx"),
                }
                for row in rows
            ],
            key=lambda item: str(item["variant_id"]),
        ),
    }
    if baseline_payload is None:
        report["baseline_groups"] = {}
        report["deltas"] = {}
        return report

    baseline_rows = normalize_variant_rows(baseline_payload)
    baseline_groups = {
        "train8": aggregate_group(baseline_rows, TRAIN_VARIANTS),
        "heldout8": aggregate_group(baseline_rows, HELDOUT_VARIANTS),
        "full16": aggregate_full(baseline_rows),
    }
    delta_train = None
    delta_heldout = None
    delta_full = None
    if baseline_groups["train8"]["success_rate"] is not None and groups["train8"]["success_rate"] is not None:
        delta_train = float(groups["train8"]["success_rate"]) - float(baseline_groups["train8"]["success_rate"])
    if baseline_groups["heldout8"]["success_rate"] is not None and groups["heldout8"]["success_rate"] is not None:
        delta_heldout = float(groups["heldout8"]["success_rate"]) - float(baseline_groups["heldout8"]["success_rate"])
    if baseline_groups["full16"]["success_rate"] is not None and groups["full16"]["success_rate"] is not None:
        delta_full = float(groups["full16"]["success_rate"]) - float(baseline_groups["full16"]["success_rate"])
    transfer_ratio = None
    if delta_train is not None and abs(float(delta_train)) > 1e-8 and delta_heldout is not None:
        transfer_ratio = float(delta_heldout) / float(delta_train)
    report["baseline_groups"] = baseline_groups
    report["deltas"] = {
        "delta_train8": delta_train,
        "delta_heldout8": delta_heldout,
        "delta_full16": delta_full,
        "transfer_ratio": transfer_ratio,
    }
    return report


def write_markdown(report: Dict, out_path: Path) -> None:
    groups = report["groups"]
    baseline_groups = report.get("baseline_groups") or {}
    deltas = report.get("deltas") or {}
    lines = [
        f"# {report['label']}",
        "",
        "| split | success | successes / episodes | mean_steps |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train8", "heldout8", "full16"):
        row = groups[split]
        lines.append(
            f"| {split} | {fmt_rate(row.get('success_rate'))} | "
            f"{int(row.get('successful_episodes', 0))}/{int(row.get('episodes', 0))} | "
            f"{fmt_float(row.get('mean_steps'))} |"
        )
    if baseline_groups:
        lines.extend(
            [
                "",
                "| delta | value |",
                "|---|---:|",
                f"| delta_train8 | {fmt_rate(deltas.get('delta_train8'))} |",
                f"| delta_heldout8 | {fmt_rate(deltas.get('delta_heldout8'))} |",
                f"| delta_full16 | {fmt_rate(deltas.get('delta_full16'))} |",
                f"| transfer_ratio | {fmt_float(deltas.get('transfer_ratio'))} |",
            ]
        )
    lines.extend(
        [
            "",
            "| variant | success | successes / episodes | mean_steps |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["per_variant"]:
        lines.append(
            f"| {row['variant_id']} | {fmt_rate(row.get('success_rate'))} | "
            f"{int(row.get('successful_episodes', 0))}/{int(row.get('episodes', 0))} | "
            f"{fmt_float(row.get('mean_steps'))} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_field(args: argparse.Namespace) -> int:
    mapping = {
        "train_variants": TRAIN_VARIANTS,
        "heldout_variants": HELDOUT_VARIANTS,
        "round_robin_schedule": ROUND_ROBIN_SCHEDULE,
        "family_names": list(FAMILY_VARIANTS.keys()),
        "cv_fold_names": list(CV_FOLDS.keys()),
    }
    for item in mapping[args.name]:
        print(item)
    return 0


def _build_manifest_view(source_manifest: Dict, selected_rows: List[Dict], split_label: str) -> Dict:
    payload = dict(source_manifest)
    payload["split_label"] = split_label
    payload["instance_worlds"] = selected_rows
    return payload


def write_bank_views_from_manifest(manifest: Dict, out_dir: Path, asset_dir_hint: str = "") -> Dict[str, object]:
    rows = list(manifest.get("instance_worlds") or [])
    by_variant = {}
    for row in rows:
        variant = str(row.get("path_obstacle_variant_id") or "").strip()
        if variant:
            by_variant[variant] = row
    required = TRAIN_VARIANTS + HELDOUT_VARIANTS
    missing = [variant for variant in required if variant not in by_variant]
    if missing:
        raise SystemExit(f"missing variants in manifest: {missing}")

    (out_dir / "singletons").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    for variant, row in sorted(by_variant.items()):
        bank_dir = out_dir / "singletons" / variant
        bank_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_manifest_view(manifest, [row], split_label=f"singleton:{variant}")
        (bank_dir / "bank_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    split_payloads = {
        "train8": [by_variant[variant] for variant in TRAIN_VARIANTS],
        "heldout8": [by_variant[variant] for variant in HELDOUT_VARIANTS],
        "full16": [by_variant[variant] for variant in sorted(by_variant.keys())],
    }
    for split_name, split_rows in split_payloads.items():
        bank_dir = out_dir / "splits" / split_name
        bank_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_manifest_view(manifest, split_rows, split_label=split_name)
        (bank_dir / "bank_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for family_name, family_variants in FAMILY_VARIANTS.items():
        bank_dir = out_dir / "families" / family_name
        bank_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_manifest_view(
            manifest,
            [by_variant[variant] for variant in family_variants],
            split_label=f"family:{family_name}",
        )
        (bank_dir / "bank_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for fold_name, spec in CV_FOLDS.items():
        val_families = list(spec["val_families"])
        test_families = list(spec["test_families"])
        train_families = [
            family_name
            for family_name in FAMILY_VARIANTS.keys()
            if family_name not in set(val_families + test_families)
        ]
        fold_payloads = {
            "train": [by_variant[variant] for family in train_families for variant in FAMILY_VARIANTS[family]],
            "val": [by_variant[variant] for family in val_families for variant in FAMILY_VARIANTS[family]],
            "test": [by_variant[variant] for family in test_families for variant in FAMILY_VARIANTS[family]],
        }
        for split_name, split_rows in fold_payloads.items():
            bank_dir = out_dir / "cv_folds" / fold_name / split_name
            bank_dir.mkdir(parents=True, exist_ok=True)
            payload = _build_manifest_view(
                manifest,
                split_rows,
                split_label=f"{fold_name}:{split_name}",
            )
            (bank_dir / "bank_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    spec_path = out_dir / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "train_variants": TRAIN_VARIANTS,
                "heldout_variants": HELDOUT_VARIANTS,
                "round_robin_schedule": ROUND_ROBIN_SCHEDULE,
                "family_variants": FAMILY_VARIANTS,
                "cv_folds": {
                    fold_name: {
                        "train_families": [
                            family_name
                            for family_name in FAMILY_VARIANTS.keys()
                            if family_name not in set(list(spec["val_families"]) + list(spec["test_families"]))
                        ],
                        "val_families": list(spec["val_families"]),
                        "test_families": list(spec["test_families"]),
                    }
                    for fold_name, spec in CV_FOLDS.items()
                },
                "asset_dir": str(asset_dir_hint or ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"out_dir": str(out_dir), "spec_path": str(spec_path)}


def cmd_make_bank_views(args: argparse.Namespace) -> int:
    asset_dir = Path(args.asset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest = load_manifest(asset_dir)
    result = write_bank_views_from_manifest(manifest, out_dir, asset_dir_hint=str(asset_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_make_bank_views_from_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest = load_manifest_path(manifest_path)
    result = write_bank_views_from_manifest(manifest, out_dir, asset_dir_hint=str(args.asset_dir_hint or ""))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_summarize_probe(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary_json).resolve()
    baseline_path = Path(args.baseline_summary).resolve() if args.baseline_summary else None
    summary_payload = load_summary(summary_path)
    baseline_payload = load_summary(baseline_path) if baseline_path else None
    report = compute_report(summary_payload, baseline_payload, args.label)
    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, out_md)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    field_parser = subparsers.add_parser("field")
    field_parser.add_argument(
        "--name",
        required=True,
        choices=["train_variants", "heldout_variants", "round_robin_schedule", "family_names", "cv_fold_names"],
    )
    field_parser.set_defaults(func=cmd_field)

    bank_parser = subparsers.add_parser("make-bank-views")
    bank_parser.add_argument("--asset-dir", required=True)
    bank_parser.add_argument("--out-dir", required=True)
    bank_parser.set_defaults(func=cmd_make_bank_views)

    bank_from_manifest_parser = subparsers.add_parser("make-bank-views-from-manifest")
    bank_from_manifest_parser.add_argument("--manifest-path", required=True)
    bank_from_manifest_parser.add_argument("--out-dir", required=True)
    bank_from_manifest_parser.add_argument("--asset-dir-hint", default="")
    bank_from_manifest_parser.set_defaults(func=cmd_make_bank_views_from_manifest)

    summary_parser = subparsers.add_parser("summarize-probe")
    summary_parser.add_argument("--summary-json", required=True)
    summary_parser.add_argument("--baseline-summary", default="")
    summary_parser.add_argument("--label", required=True)
    summary_parser.add_argument("--out-json", default="")
    summary_parser.add_argument("--out-md", default="")
    summary_parser.set_defaults(func=cmd_summarize_probe)

    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
