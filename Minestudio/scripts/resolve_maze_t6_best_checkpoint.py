#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ITER_RE = re.compile(r"iter_(\d+)_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_bank_location(run_root: Path) -> tuple[Path | None, str]:
    assets_root = run_root / "assets"
    if assets_root.is_dir():
        candidates = sorted(p for p in assets_root.iterdir() if p.is_dir())
        if candidates:
            return candidates[-1], "eval_bank"
    if (run_root / "eval_bank" / "bank_manifest.json").is_file():
        return run_root, "eval_bank"
    if (run_root / "bank_views" / "splits" / "full16" / "bank_manifest.json").is_file():
        return run_root / "bank_views" / "splits", "full16"
    return None, ""


def resolve_training_model(run_root: Path, experiment_name: str, iteration_idx: int) -> Path:
    iter_dir = run_root / experiment_name / "training" / f"iter_{iteration_idx:03d}"
    candidates = sorted(iter_dir.glob("*/suite_iteration.json"))
    if not candidates:
        raise FileNotFoundError(f"missing suite_iteration.json under {iter_dir}")
    payload = load_json(candidates[-1])
    model_path = Path(str(payload.get("next_model_path") or "")).expanduser()
    if not model_path:
        raise RuntimeError(f"missing next_model_path in {candidates[-1]}")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--metric-group", default="full16")
    parser.add_argument("--metric-key", default="success_rate")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    experiment_root = run_root / args.experiment_name
    eval_root = experiment_root / "evals"
    if not eval_root.is_dir():
        raise SystemExit(f"missing eval root: {eval_root}")

    rows: list[dict[str, Any]] = []
    for report_path in sorted(eval_root.glob("iter_*_full16_*/split_report.json")):
        match = ITER_RE.search(report_path.parent.name)
        if not match:
            continue
        iteration_idx = int(match.group(1))
        payload = load_json(report_path)
        group = (payload.get("groups") or {}).get(args.metric_group) or {}
        score = group.get(args.metric_key)
        episodes = int(group.get("episodes") or 0)
        if score is None:
            continue
        rows.append(
            {
                "iteration_idx": iteration_idx,
                "report_path": str(report_path),
                "score": float(score),
                "episodes": episodes,
            }
        )

    if not rows:
        raise SystemExit(f"no eligible split reports found under {eval_root}")

    rows.sort(key=lambda item: (item["score"], item["episodes"], item["iteration_idx"]))
    best = rows[-1]
    best_iter = int(best["iteration_idx"])
    best_model_path = resolve_training_model(run_root, args.experiment_name, best_iter)
    asset_dir, bank_name = find_bank_location(run_root)

    payload = {
        "run_root": str(run_root),
        "experiment_name": args.experiment_name,
        "metric_group": args.metric_group,
        "metric_key": args.metric_key,
        "candidate_checkpoints": rows,
        "best_iteration": best_iter,
        "best_score": best["score"],
        "best_episodes": best["episodes"],
        "best_report_path": best["report_path"],
        "best_model_path": str(best_model_path),
        "asset_dir": str(asset_dir) if asset_dir else "",
        "bank_name": bank_name,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
