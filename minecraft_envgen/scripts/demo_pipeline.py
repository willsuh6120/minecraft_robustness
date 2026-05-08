from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from minecraft_envgen.pipeline import compile_curriculum, compile_for_all_adapters


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, help="Path to a curriculum draft JSON file.")
    parser.add_argument("--feedback", help="Path to a feedback packet JSON file.")
    parser.add_argument(
        "--adapter",
        default="all",
        choices=["all", "minedojo", "minestudio"],
        help="Which adapter to compile.",
    )
    parser.add_argument("--split", choices=["train", "eval"], help="Override split.")
    parser.add_argument("--output-dir", help="Optional directory to write compiled outputs.")
    args = parser.parse_args()

    draft = _load_json(args.draft)
    feedback = _load_json(args.feedback) if args.feedback else None

    if args.adapter == "all":
        outcomes = compile_for_all_adapters(draft, feedback_data=feedback, split=args.split)
    else:
        outcomes = {
            args.adapter: compile_curriculum(
                draft_data=draft,
                adapter_name=args.adapter,
                feedback_data=feedback,
                split=args.split,
            )
        }

    for adapter_name, outcome in outcomes.items():
        print(f"[{adapter_name}]")
        print("prompt_chars:", len(outcome.prompt))
        print("errors:", sum(1 for item in outcome.report.messages if item.severity.value == "error"))
        print("warnings:", sum(1 for item in outcome.report.messages if item.severity.value == "warning"))
        for message in outcome.report.messages:
            print(f"- {message.severity.value}: {message.location}: {message.message}")
        if outcome.compiled is not None:
            print("compiled_reset_mode:", outcome.compiled.reset_mode.value)
            print("dependency_installed:", outcome.compiled.runtime["dependency"]["installed"])
        print()

        if args.output_dir:
            root = Path(args.output_dir) / adapter_name
            root.mkdir(parents=True, exist_ok=True)
            (root / "prompt.txt").write_text(outcome.prompt, encoding="utf-8")
            _write_json(root / "report.json", outcome.report.to_dict())
            if outcome.compiled is not None:
                _write_json(root / "compiled.json", outcome.compiled.to_dict())


if __name__ == "__main__":
    main()
