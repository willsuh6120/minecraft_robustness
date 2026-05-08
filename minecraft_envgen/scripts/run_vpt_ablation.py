from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from minecraft_envgen.core import CurriculumDraft, FeedbackPacket, PromptBuilder
from minecraft_envgen.llm import generate_draft_with_openai
from minecraft_envgen.pipeline import compile_curriculum
from minecraft_envgen.runtime import make_default_env_generator, make_env_generator
from minecraft_envgen.vpt_eval import load_json, run_vpt_rollout, write_json


def _scalar_only_feedback(packet: FeedbackPacket) -> FeedbackPacket:
    return FeedbackPacket(
        run_id=packet.run_id,
        split=packet.split,
        reset_mode=packet.reset_mode,
        scalar_metrics=dict(packet.scalar_metrics),
        failure_evidence=[],
        verifier_notes=["Scalar-only ablation: multimodal evidence removed before prompting."],
        episode_count=packet.episode_count,
        notes=packet.notes,
    )


def _baseline_env_generator(draft: CurriculumDraft) -> Any:
    return make_default_env_generator(
        action_type=str(draft.metadata.get("action_type", "agent")),
        obs_size=list(draft.metadata.get("image_size", [128, 128])),
        timestep_limit=int(draft.metadata.get("timestep_limit", 1000)),
    )


def _save_compilation_artifacts(cycle_dir: Path, outcome: Any) -> None:
    (cycle_dir / "prompt.txt").write_text(outcome.prompt, encoding="utf-8")
    write_json(cycle_dir / "report.json", outcome.report.to_dict())
    if outcome.compiled is not None:
        write_json(cycle_dir / "compiled.json", outcome.compiled.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "baseline_default", "scalar_envgen", "multimodal_envgen", "multimodal_envgen_verifier"],
    )
    parser.add_argument("--draft", required=True, help="Initial curriculum draft JSON.")
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--cycles", type=int, default=1, help="Number of rollout cycles.")
    parser.add_argument(
        "--model",
        default="CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        help="Hugging Face VPT baseline id.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-gpus-per-worker", type=float, default=0.25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-window", type=int, default=128)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--speed-test-interval", type=int, default=50)
    parser.add_argument("--java-max-mem", default="4G", help="Java heap size for each Minecraft process, e.g. 4G.")
    parser.add_argument("--success-key")
    parser.add_argument("--success-regex")
    parser.add_argument("--success-num", type=int)
    parser.add_argument("--llm-model", default="gpt-5-nano")
    parser.add_argument("--dotenv", help="Optional .env file containing OPENAI_API_KEY or API_KEY.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    initial_draft = CurriculumDraft.from_dict(load_json(args.draft))
    current_draft = initial_draft

    use_llm = args.mode not in {"baseline", "baseline_default"}
    use_multimodal = args.mode in {"multimodal_envgen", "multimodal_envgen_verifier"}
    use_verifiers = args.mode == "multimodal_envgen_verifier"

    for cycle_index in range(args.cycles):
        cycle_dir = output_root / f"cycle_{cycle_index:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        write_json(cycle_dir / "draft.json", current_draft.to_dict())

        if args.mode == "baseline_default":
            env_generator = _baseline_env_generator(current_draft)
            run_vpt_rollout(
                env_generator=env_generator,
                target_skill=current_draft.target_skill,
                model_uri=args.model,
                output_dir=cycle_dir,
                split=args.split,
                reset_mode="clean",
                episodes=args.episodes,
                steps=args.steps,
                num_workers=args.num_workers,
                num_gpus_per_worker=args.num_gpus_per_worker,
                success_key=args.success_key,
                success_regex=args.success_regex,
                success_num=args.success_num,
                tail_window=args.tail_window,
                max_failures=args.max_failures,
                speed_test_interval=args.speed_test_interval,
                java_max_mem=args.java_max_mem,
                notes="VPT-RL only baseline on the default MineStudio sandbox environment.",
            )
            continue

        outcome = compile_curriculum(
            current_draft.to_dict(),
            adapter_name="minestudio",
            split=args.split,
            use_verifiers=use_verifiers,
        )
        _save_compilation_artifacts(cycle_dir, outcome)
        if outcome.compiled is None:
            (cycle_dir / "status.txt").write_text("Compilation failed before rollout.\n", encoding="utf-8")
            break

        result = run_vpt_rollout(
            env_generator=make_env_generator(outcome.compiled),
            target_skill=current_draft.target_skill,
            model_uri=args.model,
            output_dir=cycle_dir,
            split=outcome.compiled.split.value,
            reset_mode=outcome.compiled.reset_mode.value,
            episodes=args.episodes,
            steps=args.steps,
            num_workers=args.num_workers,
            num_gpus_per_worker=args.num_gpus_per_worker,
            success_key=args.success_key,
            success_regex=args.success_regex,
            success_num=args.success_num,
            tail_window=args.tail_window,
            max_failures=args.max_failures,
            speed_test_interval=args.speed_test_interval,
            java_max_mem=args.java_max_mem,
            notes=(
                "VPT-RL only baseline on the fixed compiled curriculum environment."
                if args.mode == "baseline"
                else f"{args.mode} rollout from the VPT-RL baseline."
            ),
        )

        if cycle_index >= args.cycles - 1 or not use_llm:
            continue

        feedback = result["feedback_packet"]
        prompt_feedback = feedback if use_multimodal else _scalar_only_feedback(feedback)
        prompt = PromptBuilder().build_update_prompt(current_draft, prompt_feedback)
        (cycle_dir / "update_prompt.txt").write_text(prompt, encoding="utf-8")

        next_draft, raw_response = generate_draft_with_openai(
            prompt=prompt,
            model=args.llm_model,
            dotenv_path=args.dotenv,
        )
        (cycle_dir / "llm_response.txt").write_text(raw_response, encoding="utf-8")
        write_json(cycle_dir / "next_draft.json", next_draft.to_dict())
        current_draft = next_draft


if __name__ == "__main__":
    main()
