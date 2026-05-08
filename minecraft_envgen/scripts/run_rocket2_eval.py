from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from minecraft_envgen.rocket2_eval import (
    infer_reset_mode_from_env_config,
    make_goal_conditioned_env_generator,
    run_rocket2_rollout,
    snapshot_eval_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True, help="Path to a ROCKET-2/MineStudio environment YAML.")
    parser.add_argument("--target-skill", required=True, help="Skill name used for success evaluation.")
    parser.add_argument(
        "--model",
        default="hf:phython96/ROCKET-2-1x-22w",
        help="ROCKET-2 checkpoint path or hf:<repo_id>.",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.5, help="Classifier-free guidance coefficient.")
    parser.add_argument("--goal-image", required=True, help="Cross-view goal image path.")
    parser.add_argument("--goal-mask", required=True, help="Binary mask path in the goal image.")
    parser.add_argument(
        "--segment-type",
        required=True,
        choices=["Approach", "Interact", "Hunt", "Mine", "Craft", "Use", "Switch", "None"],
        help="Interaction type id used by ROCKET-2.",
    )
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-gpus-per-worker", type=float, default=0.25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-window", type=int, default=128)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--speed-test-interval", type=int, default=50)
    parser.add_argument("--java-max-mem", default="4G")
    parser.add_argument("--success-key")
    parser.add_argument("--success-regex")
    parser.add_argument("--success-num", type=int)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot_eval_inputs(
        output_dir=output_root,
        env_config_path=args.env_config,
        goal_image_path=args.goal_image,
        goal_mask_path=args.goal_mask,
        segment_type=args.segment_type,
        model_uri=args.model,
        cfg_coef=args.cfg_coef,
    )

    env_generator = make_goal_conditioned_env_generator(
        env_config_path=args.env_config,
        goal_image_path=args.goal_image,
        goal_mask_path=args.goal_mask,
        segment_type=args.segment_type,
        speed_test_interval=args.speed_test_interval,
    )
    reset_mode = infer_reset_mode_from_env_config(args.env_config)

    run_rocket2_rollout(
        env_generator=env_generator,
        target_skill=args.target_skill,
        model_uri=args.model,
        output_dir=output_root,
        split=args.split,
        reset_mode=reset_mode,
        episodes=args.episodes,
        steps=args.steps,
        num_workers=args.num_workers,
        num_gpus_per_worker=args.num_gpus_per_worker,
        cfg_coef=args.cfg_coef,
        success_key=args.success_key,
        success_regex=args.success_regex,
        success_num=args.success_num,
        tail_window=args.tail_window,
        max_failures=args.max_failures,
        java_max_mem=args.java_max_mem,
        notes=(
            "ROCKET-2 headless rollout with a fixed cross-view goal image and binary goal mask."
        ),
    )


if __name__ == "__main__":
    main()
