from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

import ray

from minecraft_envgen.evidence import build_feedback_packet, summarize_episodes
from minecraft_envgen.pipeline import compile_curriculum
from minecraft_envgen.runtime import make_env_generator, make_vpt_agent_generator


DEFAULT_SUCCESS_SPECS: Dict[str, Tuple[str, str, int]] = {
    "collect_wood": ("mine_block", ".*log.*", 1),
    "collect_stone": ("mine_block", ".*stone.*", 1),
    "collect_iron": ("mine_block", ".*iron_ore.*", 1),
    "collect_diamond": ("mine_block", ".*diamond_ore.*", 1),
    "defeat_zombie": ("kill_entity", ".*zombie.*", 1),
    "defeat_skeleton": ("kill_entity", ".*skeleton.*", 1),
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_success_spec(
    target_skill: str,
    success_key: str | None,
    success_regex: str | None,
    success_num: int | None,
) -> Tuple[str, str, int]:
    if success_key and success_regex:
        return success_key, success_regex, success_num or 1
    if target_skill in DEFAULT_SUCCESS_SPECS:
        default_key, default_regex, default_num = DEFAULT_SUCCESS_SPECS[target_skill]
        return success_key or default_key, success_regex or default_regex, success_num or default_num
    raise ValueError(
        "No default success spec is known for this skill. Pass --success-key and --success-regex explicitly."
    )


def _collect_episodes(generator: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [episode for episode in generator]


def _ensure_minestudio_engine() -> None:
    from minestudio.simulator.entry import check_engine

    # Ray workers are non-interactive, so the simulator engine must exist before
    # any worker constructs MinecraftSim.
    check_engine(skip_confirmation=True)


def _ensure_virtualgl_session() -> None:
    display = os.environ.get("DISPLAY", ":1")
    socket_path = Path(f"/tmp/.X11-unix/X{display.lstrip(':')}")

    os.environ["DISPLAY"] = display
    os.environ.setdefault("VGL_DISPLAY", "egl")
    os.environ.setdefault("VGL_REFRESHRATE", "60")
    os.environ["PATH"] = os.environ.get("PATH", "") + ":/opt/VirtualGL/bin"

    xauth_path = Path.home() / ".Xauthority"
    if not xauth_path.exists():
        xauth_path.touch()
    os.environ.setdefault("XAUTHORITY", str(xauth_path))

    if socket_path.exists():
        return

    subprocess.Popen(
        [
            "Xvfb",
            display,
            "-ac",
            "-screen",
            "0",
            "1920x1200x24",
            "-dpi",
            "72",
            "+extension",
            "RANDR",
            "+extension",
            "GLX",
            "+iglx",
            "+extension",
            "MIT-SHM",
            "+render",
            "-nolisten",
            "tcp",
            "-noreset",
            "-shmem",
            "-maxclients",
            "2048",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 15
    while time.time() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for Xvfb socket at {socket_path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, help="Path to a curriculum draft JSON file.")
    parser.add_argument("--feedback", help="Optional feedback packet JSON used to compile the next draft.")
    parser.add_argument("--split", default="eval", choices=["train", "eval"], help="Which split to compile and run.")
    parser.add_argument(
        "--model",
        default="CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        help="Hugging Face model id for the VPT policy baseline.",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Number of rollout episodes.")
    parser.add_argument("--steps", type=int, default=1200, help="Maximum rollout steps per episode.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel MineGenerator workers.")
    parser.add_argument("--num-gpus-per-worker", type=float, default=0.25, help="GPU fraction per worker.")
    parser.add_argument("--output-dir", required=True, help="Directory for prompt/report/runtime and rollout outputs.")
    parser.add_argument("--run-id", help="Optional run id for the feedback packet.")
    parser.add_argument("--success-key", help="Info dict key used for success counting.")
    parser.add_argument("--success-regex", help="Regex over the final info bucket for success counting.")
    parser.add_argument("--success-num", type=int, help="Minimum matched count required for success.")
    parser.add_argument("--tail-window", type=int, default=128, help="How many trailing steps to keep in failure packets.")
    parser.add_argument("--max-failures", type=int, default=3, help="How many failed episodes to package as evidence.")
    parser.add_argument("--speed-test-interval", type=int, default=50, help="MineStudio speed log interval.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    draft = _load_json(args.draft)
    feedback = _load_json(args.feedback) if args.feedback else None
    outcome = compile_curriculum(
        draft_data=draft,
        adapter_name="minestudio",
        feedback_data=feedback,
        split=args.split,
    )

    (output_dir / "prompt.txt").write_text(outcome.prompt, encoding="utf-8")
    _write_json(output_dir / "report.json", outcome.report.to_dict())
    if outcome.compiled is None:
        raise SystemExit("Compilation failed. See report.json for verifier errors.")
    _write_json(output_dir / "compiled.json", outcome.compiled.to_dict())

    success_key, success_regex, success_num = _resolve_success_spec(
        target_skill=draft["target_skill"],
        success_key=args.success_key,
        success_regex=args.success_regex,
        success_num=args.success_num,
    )

    extra_callbacks = []
    if args.speed_test_interval > 0:
        extra_callbacks.append({"name": "SpeedTestCallback", "config": {"interval": args.speed_test_interval}})

    _ensure_minestudio_engine()
    _ensure_virtualgl_session()

    env_generator = make_env_generator(outcome.compiled, extra_callbacks=extra_callbacks)
    agent_generator = make_vpt_agent_generator(args.model)
    episodes_dir = output_dir / "episodes"

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from minestudio.inference import MineGenerator

    generator = MineGenerator(
        num_workers=args.num_workers,
        num_gpus=args.num_gpus_per_worker,
        env_generator=env_generator,
        agent_generator=agent_generator,
        num_max_steps=args.steps,
        num_episodes=args.episodes,
        tmpdir=str(episodes_dir),
        image_media="h264",
    )

    episodes = _collect_episodes(generator.generate())
    summary = summarize_episodes(
        episodes,
        success_key=success_key,
        success_regex=success_regex,
        success_num=success_num,
    )
    _write_json(output_dir / "summary.json", summary)

    feedback_packet = build_feedback_packet(
        episodes=episodes,
        run_id=args.run_id or str(uuid.uuid4()),
        output_dir=output_dir,
        split=outcome.compiled.split.value,
        reset_mode=outcome.compiled.reset_mode.value,
        success_key=success_key,
        success_regex=success_regex,
        success_num=success_num,
        tail_window=args.tail_window,
        max_failure_episodes=args.max_failures,
        notes="Built from the MineStudio VPT baseline with EnvGen-compiled curriculum runtime.",
    )
    _write_json(output_dir / "feedback_packet.json", feedback_packet.to_dict())

    print("compiled_reset_mode:", outcome.compiled.reset_mode.value)
    print("success_rate:", summary["yes_rate"])
    print("episodes_dir:", episodes_dir)
    print("feedback_packet:", output_dir / "feedback_packet.json")


if __name__ == "__main__":
    main()
