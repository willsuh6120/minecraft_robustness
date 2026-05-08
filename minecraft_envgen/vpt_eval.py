from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import ray

from minecraft_envgen.evidence import build_feedback_packet, summarize_episodes
from minecraft_envgen.runtime import make_vpt_agent_generator


DEFAULT_SUCCESS_SPECS: Dict[str, Tuple[str, str, int]] = {
    "collect_wood": ("mine_block", ".*log.*", 1),
    "collect_stone": ("mine_block", ".*stone.*", 1),
    "collect_iron": ("mine_block", ".*iron_ore.*", 1),
    "collect_diamond": ("mine_block", ".*diamond_ore.*", 1),
    "defeat_zombie": ("kill_entity", ".*zombie.*", 1),
    "defeat_skeleton": ("kill_entity", ".*skeleton.*", 1),
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_success_spec(
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


def collect_episodes(generator: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [episode for episode in generator]


def ensure_minestudio_engine() -> None:
    from minestudio.simulator.entry import check_engine

    check_engine(skip_confirmation=True)


def ensure_virtualgl_session() -> None:
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


def run_vpt_rollout(
    *,
    env_generator: Any,
    target_skill: str,
    model_uri: str,
    output_dir: str | Path,
    split: str,
    reset_mode: str,
    episodes: int,
    steps: int,
    num_workers: int,
    num_gpus_per_worker: float,
    success_key: str | None = None,
    success_regex: str | None = None,
    success_num: int | None = None,
    tail_window: int = 128,
    max_failures: int = 3,
    speed_test_interval: int = 50,
    java_max_mem: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"

    success_key, success_regex, success_num = resolve_success_spec(
        target_skill=target_skill,
        success_key=success_key,
        success_regex=success_regex,
        success_num=success_num,
    )

    ensure_minestudio_engine()
    ensure_virtualgl_session()
    if java_max_mem:
        os.environ["MINESTUDIO_JAVA_MAX_MEM"] = java_max_mem

    agent_generator = make_vpt_agent_generator(model_uri)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from minestudio.inference import MineGenerator
    from minestudio.simulator.callbacks import SpeedTestCallback

    wrapped_env_generator = env_generator
    if speed_test_interval > 0:
        speed_callback = SpeedTestCallback(int(speed_test_interval))

        def _wrapped() -> Any:
            env = env_generator()
            env.callbacks.append(speed_callback)
            return env

        wrapped_env_generator = _wrapped

    generator = MineGenerator(
        num_workers=num_workers,
        num_gpus=num_gpus_per_worker,
        env_generator=wrapped_env_generator,
        agent_generator=agent_generator,
        num_max_steps=steps,
        num_episodes=episodes,
        tmpdir=str(episodes_dir),
        image_media="h264",
    )

    rollout_episodes = collect_episodes(generator.generate())
    summary = summarize_episodes(
        rollout_episodes,
        success_key=success_key,
        success_regex=success_regex,
        success_num=success_num,
    )
    write_json(output_root / "summary.json", summary)

    feedback_packet = build_feedback_packet(
        episodes=rollout_episodes,
        run_id=str(uuid.uuid4()),
        output_dir=output_root,
        split=split,
        reset_mode=reset_mode,
        success_key=success_key,
        success_regex=success_regex,
        success_num=success_num,
        tail_window=tail_window,
        max_failure_episodes=max_failures,
        notes=notes,
    )
    write_json(output_root / "feedback_packet.json", feedback_packet.to_dict())

    print("compiled_reset_mode:", reset_mode)
    print("success_rate:", summary["yes_rate"])
    print("episodes_dir:", episodes_dir)
    print("feedback_packet:", output_root / "feedback_packet.json")

    return {
        "summary": summary,
        "feedback_packet": feedback_packet,
        "episodes": rollout_episodes,
        "episodes_dir": str(episodes_dir),
    }
