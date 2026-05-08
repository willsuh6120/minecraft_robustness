from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import ray
import torch
from PIL import Image

from minecraft_envgen.evidence import build_feedback_packet, summarize_episodes
from minecraft_envgen.vpt_eval import resolve_success_spec, write_json
from minestudio.simulator.callbacks import MinecraftCallback


ROOT_DIR = Path(__file__).resolve().parents[1]
ROCKET2_DIR = ROOT_DIR / "ROCKET-2"
if str(ROCKET2_DIR) not in sys.path:
    sys.path.insert(0, str(ROCKET2_DIR))

from cfg_wrapper import CFGWrapper
from model import CrossViewRocket, load_cross_view_rocket


DEFAULT_OBS_SIZE: Tuple[int, int] = (224, 224)
SEGMENT_MAPPING = {
    "Hunt": 0,
    "Use": 3,
    "Mine": 2,
    "Interact": 3,
    "Craft": 4,
    "Switch": 5,
    "Approach": 6,
    "None": -1,
}


def ensure_minestudio_engine() -> None:
    from minestudio.simulator.entry import check_engine

    check_engine(skip_confirmation=True)


def ensure_virtualgl_session() -> None:
    display = os.environ.get("DISPLAY", ":1")
    socket_path = Path(f"/tmp/.X11-unix/X{display.lstrip(':')}")

    os.environ["DISPLAY"] = display
    os.environ.setdefault("VGL_DISPLAY", "egl")
    os.environ.setdefault("VGL_REFRESHRATE", "60")
    os.environ.setdefault("ALSOFT_DRIVERS", "null")
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


def load_goal_assets(
    goal_image_path: str | Path,
    goal_mask_path: str | Path,
    *,
    obs_size: Tuple[int, int] = DEFAULT_OBS_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(obs_size[0]), int(obs_size[1])

    goal_image = Image.open(goal_image_path).convert("RGB")
    goal_image = goal_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    goal_image_np = np.asarray(goal_image, dtype=np.uint8)

    raw_mask = Image.open(goal_mask_path)
    alpha = np.asarray(raw_mask.getchannel("A")) if "A" in raw_mask.getbands() else None
    if alpha is not None and np.any(alpha > 0):
        mask_gray = Image.fromarray(alpha, mode="L")
    else:
        mask_gray = raw_mask.convert("L")
    mask_gray = mask_gray.resize((width, height), resample=Image.Resampling.NEAREST)
    goal_mask_np = (np.asarray(mask_gray) > 0).astype(np.uint8)
    return goal_image_np, goal_mask_np


class GoalConditionCallback(MinecraftCallback):
    def __init__(self, goal_image: np.ndarray, goal_mask: np.ndarray, segment_type: str) -> None:
        if segment_type not in SEGMENT_MAPPING:
            valid = ", ".join(sorted(SEGMENT_MAPPING))
            raise ValueError(f"Unknown segment type '{segment_type}'. Expected one of: {valid}.")
        self.goal_image = np.asarray(goal_image, dtype=np.uint8)
        self.goal_mask = (np.asarray(goal_mask) > 0).astype(np.uint8)
        self.segment_type = segment_type
        self.obj_id = SEGMENT_MAPPING[segment_type]

    def _inject(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(obs)
        obs["cross_view"] = {
            "cross_view_image": self.goal_image.copy(),
            "cross_view_obj_id": torch.tensor(self.obj_id, dtype=torch.long),
            "cross_view_obj_mask": torch.tensor(self.goal_mask.copy(), dtype=torch.uint8),
        }
        return obs

    def after_reset(self, sim: Any, obs: Dict[str, Any], info: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        return self._inject(obs), info

    def after_step(
        self,
        sim: Any,
        obs: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return self._inject(obs), reward, terminated, truncated, info


class Rocket2Agent:
    def __init__(self, model_uri: str, cfg_coef: float = 1.0) -> None:
        self.model_uri = model_uri
        self.cfg_coef = cfg_coef
        self.model = None
        self.agent = None
        self.cache_latents: Dict[str, Any] = {}

    def _load_model(self) -> Any:
        if self.model_uri.startswith("hf:"):
            return CrossViewRocket.from_pretrained(self.model_uri.split(":", 1)[1])
        return load_cross_view_rocket(self.model_uri)

    def to(self, device: Any) -> "Rocket2Agent":
        if self.model is None:
            self.model = self._load_model()
        self.model = self.model.to(device)
        self.agent = CFGWrapper(self.model, k=self.cfg_coef)
        return self

    def eval(self) -> "Rocket2Agent":
        if self.model is not None:
            self.model.eval()
        return self

    def initial_state(self) -> Any:
        if self.agent is None:
            raise RuntimeError("Rocket2Agent must be moved to a device before calling initial_state().")
        return self.agent.initial_state()

    def get_action(self, obs: Dict[str, Any], memory: Any, **kwargs: Any) -> tuple[Dict[str, Any], Any]:
        if self.agent is None:
            raise RuntimeError("Rocket2Agent must be moved to a device before calling get_action().")
        action, memory = self.agent.get_action(obs, memory, **kwargs)
        self.cache_latents = getattr(self.agent, "cache_latents", {})
        return action, memory


def make_rocket2_agent_generator(model_uri: str, cfg_coef: float = 1.0) -> Any:
    return lambda: Rocket2Agent(model_uri=model_uri, cfg_coef=cfg_coef)


def make_goal_conditioned_env_generator(
    *,
    env_config_path: str | Path,
    goal_image_path: str | Path,
    goal_mask_path: str | Path,
    segment_type: str,
    obs_size: Tuple[int, int] = DEFAULT_OBS_SIZE,
    speed_test_interval: int = 50,
) -> Any:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import PrevActionCallback, SpeedTestCallback, load_callbacks_from_config

    goal_image, goal_mask = load_goal_assets(goal_image_path, goal_mask_path, obs_size=obs_size)
    callbacks = list(load_callbacks_from_config(str(env_config_path)))
    if not any(isinstance(callback, PrevActionCallback) for callback in callbacks):
        callbacks.append(PrevActionCallback())
    callbacks.append(GoalConditionCallback(goal_image=goal_image, goal_mask=goal_mask, segment_type=segment_type))
    if speed_test_interval > 0:
        callbacks.append(SpeedTestCallback(int(speed_test_interval)))
    return partial(MinecraftSim, callbacks=callbacks, obs_size=obs_size)


def infer_reset_mode_from_env_config(env_config_path: str | Path) -> str:
    from minestudio.simulator.callbacks import FastResetCallback, HardResetCallback, load_callbacks_from_config

    callbacks = list(load_callbacks_from_config(str(env_config_path)))
    if any(isinstance(callback, HardResetCallback) for callback in callbacks):
        return "clean"
    if any(isinstance(callback, FastResetCallback) for callback in callbacks):
        return "fast"
    return "clean"


def snapshot_eval_inputs(
    *,
    output_dir: str | Path,
    env_config_path: str | Path,
    goal_image_path: str | Path,
    goal_mask_path: str | Path,
    segment_type: str,
    model_uri: str,
    cfg_coef: float,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    env_target = output_root / "env_config.yaml"
    goal_image_target = output_root / "goal_image.png"
    goal_mask_target = output_root / "goal_mask.png"
    for source, target in (
        (Path(env_config_path), env_target),
        (Path(goal_image_path), goal_image_target),
        (Path(goal_mask_path), goal_mask_target),
    ):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    write_json(
        output_root / "goal_spec.json",
        {
            "env_config_path": str(Path(env_config_path).resolve()),
            "goal_image_path": str(Path(goal_image_path).resolve()),
            "goal_mask_path": str(Path(goal_mask_path).resolve()),
            "copied_env_config": str(env_target.resolve()),
            "copied_goal_image": str(goal_image_target.resolve()),
            "copied_goal_mask": str(goal_mask_target.resolve()),
            "segment_type": segment_type,
            "segment_id": SEGMENT_MAPPING[segment_type],
            "model_uri": model_uri,
            "cfg_coef": cfg_coef,
            "obs_size": list(DEFAULT_OBS_SIZE),
        },
    )


def run_rocket2_rollout(
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
    cfg_coef: float = 1.0,
    success_key: str | None = None,
    success_regex: str | None = None,
    success_num: int | None = None,
    tail_window: int = 128,
    max_failures: int = 3,
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

    agent_generator = make_rocket2_agent_generator(model_uri=model_uri, cfg_coef=cfg_coef)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from minestudio.inference import MineGenerator

    generator = MineGenerator(
        num_workers=num_workers,
        num_gpus=num_gpus_per_worker,
        env_generator=env_generator,
        agent_generator=agent_generator,
        num_max_steps=steps,
        num_episodes=episodes,
        tmpdir=str(episodes_dir),
        image_media="h264",
    )

    rollout_episodes = [episode for episode in generator.generate()]
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
