from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from minecraft_envgen.rocket2_eval import ensure_minestudio_engine, ensure_virtualgl_session
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import load_callbacks_from_config


DEFAULT_ENV_CONF_DIR = "/home/gyulab/envgen2/ROCKET-2/env_conf"
DEFAULT_OUT_DIR = "/home/gyulab/envgen2/outputs/rocket2_step0_dump"


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-conf-dir", type=str, default=DEFAULT_ENV_CONF_DIR)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--display", type=str, default=":1")
    parser.add_argument("--obs-width", type=int, default=224)
    parser.add_argument("--obs-height", type=int, default=224)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def discover_env_files(env_conf_dir: Path, tasks_arg: str) -> List[Path]:
    if tasks_arg.strip():
        requested = [task.strip() for task in tasks_arg.split(",") if task.strip()]
        files = []
        for task in requested:
            path = env_conf_dir / f"{task}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Task env config not found: {path}")
            files.append(path)
        return files
    return sorted(env_conf_dir.glob("*.yaml"))


def save_frame(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    if args.display:
        import os

        os.environ["DISPLAY"] = args.display
    ensure_virtualgl_session()
    ensure_minestudio_engine()

    env_conf_dir = Path(args.env_conf_dir)
    out_root = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    env_files = discover_env_files(env_conf_dir, args.tasks)
    summary_rows: List[Dict[str, Any]] = []

    for env_file in env_files:
        env_data = yaml.safe_load(env_file.read_text()) or {}
        task_dir = out_root / env_file.stem
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "env_config.yaml").write_text(env_file.read_text(), encoding="utf-8")

        for episode_idx in range(args.episodes_per_task):
            callbacks = list(load_callbacks_from_config(str(env_file)))
            sim = MinecraftSim(
                callbacks=callbacks,
                obs_size=(args.obs_height, args.obs_width),
            )
            try:
                obs, info = sim.reset()
                raw_frame = info["pov"]
                resized_frame = obs["image"]
                episode_dir = task_dir / f"reset_{episode_idx:03d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                save_frame(episode_dir / "step0_pov.png", raw_frame)
                save_frame(episode_dir / "step0_obs.png", resized_frame)

                row = {
                    "task": env_file.stem,
                    "episode_index": episode_idx,
                    "env_config": str(env_file.resolve()),
                    "spawn_positions_defined": len(env_data.get("spawn_positions", []) or []),
                    "player_pos": sanitize(info.get("player_pos") or info.get("location_stats") or {}),
                    "inventory": sanitize(info.get("inventory", {})),
                    "output_dir": str(episode_dir.resolve()),
                }
                (episode_dir / "metadata.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                summary_rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
            finally:
                sim.close()

    (out_root / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {out_root}")


if __name__ == "__main__":
    main()
