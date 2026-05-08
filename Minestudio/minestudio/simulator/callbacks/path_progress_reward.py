from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple, Union

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers


def _player_pos(info: Mapping) -> Tuple[float, float]:
    player_pos = info.get("player_pos")
    if isinstance(player_pos, Mapping):
        return float(player_pos.get("x", 0.0)), float(player_pos.get("z", 0.0))
    location_stats = info.get("location_stats")
    if isinstance(location_stats, Mapping):
        return float(location_stats.get("xpos", 0.0)), float(location_stats.get("zpos", 0.0))
    return 0.0, 0.0


@Registers.simulator_callback.register
class PathProgressRewardCallback(MinecraftCallback):
    def create_from_conf(source: Union[str, Dict]):
        data = MinecraftCallback.load_data_from_conf(source)
        reward_cfg = data.get("path_progress_reward")
        if isinstance(reward_cfg, dict) and reward_cfg.get("zones"):
            return PathProgressRewardCallback(reward_cfg)
        return None

    def __init__(self, reward_cfg: Mapping[str, object]):
        super().__init__()
        self.reward_cfg = dict(reward_cfg)
        self.zones = []
        for zone in self.reward_cfg.get("zones") or []:
            if not isinstance(zone, Mapping):
                continue
            cells = {
                (int(cell[0]), int(cell[1]))
                for cell in zone.get("cells_local") or []
                if isinstance(cell, Sequence) and len(cell) >= 2
            }
            if not cells:
                continue
            self.zones.append(
                {
                    "id": str(zone.get("id") or f"zone_{len(self.zones)}"),
                    "label": str(zone.get("label") or ""),
                    "reward": float(zone.get("reward", 0.0) or 0.0),
                    "cells": cells,
                }
            )
        self.origin_x = 0.0
        self.origin_z = 0.0
        self.next_zone_index = 0

    def after_reset(self, sim, obs, info):
        self.origin_x, self.origin_z = _player_pos(info)
        self.next_zone_index = 0
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        shaped_reward = 0.0
        if self.next_zone_index < len(self.zones):
            player_x, player_z = _player_pos(info)
            local_cell = (
                int(round(float(player_x) - float(self.origin_x))),
                int(round(float(player_z) - float(self.origin_z))),
            )
            zone = self.zones[self.next_zone_index]
            if local_cell in zone["cells"]:
                shaped_reward = float(zone["reward"])
                self.next_zone_index += 1
                reward_debug = {
                    "zone_id": str(zone["id"]),
                    "zone_label": str(zone["label"]),
                    "local_cell": [int(local_cell[0]), int(local_cell[1])],
                    "reward": float(shaped_reward),
                    "next_zone_index": int(self.next_zone_index),
                }
                if isinstance(info, dict):
                    info["path_progress_reward"] = reward_debug
        return obs, float(reward) + float(shaped_reward), terminated, truncated, info
