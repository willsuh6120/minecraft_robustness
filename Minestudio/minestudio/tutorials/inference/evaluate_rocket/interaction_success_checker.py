from dataclasses import dataclass
from math import sqrt
from typing import Dict, Mapping, Optional

import numpy as np

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import InteractionBenchmarkTaskSpec
from minestudio.tutorials.inference.evaluate_rocket.success_checker import (
    _to_scalar,
    event_delta,
    inventory_delta,
    inventory_totals,
)


@dataclass
class InteractionSuccessResult:
    supported: bool
    success: Optional[bool]
    metric: str
    progress: Dict[str, float]
    reason: str


def _player_pos(info: Mapping):
    player_pos = info.get("player_pos")
    if isinstance(player_pos, Mapping):
        return {
            "x": float(player_pos.get("x", 0.0)),
            "y": float(player_pos.get("y", 0.0)),
            "z": float(player_pos.get("z", 0.0)),
        }
    location_stats = info.get("location_stats")
    if isinstance(location_stats, Mapping):
        return {
            "x": float(location_stats.get("xpos", 0.0)),
            "y": float(location_stats.get("ypos", 0.0)),
            "z": float(location_stats.get("zpos", 0.0)),
        }
    return {"x": 0.0, "y": 0.0, "z": 0.0}


class InteractionBenchmarkSuccessTracker:
    def __init__(self, task_spec: InteractionBenchmarkTaskSpec, initial_info: Mapping):
        self.task_spec = task_spec
        self.initial_info = dict(initial_info)
        self.initial_inventory = inventory_totals(initial_info)
        self.initial_player_pos = _player_pos(initial_info)

    def update(self, info: Mapping) -> InteractionSuccessResult:
        kind = self.task_spec.auto_eval_kind

        if kind == "mine_emerald":
            mined = event_delta(info, self.initial_info, "mine_block", ["emerald_ore", "emerald_block"])
            emerald = inventory_delta(info, self.initial_inventory, ["emerald"])
            return InteractionSuccessResult(
                supported=True,
                success=(mined > 0 or emerald > 0),
                metric="emerald_mined",
                progress={"mine_block": mined, "inventory_delta": emerald},
                reason="Success when emerald ore is mined or emerald appears in inventory.",
            )

        if kind == "mine_coal":
            mined = event_delta(info, self.initial_info, "mine_block", ["coal_ore", "coal_block"])
            coal = inventory_delta(info, self.initial_inventory, ["coal"])
            return InteractionSuccessResult(
                supported=True,
                success=(mined > 0 or coal > 0),
                metric="coal_mined",
                progress={"mine_block": mined, "inventory_delta": coal},
                reason="Success when coal ore is mined or coal appears in inventory.",
            )

        if kind == "hunt_cow_without_sheep":
            cow_kills = event_delta(info, self.initial_info, "kill_entity", ["cow"])
            sheep_kills = event_delta(info, self.initial_info, "kill_entity", ["sheep"])
            return InteractionSuccessResult(
                supported=True,
                success=(cow_kills > 0 and sheep_kills == 0),
                metric="cow_hunted_without_sheep",
                progress={"cow_kills": cow_kills, "sheep_kills": sheep_kills},
                reason="Success when a cow is killed and no sheep are killed.",
            )

        if kind == "approach_target":
            pos = _player_pos(info)
            if self.task_spec.target_center is None or self.task_spec.success_radius is None:
                return InteractionSuccessResult(
                    supported=False,
                    success=None,
                    metric="unsupported",
                    progress={},
                    reason="Missing target center metadata for approach task.",
                )
            target_x, _, target_z = self.task_spec.target_center
            target_x = self.initial_player_pos["x"] + target_x
            target_z = self.initial_player_pos["z"] + target_z
            distance = sqrt((pos["x"] - target_x) ** 2 + (pos["z"] - target_z) ** 2)
            return InteractionSuccessResult(
                supported=True,
                success=(distance <= float(self.task_spec.success_radius)),
                metric="distance_to_target",
                progress={"distance": round(distance, 4)},
                reason="Success when the player is within the configured radius of the target center.",
            )

        if kind == "lava_bucket":
            lava_bucket = inventory_delta(info, self.initial_inventory, ["lava_bucket"])
            return InteractionSuccessResult(
                supported=True,
                success=(lava_bucket > 0),
                metric="lava_bucket_collected",
                progress={"lava_bucket_delta": lava_bucket},
                reason="Success when a lava bucket appears in inventory.",
            )

        if kind == "place_minecart":
            current_inventory = inventory_totals(info)
            before = sum(v for k, v in self.initial_inventory.items() if "minecart" in k)
            after = sum(v for k, v in current_inventory.items() if "minecart" in k)
            use_count = event_delta(info, self.initial_info, "use_item", ["minecart"])
            return InteractionSuccessResult(
                supported=True,
                success=(use_count > 0 and after < before),
                metric="minecart_placed",
                progress={"use_item": use_count, "minecart_inventory_before": before, "minecart_inventory_after": after},
                reason="Success when minecart use is recorded and minecart inventory decreases.",
            )

        return InteractionSuccessResult(
            supported=False,
            success=None,
            metric="manual_review_required",
            progress={},
            reason="This benchmark task currently requires manual review via saved video and trajectory logs.",
        )
