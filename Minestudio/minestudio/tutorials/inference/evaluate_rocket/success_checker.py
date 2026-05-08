from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np


def _to_scalar(value) -> float:
    if isinstance(value, np.ndarray):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def normalize_name(name: str) -> str:
    return str(name).lower().replace("minecraft:", "").strip()


def inventory_totals(info: Mapping) -> Dict[str, int]:
    inventory = info.get("inventory", {}) or {}
    totals: Counter = Counter()
    values = inventory.values() if isinstance(inventory, Mapping) else inventory
    for slot in values:
        if not isinstance(slot, Mapping):
            continue
        item_type = normalize_name(slot.get("type", ""))
        quantity = int(_to_scalar(slot.get("quantity", 0)))
        if item_type and item_type != "air" and quantity > 0:
            totals[item_type] += quantity
    return dict(totals)


def inventory_delta(info: Mapping, initial_inventory: Mapping[str, int], aliases: Iterable[str]) -> int:
    aliases = tuple(normalize_name(alias) for alias in aliases)
    current = inventory_totals(info)
    delta = 0
    for item_name, quantity in current.items():
        if any(alias in item_name for alias in aliases):
            delta += quantity - int(initial_inventory.get(item_name, 0))
    return delta


def event_delta(info: Mapping, initial_info: Mapping, event_key: str, aliases: Iterable[str]) -> int:
    aliases = tuple(normalize_name(alias) for alias in aliases)
    current_events = info.get(event_key, {}) or {}
    initial_events = initial_info.get(event_key, {}) or {}
    delta = 0
    for item_name, value in current_events.items():
        normalized = normalize_name(item_name)
        if any(alias in normalized for alias in aliases):
            delta += int(_to_scalar(value) - _to_scalar(initial_events.get(item_name, 0)))
    return delta


@dataclass
class SuccessResult:
    success: bool
    metric: str
    progress: Dict[str, int]
    reason: str


class TaskSuccessTracker:
    def __init__(self, task_name: str, initial_info: Mapping):
        self.task_name = task_name
        self.initial_info = dict(initial_info)
        self.initial_inventory = inventory_totals(initial_info)

    def _result(self, success: bool, metric: str, progress: Dict[str, int], reason: str) -> SuccessResult:
        return SuccessResult(success=success, metric=metric, progress=progress, reason=reason)

    def update(self, info: Mapping) -> SuccessResult:
        task_name = self.task_name.lower()

        if task_name == "collect_wood":
            mined = event_delta(info, self.initial_info, "mine_block", ["oak_log", "_log", "log"])
            gained = inventory_delta(info, self.initial_inventory, ["oak_log", "_log", "log"])
            return self._result(
                success=(mined > 0 or gained > 0),
                metric="wood_collected",
                progress={"mine_block": mined, "inventory_delta": gained},
                reason="Success when at least one log is mined or added to inventory.",
            )

        if task_name == "hunt_a_sheep":
            killed = event_delta(info, self.initial_info, "kill_entity", ["sheep"])
            wool = inventory_delta(info, self.initial_inventory, ["wool"])
            mutton = inventory_delta(info, self.initial_inventory, ["mutton"])
            return self._result(
                success=(killed > 0 or wool > 0 or mutton > 0),
                metric="sheep_hunted",
                progress={"kill_entity": killed, "wool_delta": wool, "mutton_delta": mutton},
                reason="Success when the sheep is killed or sheep drops are collected.",
            )

        if task_name == "hunt_animals":
            killed = event_delta(info, self.initial_info, "kill_entity", ["sheep", "cow", "pig", "chicken"])
            loot = inventory_delta(info, self.initial_inventory, ["wool", "mutton", "beef", "porkchop", "chicken"])
            return self._result(
                success=(killed > 0 or loot > 0),
                metric="animal_hunted",
                progress={"kill_entity": killed, "loot_delta": loot},
                reason="Success when any target animal is killed or loot is collected.",
            )

        if task_name == "collect_wool":
            wool = inventory_delta(info, self.initial_inventory, ["wool"])
            return self._result(
                success=(wool > 0),
                metric="wool_collected",
                progress={"wool_delta": wool},
                reason="Success when wool appears in inventory.",
            )

        if task_name == "mine_obsidian":
            mined = event_delta(info, self.initial_info, "mine_block", ["obsidian"])
            obsidian = inventory_delta(info, self.initial_inventory, ["obsidian"])
            return self._result(
                success=(mined > 0 or obsidian > 0),
                metric="obsidian_mined",
                progress={"mine_block": mined, "inventory_delta": obsidian},
                reason="Success when obsidian is mined or collected.",
            )

        if task_name == "mine_diamond_ore":
            mined = event_delta(info, self.initial_info, "mine_block", ["diamond_ore"])
            diamond = inventory_delta(info, self.initial_inventory, ["diamond"])
            return self._result(
                success=(mined > 0 or diamond > 0),
                metric="diamond_ore_mined",
                progress={"mine_block": mined, "inventory_delta": diamond},
                reason="Success when diamond ore is mined or diamond is collected.",
            )

        if task_name in {"craft_table", "craft_the_crafting_table"}:
            crafted = event_delta(info, self.initial_info, "craft_item", ["crafting_table"])
            table = inventory_delta(info, self.initial_inventory, ["crafting_table"])
            return self._result(
                success=(crafted > 0 or table > 0),
                metric="crafting_table_crafted",
                progress={"craft_item": crafted, "inventory_delta": table},
                reason="Success when a crafting table is crafted or appears in inventory.",
            )

        return self._result(
            success=False,
            metric="unsupported_task",
            progress={},
            reason="No automatic success rule defined for this task yet.",
        )
