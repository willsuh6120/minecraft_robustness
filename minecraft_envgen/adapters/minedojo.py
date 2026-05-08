from __future__ import annotations

from typing import Any, Dict, List

from minecraft_envgen.adapters.base import EnvironmentAdapter
from minecraft_envgen.core.contracts import CompiledEnvironment, CurriculumDraft, EngineTarget, Split


ITEM_ALIASES = {
    "wood_pickaxe": "wooden_pickaxe",
    "wood_sword": "wooden_sword",
}


class MineDojoAdapter(EnvironmentAdapter):
    engine = EngineTarget.MINEDOJO
    dependency_name = "minedojo"

    def compile(self, draft: CurriculumDraft, split: Split) -> CompiledEnvironment:
        reset_mode = self.default_reset_mode(split)
        use_voxel = "voxel" in draft.privileged_observations
        use_lidar = "lidar" in draft.privileged_observations

        make_kwargs: Dict[str, Any] = {
            "task_id": draft.metadata.get("task_id", draft.target_skill),
            "image_size": draft.metadata.get("image_size", [160, 256]),
            "seed": int(draft.metadata.get("seed", 0)),
            "world_seed": str(draft.metadata.get("world_seed", draft.metadata.get("seed", 0))),
            "fast_reset": reset_mode.value == "fast",
            "event_level_control": True,
            "use_voxel": use_voxel,
            "use_lidar": use_lidar,
            "initial_inventory": self._inventory_to_items(draft.initial_inventory),
            "specified_biome": draft.interventions.get("spawn_biome"),
            "start_at_night": draft.interventions.get("time_of_day") == "night",
            "allow_time_passage": draft.interventions.get("allow_time_passage", True),
            "initial_weather": draft.interventions.get("weather"),
            "start_health": draft.interventions.get("start_health"),
            "start_food": draft.interventions.get("start_food"),
            "start_position": draft.interventions.get("start_position"),
            "break_speed_multiplier": draft.interventions.get("break_speed_multiplier", 1.0),
        }

        if use_voxel:
            make_kwargs["voxel_size"] = draft.metadata.get(
                "voxel_size",
                {"xmin": -4, "xmax": 4, "ymin": -4, "ymax": 4, "zmin": -4, "zmax": 4},
            )
        if use_lidar:
            make_kwargs["lidar_rays"] = draft.metadata.get(
                "lidar_rays",
                [(0.0, 0.0, 10.0), (0.0, 1.57, 10.0), (0.0, -1.57, 10.0)],
            )

        make_kwargs = {key: value for key, value in make_kwargs.items() if value is not None}

        runtime = {
            "dependency": self.dependency_status(),
            "make_kwargs": make_kwargs,
            "adapter_payload": {
                "target_skill": draft.target_skill,
                "task_class": draft.task_class,
                "interventions": draft.interventions,
                "privileged_observations": draft.privileged_observations,
            },
        }
        notes = [
            "MineDojo runtime args are kept conservative and close to documented shared kwargs.",
            "Curriculum interventions that are not first-class MineDojo kwargs remain in adapter_payload until task-family-specific bindings are added.",
            "task_id defaults to target_skill if no explicit metadata.task_id is supplied.",
        ]
        return CompiledEnvironment(
            engine=self.engine,
            split=split,
            reset_mode=reset_mode,
            runtime=runtime,
            notes=notes,
        )

    def _inventory_to_items(self, inventory: Dict[str, int]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for name, quantity in inventory.items():
            if quantity <= 0:
                continue
            items.append({"name": ITEM_ALIASES.get(name, name), "quantity": int(quantity)})
        return items
