from __future__ import annotations

from typing import Any, Dict, List

from minecraft_envgen.adapters.base import EnvironmentAdapter
from minecraft_envgen.core.contracts import CompiledEnvironment, CurriculumDraft, EngineTarget, Split


TIME_TO_TICKS = {
    "dawn": 0,
    "day": 1000,
    "noon": 6000,
    "dusk": 12000,
    "night": 13000,
}

ITEM_ALIASES = {
    "wood_pickaxe": "wooden_pickaxe",
    "wood_sword": "wooden_sword",
}

MINESTUDIO_BIOME_ALIASES = {
    "mountains": "extreme_hills",
    "mountain": "extreme_hills",
    "beaches": "beach",
    "beach": "beach",
    "snowy_taiga": "icy",
}


class MineStudioAdapter(EnvironmentAdapter):
    engine = EngineTarget.MINESTUDIO
    dependency_name = "minestudio"

    def compile(self, draft: CurriculumDraft, split: Split) -> CompiledEnvironment:
        reset_mode = self.default_reset_mode(split)
        sim_kwargs: Dict[str, Any] = {
            "action_type": draft.metadata.get("action_type", "agent"),
            "obs_size": draft.metadata.get("image_size", [128, 128]),
            "preferred_spawn_biome": self._runtime_biome(draft.interventions.get("spawn_biome", "plains")),
            "timestep_limit": int(draft.metadata.get("timestep_limit", 1000)),
        }

        callbacks: List[Dict[str, Any]] = []
        commands = self._build_commands(draft)
        if commands:
            callbacks.append(
                {
                    "name": "CommandsCallback",
                    "config": {"commands": commands},
                }
            )

        if draft.interventions.get("mob_spawns"):
            callbacks.append(
                {
                    "name": "SummonMobsCallback",
                    "config": {"mobs": draft.interventions["mob_spawns"]},
                }
            )

        if "voxel" in draft.privileged_observations:
            callbacks.append(
                {
                    "name": "VoxelsCallback",
                    "config": {"voxels_ins": self._voxel_bounds(draft)},
                }
            )

        if split == Split.TRAIN:
            callbacks.append(
                {
                    "name": "FastResetCallback",
                    "config": {
                        "biomes": [self._runtime_biome(draft.interventions.get("spawn_biome", "plains"))],
                        "random_tp_range": int(draft.interventions.get("random_tp_range", 1000)),
                        "start_time": TIME_TO_TICKS.get(draft.interventions.get("time_of_day", "day"), 1000),
                        "start_weather": draft.interventions.get("weather", "clear"),
                    },
                }
            )
        else:
            callbacks.append(
                {
                    "name": "HardResetCallback",
                    "config": {
                        "spawn_positions": [
                            {
                                "seed": int(draft.metadata.get("seed", 0)),
                                "position": self._spawn_position(draft),
                            }
                        ]
                    },
                }
            )

        reward_specs = draft.metadata.get("reward_specs")
        if reward_specs:
            callbacks.append(
                {
                    "name": "RewardsCallback",
                    "config": {"rewards": reward_specs},
                }
            )

        runtime = {
            "dependency": self.dependency_status(),
            "sim_kwargs": sim_kwargs,
            "callbacks": callbacks,
            "adapter_payload": {
                "target_skill": draft.target_skill,
                "task_class": draft.task_class,
                "privileged_observations": list(draft.privileged_observations),
                "unbound_interventions": {
                    key: value
                    for key, value in draft.interventions.items()
                    if key not in {"spawn_biome", "weather", "time_of_day", "mob_spawns", "random_tp_range"}
                },
            },
        }
        notes = [
            "MineStudio compiles train resets to FastResetCallback and eval resets to HardResetCallback by default.",
            "Inventory bootstrap is translated into CommandsCallback /give commands.",
            "Interventions that are not cleanly expressible as built-in callbacks remain in adapter_payload.",
        ]
        return CompiledEnvironment(
            engine=self.engine,
            split=split,
            reset_mode=reset_mode,
            runtime=runtime,
            notes=notes,
        )

    def _build_commands(self, draft: CurriculumDraft) -> List[str]:
        commands: List[str] = []
        time_of_day = draft.interventions.get("time_of_day")
        weather = draft.interventions.get("weather")

        if time_of_day:
            commands.append(f"/time set {time_of_day}")
        if weather:
            commands.append(f"/weather {weather}")

        for item_name, quantity in draft.initial_inventory.items():
            if quantity <= 0:
                continue
            commands.append(f"/give @p minecraft:{ITEM_ALIASES.get(item_name, item_name)} {int(quantity)}")

        if draft.intervention_family.value == "safety_scaffold":
            commands.append("/effect @p minecraft:resistance 10 1 true")

        return commands

    def _spawn_position(self, draft: CurriculumDraft) -> List[float]:
        position = draft.interventions.get("start_position")
        if isinstance(position, dict):
            return [
                float(position.get("x", 0.0)),
                float(position.get("y", 64.0)),
                float(position.get("z", 0.0)),
            ]
        return [0.0, 64.0, 0.0]

    def _voxel_bounds(self, draft: CurriculumDraft) -> List[int]:
        voxel_size = draft.metadata.get("voxel_size")
        if isinstance(voxel_size, dict):
            return [
                int(voxel_size.get("xmin", -7)),
                int(voxel_size.get("xmax", 7)),
                int(voxel_size.get("ymin", -7)),
                int(voxel_size.get("ymax", 7)),
                int(voxel_size.get("zmin", -7)),
                int(voxel_size.get("zmax", 7)),
            ]
        return [-7, 7, -7, 7, -7, 7]

    def _runtime_biome(self, biome: str) -> str:
        return MINESTUDIO_BIOME_ALIASES.get(biome, biome)
