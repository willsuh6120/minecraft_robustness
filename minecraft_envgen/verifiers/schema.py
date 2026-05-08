from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from minecraft_envgen.core.contracts import (
    CompiledEnvironment,
    CurriculumDraft,
    EngineTarget,
    Severity,
    VerificationMessage,
)


ALLOWED_TASK_CLASSES = {"harvest", "combat", "tech_tree", "survival", "creative"}
ALLOWED_BIOMES = {
    "plains",
    "forest",
    "mountains",
    "desert",
    "savanna",
    "taiga",
    "swamp",
    "snowy_taiga",
}
ALLOWED_WEATHER = {"clear", "rain", "thunder", "normal"}
ALLOWED_TIMES = {"dawn", "day", "noon", "dusk", "night"}
ALLOWED_PRIV_OBS = {"inventory", "voxel", "lidar", "events", "position"}
ALLOWED_MINESTUDIO_BIOMES = {
    "forest",
    "nether",
    "taiga",
    "the_end",
    "none",
    "swamp",
    "ocean",
    "mesa",
    "extreme_hills",
    "savanna",
    "plains",
    "beach",
    "jungle",
    "river",
    "desert",
    "mushroom",
    "icy",
}
ALLOWED_MINEDOJO_KEYS = {
    "task_id",
    "image_size",
    "seed",
    "world_seed",
    "fast_reset",
    "event_level_control",
    "use_voxel",
    "voxel_size",
    "use_lidar",
    "lidar_rays",
    "initial_inventory",
    "specified_biome",
    "start_at_night",
    "allow_time_passage",
    "initial_weather",
    "start_health",
    "start_food",
    "start_position",
    "break_speed_multiplier",
}
ALLOWED_MINESTUDIO_CALLBACKS = {
    "CommandsCallback",
    "FastResetCallback",
    "HardResetCallback",
    "SummonMobsCallback",
    "RewardsCallback",
    "VoxelsCallback",
}


class DraftSchemaVerifier:
    def verify(self, draft: CurriculumDraft) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        self._require_nonempty(messages, draft.draft_id, "draft.draft_id")
        self._require_nonempty(messages, draft.target_skill, "draft.target_skill")
        self._require_nonempty(messages, draft.rationale, "draft.rationale")

        if draft.task_class not in ALLOWED_TASK_CLASSES:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.task_class",
                    message=f"Unsupported task_class '{draft.task_class}'.",
                )
            )

        if not isinstance(draft.interventions, dict) or not draft.interventions:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions",
                    message="At least one intervention is required.",
                )
            )
        else:
            messages.extend(self._verify_interventions(draft.interventions))

        for item_name, quantity in draft.initial_inventory.items():
            if not isinstance(quantity, int):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"draft.initial_inventory.{item_name}",
                        message="Inventory quantities must be integers.",
                    )
                )
                continue
            if quantity < 0 or quantity > 64:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"draft.initial_inventory.{item_name}",
                        message="Inventory quantities must be in [0, 64].",
                    )
                )

        for item in draft.privileged_observations:
            if item not in ALLOWED_PRIV_OBS:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.privileged_observations",
                        message=f"Unsupported privileged observation '{item}'.",
                    )
                )
        return messages

    def _require_nonempty(self, messages: List[VerificationMessage], value: str, location: str) -> None:
        if not isinstance(value, str) or not value.strip():
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="Value must be a non-empty string.",
                )
            )

    def _verify_interventions(self, interventions: Mapping[str, Any]) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []

        if "spawn_biome" in interventions and interventions["spawn_biome"] not in ALLOWED_BIOMES:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions.spawn_biome",
                    message=f"Biome must be one of {sorted(ALLOWED_BIOMES)}.",
                )
            )

        if "weather" in interventions and interventions["weather"] not in ALLOWED_WEATHER:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions.weather",
                    message=f"Weather must be one of {sorted(ALLOWED_WEATHER)}.",
                )
            )

        if "time_of_day" in interventions and interventions["time_of_day"] not in ALLOWED_TIMES:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions.time_of_day",
                    message=f"time_of_day must be one of {sorted(ALLOWED_TIMES)}.",
                )
            )

        if "start_health" in interventions:
            health = interventions["start_health"]
            if not isinstance(health, (int, float)) or health <= 0 or health > 20:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.start_health",
                        message="start_health must be in (0, 20].",
                    )
                )

        if "start_food" in interventions:
            food = interventions["start_food"]
            if not isinstance(food, int) or food < 0 or food > 20:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.start_food",
                        message="start_food must be an integer in [0, 20].",
                    )
                )

        if "break_speed_multiplier" in interventions:
            multiplier = interventions["break_speed_multiplier"]
            if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.break_speed_multiplier",
                        message="break_speed_multiplier must be > 0.",
                    )
                )

        if "random_tp_range" in interventions:
            random_tp_range = interventions["random_tp_range"]
            if not isinstance(random_tp_range, int) or random_tp_range <= 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.random_tp_range",
                        message="random_tp_range must be a positive integer.",
                    )
                )

        if "resource_multipliers" in interventions:
            multipliers = interventions["resource_multipliers"]
            if not isinstance(multipliers, dict) or not multipliers:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.resource_multipliers",
                        message="resource_multipliers must be a non-empty object.",
                    )
                )
            else:
                for name, value in multipliers.items():
                    if not isinstance(value, (int, float)) or value <= 0 or value > 10:
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"draft.interventions.resource_multipliers.{name}",
                                message="Each multiplier must be in (0, 10].",
                            )
                        )

        if "start_position" in interventions:
            messages.extend(self._verify_start_position(interventions["start_position"], "draft.interventions.start_position"))

        if "mob_spawns" in interventions:
            mob_spawns = interventions["mob_spawns"]
            if not isinstance(mob_spawns, list) or not mob_spawns:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="draft.interventions.mob_spawns",
                        message="mob_spawns must be a non-empty list.",
                    )
                )
            else:
                for index, mob_spec in enumerate(mob_spawns):
                    messages.extend(self._verify_mob_spawn(mob_spec, f"draft.interventions.mob_spawns[{index}]"))

        return messages

    def _verify_start_position(self, position: Any, location: str) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        if not isinstance(position, dict):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="start_position must be an object.",
                )
            )
            return messages
        for key in ("x", "y", "z"):
            if key not in position or not isinstance(position[key], (int, float)):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.{key}",
                        message="x, y, z must be numeric.",
                    )
                )
        return messages

    def _verify_mob_spawn(self, mob_spec: Any, location: str) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        if not isinstance(mob_spec, dict):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="Each mob spawn must be an object.",
                )
            )
            return messages
        if not isinstance(mob_spec.get("name"), str) or not mob_spec["name"]:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=f"{location}.name",
                    message="Mob name must be a non-empty string.",
                )
            )
        if not isinstance(mob_spec.get("number"), int) or mob_spec["number"] <= 0:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=f"{location}.number",
                    message="Mob number must be a positive integer.",
                )
            )
        for axis in ("range_x", "range_z"):
            value = mob_spec.get(axis)
            if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.{axis}",
                        message="Each spawn range must be a numeric [min, max] pair.",
                    )
                )
        return messages


class AdapterSchemaVerifier:
    def verify(self, compiled: CompiledEnvironment) -> List[VerificationMessage]:
        if compiled.engine == EngineTarget.MINEDOJO:
            return self._verify_minedojo(compiled.runtime)
        if compiled.engine == EngineTarget.MINESTUDIO:
            return self._verify_minestudio(compiled.runtime)
        return [
            VerificationMessage(
                severity=Severity.ERROR,
                location="compiled.engine",
                message=f"Unsupported engine '{compiled.engine.value}'.",
            )
        ]

    def _verify_minedojo(self, runtime: Mapping[str, Any]) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        make_kwargs = runtime.get("make_kwargs")
        if not isinstance(make_kwargs, dict):
            return [
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.make_kwargs",
                    message="MineDojo runtime must include make_kwargs.",
                )
            ]

        for key in make_kwargs:
            if key not in ALLOWED_MINEDOJO_KEYS:
                messages.append(
                    VerificationMessage(
                        severity=Severity.WARNING,
                        location=f"compiled.runtime.make_kwargs.{key}",
                        message="Key is not in the current allowed MineDojo schema whitelist.",
                    )
                )

        self._require_key(messages, make_kwargs, "task_id", "compiled.runtime.make_kwargs.task_id")
        self._verify_image_size(messages, make_kwargs.get("image_size"), "compiled.runtime.make_kwargs.image_size")
        self._verify_bool(messages, make_kwargs, "fast_reset")
        self._verify_bool(messages, make_kwargs, "event_level_control")
        self._verify_bool(messages, make_kwargs, "use_voxel")
        self._verify_bool(messages, make_kwargs, "use_lidar")

        if "specified_biome" in make_kwargs and make_kwargs["specified_biome"] not in ALLOWED_BIOMES:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.make_kwargs.specified_biome",
                    message=f"Biome must be one of {sorted(ALLOWED_BIOMES)}.",
                )
            )

        if "initial_weather" in make_kwargs and make_kwargs["initial_weather"] not in ALLOWED_WEATHER:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.make_kwargs.initial_weather",
                    message=f"Weather must be one of {sorted(ALLOWED_WEATHER)}.",
                )
            )

        if "start_health" in make_kwargs:
            value = make_kwargs["start_health"]
            if not isinstance(value, (int, float)) or value <= 0 or value > 20:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.make_kwargs.start_health",
                        message="start_health must be in (0, 20].",
                    )
                )

        if "start_food" in make_kwargs:
            value = make_kwargs["start_food"]
            if not isinstance(value, int) or value < 0 or value > 20:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.make_kwargs.start_food",
                        message="start_food must be an integer in [0, 20].",
                    )
                )

        if "break_speed_multiplier" in make_kwargs:
            value = make_kwargs["break_speed_multiplier"]
            if not isinstance(value, (int, float)) or value <= 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.make_kwargs.break_speed_multiplier",
                        message="break_speed_multiplier must be > 0.",
                    )
                )

        if "start_position" in make_kwargs:
            messages.extend(
                DraftSchemaVerifier()._verify_start_position(
                    make_kwargs["start_position"], "compiled.runtime.make_kwargs.start_position"
                )
            )

        if "voxel_size" in make_kwargs:
            messages.extend(self._verify_voxel_size(make_kwargs["voxel_size"], "compiled.runtime.make_kwargs.voxel_size"))

        if "lidar_rays" in make_kwargs:
            lidar_rays = make_kwargs["lidar_rays"]
            if not isinstance(lidar_rays, list) or not all(
                isinstance(ray, (list, tuple)) and len(ray) == 3 and all(isinstance(item, (int, float)) for item in ray)
                for ray in lidar_rays
            ):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.make_kwargs.lidar_rays",
                        message="lidar_rays must be a list of numeric (pitch, yaw, distance) triples.",
                    )
                )

        if "initial_inventory" in make_kwargs:
            inventory = make_kwargs["initial_inventory"]
            if not isinstance(inventory, list):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.make_kwargs.initial_inventory",
                        message="initial_inventory must be a list of inventory item objects.",
                    )
                )
            else:
                for index, item in enumerate(inventory):
                    if not isinstance(item, dict):
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"compiled.runtime.make_kwargs.initial_inventory[{index}]",
                                message="Each initial inventory item must be an object.",
                            )
                        )
                        continue
                    if not isinstance(item.get("name"), str) or not item["name"]:
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"compiled.runtime.make_kwargs.initial_inventory[{index}].name",
                                message="Inventory item name must be a non-empty string.",
                            )
                        )
                    if not isinstance(item.get("quantity"), int) or item["quantity"] <= 0:
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"compiled.runtime.make_kwargs.initial_inventory[{index}].quantity",
                                message="Inventory item quantity must be a positive integer.",
                            )
                        )

        return messages

    def _verify_minestudio(self, runtime: Mapping[str, Any]) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        sim_kwargs = runtime.get("sim_kwargs")
        callbacks = runtime.get("callbacks")

        if not isinstance(sim_kwargs, dict):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.sim_kwargs",
                    message="MineStudio runtime must include sim_kwargs.",
                )
            )
            return messages

        action_type = sim_kwargs.get("action_type")
        if action_type not in {"env", "agent"}:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.sim_kwargs.action_type",
                    message="action_type must be 'env' or 'agent'.",
                )
            )

        self._verify_image_size(messages, sim_kwargs.get("obs_size"), "compiled.runtime.sim_kwargs.obs_size")

        if "preferred_spawn_biome" in sim_kwargs and sim_kwargs["preferred_spawn_biome"] not in ALLOWED_MINESTUDIO_BIOMES:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.sim_kwargs.preferred_spawn_biome",
                    message=f"Biome must be one of {sorted(ALLOWED_MINESTUDIO_BIOMES)}.",
                )
            )

        if "timestep_limit" in sim_kwargs:
            timestep_limit = sim_kwargs["timestep_limit"]
            if not isinstance(timestep_limit, int) or timestep_limit <= 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location="compiled.runtime.sim_kwargs.timestep_limit",
                        message="timestep_limit must be a positive integer.",
                    )
                )

        if not isinstance(callbacks, list):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="compiled.runtime.callbacks",
                    message="callbacks must be a list.",
                )
            )
            return messages

        for index, callback in enumerate(callbacks):
            if not isinstance(callback, dict):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"compiled.runtime.callbacks[{index}]",
                        message="Each callback must be an object.",
                    )
                )
                continue
            name = callback.get("name")
            config = callback.get("config")
            if name not in ALLOWED_MINESTUDIO_CALLBACKS:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"compiled.runtime.callbacks[{index}].name",
                        message=f"Unsupported callback '{name}'.",
                    )
                )
                continue
            if not isinstance(config, dict):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"compiled.runtime.callbacks[{index}].config",
                        message="Callback config must be an object.",
                    )
                )
                continue
            messages.extend(self._verify_callback(name, config, f"compiled.runtime.callbacks[{index}].config"))

        return messages

    def _verify_callback(self, name: str, config: Mapping[str, Any], location: str) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        if name == "CommandsCallback":
            commands = config.get("commands")
            if not isinstance(commands, list) or not commands or not all(isinstance(item, str) and item for item in commands):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.commands",
                        message="CommandsCallback requires a non-empty list of commands.",
                    )
                )
        elif name == "FastResetCallback":
            biomes = config.get("biomes")
            random_tp_range = config.get("random_tp_range")
            start_time = config.get("start_time")
            start_weather = config.get("start_weather")
            if not isinstance(biomes, list) or not biomes or not all(isinstance(item, str) for item in biomes):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.biomes",
                        message="FastResetCallback requires a non-empty list of biome strings.",
                    )
                )
            elif not all(item in ALLOWED_MINESTUDIO_BIOMES for item in biomes):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.biomes",
                        message=f"FastResetCallback biomes must be drawn from {sorted(ALLOWED_MINESTUDIO_BIOMES)}.",
                    )
                )
            if not isinstance(random_tp_range, int) or random_tp_range <= 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.random_tp_range",
                        message="random_tp_range must be a positive integer.",
                    )
                )
            if not isinstance(start_time, int) or start_time < 0 or start_time > 24000:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.start_time",
                        message="start_time must be an integer in [0, 24000].",
                    )
                )
            if start_weather not in {"clear", "rain", "thunder"}:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.start_weather",
                        message="start_weather must be clear, rain, or thunder.",
                    )
                )
        elif name == "HardResetCallback":
            spawn_positions = config.get("spawn_positions")
            if not isinstance(spawn_positions, list) or not spawn_positions:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.spawn_positions",
                        message="HardResetCallback requires a non-empty list of spawn positions.",
                    )
                )
            else:
                for index, item in enumerate(spawn_positions):
                    if not isinstance(item, dict):
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"{location}.spawn_positions[{index}]",
                                message="Each spawn position must be an object.",
                            )
                        )
                        continue
                    if not isinstance(item.get("seed"), int):
                        messages.append(
                            VerificationMessage(
                                severity=Severity.ERROR,
                                location=f"{location}.spawn_positions[{index}].seed",
                                message="Each spawn position requires an integer seed.",
                            )
                        )
                    position = item.get("position")
                    if not isinstance(position, list) or len(position) != 3 or not all(
                        isinstance(value, (int, float)) for value in position
                    ):
                        messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.spawn_positions[{index}].position",
                        message="Each position must be a numeric [x, y, z] list.",
                    )
                )
        elif name == "SummonMobsCallback":
            mobs = config.get("mobs")
            if not isinstance(mobs, list) or not mobs:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.mobs",
                        message="SummonMobsCallback requires a non-empty list of mob specs.",
                    )
                )
            else:
                for index, mob in enumerate(mobs):
                    messages.extend(
                        DraftSchemaVerifier()._verify_mob_spawn(mob, f"{location}.mobs[{index}]")
                    )
        elif name == "RewardsCallback":
            rewards = config.get("rewards")
            if not isinstance(rewards, list) or not rewards:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.rewards",
                        message="RewardsCallback requires a non-empty rewards list.",
                    )
                )
        return messages

    def _require_key(
        self,
        messages: List[VerificationMessage],
        mapping: Mapping[str, Any],
        key: str,
        location: str,
    ) -> None:
        if key not in mapping:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message=f"Missing required key '{key}'.",
                )
            )

    def _verify_image_size(
        self, messages: List[VerificationMessage], value: Any, location: str
    ) -> None:
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(item, int) for item in value):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="Image size must be a pair of integers.",
                )
            )
            return
        if any(item < 32 or item > 2048 for item in value):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="Image dimensions must stay in [32, 2048].",
                )
            )

    def _verify_bool(self, messages: List[VerificationMessage], mapping: Mapping[str, Any], key: str) -> None:
        if key in mapping and not isinstance(mapping[key], bool):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=f"compiled.runtime.make_kwargs.{key}",
                    message=f"{key} must be a boolean.",
                )
            )

    def _verify_voxel_size(self, value: Any, location: str) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        if not isinstance(value, dict):
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location=location,
                    message="voxel_size must be an object.",
                )
            )
            return messages
        required = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        for key in required:
            if key not in value or not isinstance(value[key], int):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"{location}.{key}",
                        message="Each voxel bound must be an integer.",
                    )
                )
        return messages
