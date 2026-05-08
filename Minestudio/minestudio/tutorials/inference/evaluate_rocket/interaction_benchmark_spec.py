from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ScriptedSubtask:
    target_text: str
    interaction_type: str
    step_budget: int
    reprompt_interval: int = 30
    prompt_text: Optional[str] = None
    verification_region: str = "any"
    max_point_retries: int = 1


@dataclass(frozen=True)
class InteractionBenchmarkTaskSpec:
    benchmark_name: str
    task_config_name: str
    category: str
    subtasks: List[ScriptedSubtask]
    task_key: str = ""
    official_task_config_name: Optional[str] = None
    auto_eval_kind: str = "unsupported"
    target_center: Optional[Tuple[float, float, float]] = None
    success_radius: Optional[float] = None
    notes: str = ""


INTERACTION_BENCHMARK_TASKS: Dict[str, InteractionBenchmarkTaskSpec] = {
    "hunt_sheep_right_fence": InteractionBenchmarkTaskSpec(
        benchmark_name="hunt the sheep in the right fence",
        task_config_name="hunt_sheep_right_fence",
        official_task_config_name="hunt_fence",
        category="Hunt",
        subtasks=[
            ScriptedSubtask(
                "sheep in the right fence",
                "Hunt",
                120,
                prompt_text=(
                    "Point to the sheep standing inside the fenced pen on the right. "
                    "Choose the sheep in the right pen, not the sheep in the left pen. "
                    "Place the point on the sheep's body."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="unsupported",
    ),
    "hunt_cow_do_not_touch_sheep": InteractionBenchmarkTaskSpec(
        benchmark_name="hunt the cow while do not touch the sheep",
        task_config_name="hunt_cow_do_not_touch_sheep",
        official_task_config_name="hunt_cowonly",
        category="Hunt",
        subtasks=[
            ScriptedSubtask(
                "cow",
                "Hunt",
                120,
                prompt_text=(
                    "Point to the cow. "
                    "Do not point to any sheep. "
                    "Place the point on the cow's body."
                ),
                verification_region="any",
            )
        ],
        auto_eval_kind="hunt_cow_without_sheep",
    ),
    "mine_emerald": InteractionBenchmarkTaskSpec(
        benchmark_name="mine the emerald",
        task_config_name="mine_emerald",
        official_task_config_name="mine_emerald",
        category="Mine",
        subtasks=[
            ScriptedSubtask(
                "emerald ore",
                "Mine",
                90,
                prompt_text=(
                    "Point to the emerald ore block on the ground. "
                    "Choose the green-speckled ore block, not the coal ore block. "
                    "Place the point near the center of the emerald ore block face."
                ),
                verification_region="any",
            )
        ],
        auto_eval_kind="mine_emerald",
    ),
    "mine_coal": InteractionBenchmarkTaskSpec(
        benchmark_name="mine the coal",
        task_config_name="mine_coal",
        official_task_config_name="mine_coal",
        category="Mine",
        subtasks=[
            ScriptedSubtask(
                "coal ore",
                "Mine",
                90,
                prompt_text=(
                    "Point to the coal ore block on the ground. "
                    "Choose the dark coal ore block, not the emerald ore block. "
                    "Place the point near the center of the coal ore block face."
                ),
                verification_region="any",
            )
        ],
        auto_eval_kind="mine_coal",
    ),
    "interact_left_chest": InteractionBenchmarkTaskSpec(
        benchmark_name="interact with the left chest",
        task_config_name="interact_left_chest",
        official_task_config_name="interact_chest",
        category="Interact",
        subtasks=[
            ScriptedSubtask(
                "left chest",
                "Interact",
                90,
                prompt_text=(
                    "Point to the leftmost chest. "
                    "Do not point to the center chest or the right chest. "
                    "Place the point on the front face of the left chest."
                ),
                verification_region="left",
            )
        ],
        auto_eval_kind="unsupported",
    ),
    "open_door_then_open_chest_in_house": InteractionBenchmarkTaskSpec(
        benchmark_name="open the door then open the chest in the house",
        task_config_name="open_door_then_open_chest_in_house",
        official_task_config_name="interact_house",
        category="Interact",
        subtasks=[
            ScriptedSubtask(
                "door of the house",
                "Interact",
                45,
                prompt_text=(
                    "Point to the closed front door of the house. "
                    "Place the point on the middle of the door."
                ),
                verification_region="right",
            ),
            ScriptedSubtask(
                "chest in the house",
                "Interact",
                75,
                prompt_text=(
                    "Point to the chest inside the house. "
                    "Place the point on the front face of the chest."
                ),
                verification_region="right",
            ),
        ],
        auto_eval_kind="unsupported",
    ),
    "approach_nearest_village": InteractionBenchmarkTaskSpec(
        benchmark_name="approach the nearest village",
        task_config_name="approach_nearest_village",
        official_task_config_name="navigate_village",
        category="Navigate",
        subtasks=[
            ScriptedSubtask(
                "nearest village",
                "Approach",
                140,
                prompt_text=(
                    "Point to a visible building or doorway in the nearest village. "
                    "Choose a stable landmark in the village, not grass or sky."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="approach_target",
        target_center=(0.0, 0.0, 9.0),
        success_radius=5.0,
    ),
    "approach_ocean": InteractionBenchmarkTaskSpec(
        benchmark_name="approach the ocean",
        task_config_name="approach_ocean",
        official_task_config_name="navigate_ocean",
        category="Navigate",
        subtasks=[
            ScriptedSubtask(
                "ocean water",
                "Approach",
                140,
                prompt_text=(
                    "Point to the visible ocean water. "
                    "Place the point on the water surface, not on grass or sand."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="approach_target",
        target_center=(0.0, 0.0, 14.0),
        success_radius=8.0,
    ),
    "set_fire_on_tree": InteractionBenchmarkTaskSpec(
        benchmark_name="set fire on a tree",
        task_config_name="set_fire_on_tree",
        official_task_config_name="tool_fire",
        category="Tool",
        subtasks=[
            ScriptedSubtask(
                "oak tree trunk",
                "Use",
                90,
                prompt_text=(
                    "Point to the visible oak tree trunk. "
                    "Place the point on the wooden trunk, not on leaves or grass."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="unsupported",
    ),
    "use_bucket_get_lava": InteractionBenchmarkTaskSpec(
        benchmark_name="use bucket to get lava",
        task_config_name="use_bucket_get_lava",
        official_task_config_name="tool_lava",
        category="Tool",
        subtasks=[
            ScriptedSubtask(
                "lava source",
                "Use",
                90,
                prompt_text=(
                    "Point to the bright orange lava source block on the ground. "
                    "Place the point near the center of the lava source, not on grass, stone, or the HUD."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="lava_bucket",
    ),
    "place_minecart_on_rail": InteractionBenchmarkTaskSpec(
        benchmark_name="place minecart on the rail",
        task_config_name="place_minecart_on_rail",
        official_task_config_name="place_minecart",
        category="Place",
        subtasks=[
            ScriptedSubtask(
                "rail",
                "Use",
                90,
                prompt_text=(
                    "Point to the rail track on the ground. "
                    "Place the point on the rail itself."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="place_minecart",
    ),
    "place_oak_door_on_diamond_block": InteractionBenchmarkTaskSpec(
        benchmark_name="place the oak door on the diamond block",
        task_config_name="place_oak_door_on_diamond_block",
        official_task_config_name="place_door",
        category="Place",
        subtasks=[
            ScriptedSubtask(
                "diamond block",
                "Use",
                90,
                prompt_text=(
                    "Point to the cyan diamond block on the ground. "
                    "Place the point near the center of the top face of the diamond block."
                ),
                verification_region="right",
            )
        ],
        auto_eval_kind="unsupported",
    ),
}


def resolve_interaction_task_specs(
    task_names: Optional[List[str]] = None,
    env_source: str = "local",
) -> List[InteractionBenchmarkTaskSpec]:
    if not task_names:
        selected_items = list(INTERACTION_BENCHMARK_TASKS.items())
    else:
        selected_items = []
        for task_name in task_names:
            key = task_name.strip()
            if key not in INTERACTION_BENCHMARK_TASKS:
                raise KeyError(f"Unknown interaction benchmark task: {task_name}")
            selected_items.append((key, INTERACTION_BENCHMARK_TASKS[key]))

    specs = [replace(spec, task_key=key) for key, spec in selected_items]

    if env_source == "local":
        return specs
    if env_source == "rocket2_official":
        resolved_specs: List[InteractionBenchmarkTaskSpec] = []
        for spec in specs:
            if not spec.official_task_config_name:
                raise KeyError(f"Official ROCKET-2 task mapping not defined for: {spec.task_config_name}")
            resolved_specs.append(replace(spec, task_config_name=spec.official_task_config_name))
        return resolved_specs
    raise ValueError(f"Unknown env_source: {env_source}")
