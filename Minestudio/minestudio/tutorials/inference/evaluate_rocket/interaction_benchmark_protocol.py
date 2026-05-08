from dataclasses import dataclass, replace
from typing import Dict, List

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import (
    InteractionBenchmarkTaskSpec,
    ScriptedSubtask,
)


@dataclass(frozen=True)
class ScriptedSubtaskProtocol:
    step_budget: int
    reprompt_interval: int = 30


@dataclass(frozen=True)
class InteractionBenchmarkProtocol:
    name: str
    description: str
    warmup_noop_steps: int
    task_subtasks: Dict[str, List[ScriptedSubtaskProtocol]]
    default_eval_episodes_per_task: int = 32


INTERACTION_BENCHMARK_PROTOCOLS: Dict[str, InteractionBenchmarkProtocol] = {
    "ours_v1": InteractionBenchmarkProtocol(
        name="ours_v1",
        description="Current local evaluation protocol with fixed task-specific step budgets and 30-step reprompt cadence.",
        warmup_noop_steps=30,
        default_eval_episodes_per_task=32,
        task_subtasks={
            "hunt_sheep_right_fence": [ScriptedSubtaskProtocol(120, 30)],
            "hunt_cow_do_not_touch_sheep": [ScriptedSubtaskProtocol(120, 30)],
            "mine_emerald": [ScriptedSubtaskProtocol(90, 30)],
            "mine_coal": [ScriptedSubtaskProtocol(90, 30)],
            "interact_left_chest": [ScriptedSubtaskProtocol(90, 30)],
            "open_door_then_open_chest_in_house": [
                ScriptedSubtaskProtocol(45, 30),
                ScriptedSubtaskProtocol(75, 30),
            ],
            "approach_nearest_village": [ScriptedSubtaskProtocol(140, 30)],
            "approach_ocean": [ScriptedSubtaskProtocol(140, 30)],
            "set_fire_on_tree": [ScriptedSubtaskProtocol(90, 30)],
            "use_bucket_get_lava": [ScriptedSubtaskProtocol(90, 30)],
            "place_minecart_on_rail": [ScriptedSubtaskProtocol(90, 30)],
            "place_oak_door_on_diamond_block": [ScriptedSubtaskProtocol(90, 30)],
        },
    ),
}


def resolve_interaction_protocol(name: str) -> InteractionBenchmarkProtocol:
    if name not in INTERACTION_BENCHMARK_PROTOCOLS:
        available = ", ".join(sorted(INTERACTION_BENCHMARK_PROTOCOLS))
        raise KeyError(f"Unknown interaction benchmark protocol: {name}. Available: {available}")
    return INTERACTION_BENCHMARK_PROTOCOLS[name]


def apply_protocol_to_task_specs(
    specs: List[InteractionBenchmarkTaskSpec],
    protocol_name: str,
) -> List[InteractionBenchmarkTaskSpec]:
    protocol = resolve_interaction_protocol(protocol_name)
    resolved_specs: List[InteractionBenchmarkTaskSpec] = []
    for spec in specs:
        task_key = spec.task_key or spec.task_config_name
        if task_key not in protocol.task_subtasks:
            raise KeyError(f"Protocol '{protocol_name}' does not define task budgets for: {spec.task_config_name}")

        protocol_subtasks = protocol.task_subtasks[task_key]
        if len(protocol_subtasks) != len(spec.subtasks):
            raise ValueError(
                f"Protocol '{protocol_name}' defines {len(protocol_subtasks)} subtasks for {task_key}, "
                f"but task spec has {len(spec.subtasks)}."
            )

        updated_subtasks = [
            replace(
                subtask,
                step_budget=protocol_subtask.step_budget,
                reprompt_interval=protocol_subtask.reprompt_interval,
            )
            for subtask, protocol_subtask in zip(spec.subtasks, protocol_subtasks)
        ]
        resolved_specs.append(replace(spec, subtasks=updated_subtasks))
    return resolved_specs
