from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DifficultyFactors:
    visibility: str
    distractors: str
    path_difficulty: str
    view_difficulty: str


@dataclass(frozen=True)
class TaskEvalSpec:
    task_name: str
    planner_task: str
    default_planned_steps: int = 120
    default_replan_interval: int = 30
    notes: str = ""


@dataclass(frozen=True)
class EpisodeSpec:
    task_name: str
    planner_task: str
    split: str
    seed: int
    factors: DifficultyFactors
    planned_steps: int
    replan_interval: int


PHASE1_CORE_TASKS: Dict[str, TaskEvalSpec] = {
    "collect_wood": TaskEvalSpec(
        task_name="collect_wood",
        planner_task="collect wood from trees",
        default_planned_steps=120,
        default_replan_interval=30,
        notes="Event/inventory-based success check is stable.",
    ),
    "hunt_a_sheep": TaskEvalSpec(
        task_name="hunt_a_sheep",
        planner_task="kill the sheep and collect wool or mutton",
        default_planned_steps=120,
        default_replan_interval=30,
        notes="Short-horizon combat + loot pickup.",
    ),
    "hunt_animals": TaskEvalSpec(
        task_name="hunt_animals",
        planner_task="hunt an animal for food or drops",
        default_planned_steps=120,
        default_replan_interval=30,
        notes="More distractor-heavy than hunt_a_sheep.",
    ),
    "collect_wool": TaskEvalSpec(
        task_name="collect_wool",
        planner_task="collect wool from a sheep",
        default_planned_steps=150,
        default_replan_interval=30,
        notes="Useful for sheep localization and interaction.",
    ),
    "mine_obsidian": TaskEvalSpec(
        task_name="mine_obsidian",
        planner_task="mine obsidian blocks",
        default_planned_steps=150,
        default_replan_interval=30,
        notes="Tool-conditional mining.",
    ),
    "mine_diamond_ore": TaskEvalSpec(
        task_name="mine_diamond_ore",
        planner_task="mine diamond ore",
        default_planned_steps=150,
        default_replan_interval=30,
        notes="Mining benchmark with precise pointing.",
    ),
    "craft_table": TaskEvalSpec(
        task_name="craft_table",
        planner_task="craft a crafting table",
        default_planned_steps=180,
        default_replan_interval=30,
        notes="Inventory/crafting-oriented, included for future expansion.",
    ),
}


SPLIT_FACTORS: Dict[str, DifficultyFactors] = {
    "id": DifficultyFactors(
        visibility="easy",
        distractors="easy",
        path_difficulty="easy",
        view_difficulty="easy",
    ),
    "ood": DifficultyFactors(
        visibility="medium",
        distractors="medium",
        path_difficulty="medium",
        view_difficulty="medium",
    ),
    "stress": DifficultyFactors(
        visibility="hard",
        distractors="hard",
        path_difficulty="hard",
        view_difficulty="hard",
    ),
}


SPLIT_SEED_OFFSETS: Dict[str, int] = {
    "id": 0,
    "ood": 10_000,
    "stress": 20_000,
}


def resolve_task_specs(task_names: Optional[Iterable[str]] = None) -> List[TaskEvalSpec]:
    if task_names is None:
        return list(PHASE1_CORE_TASKS.values())
    resolved = []
    for task_name in task_names:
        key = task_name.strip()
        if key not in PHASE1_CORE_TASKS:
            raise KeyError(f"Unknown task spec: {task_name}")
        resolved.append(PHASE1_CORE_TASKS[key])
    return resolved


def build_split_seeds(split: str, count: int, base_seed: int = 0) -> List[int]:
    if split not in SPLIT_SEED_OFFSETS:
        raise KeyError(f"Unknown split: {split}")
    offset = SPLIT_SEED_OFFSETS[split]
    return [base_seed + offset + idx for idx in range(count)]


def build_episode_specs(
    task_specs: List[TaskEvalSpec],
    split: str,
    count: int,
    base_seed: int = 0,
) -> List[EpisodeSpec]:
    if split not in SPLIT_FACTORS:
        raise KeyError(f"Unknown split: {split}")
    factors = SPLIT_FACTORS[split]
    seeds = build_split_seeds(split, count, base_seed=base_seed)
    episodes: List[EpisodeSpec] = []
    for task_spec in task_specs:
        for seed in seeds:
            episodes.append(
                EpisodeSpec(
                    task_name=task_spec.task_name,
                    planner_task=task_spec.planner_task,
                    split=split,
                    seed=seed,
                    factors=factors,
                    planned_steps=task_spec.default_planned_steps,
                    replan_interval=task_spec.default_replan_interval,
                )
            )
    return episodes


def episode_spec_to_dict(spec: EpisodeSpec) -> Dict[str, object]:
    data = asdict(spec)
    data["factors"] = asdict(spec.factors)
    return data
