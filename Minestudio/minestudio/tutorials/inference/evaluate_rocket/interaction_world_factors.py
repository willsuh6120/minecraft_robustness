import random
from typing import Dict, List


FACTOR_VALUE_SPACE: Dict[str, List[str]] = {
    "visibility": ["keep", "increase", "decrease"],
    "distractors": ["keep", "increase", "decrease"],
    "path_difficulty": ["keep", "increase", "decrease"],
    "view_difficulty": ["keep", "increase", "decrease"],
}


FACTOR_SEMANTICS: Dict[str, str] = {
    "visibility": (
        "'increase' means make the target harder to see, for example by occlusion, clutter, "
        "or partial visibility. 'decrease' means make the target easier to see."
    ),
    "distractors": (
        "'increase' means add or strengthen similar distractors. "
        "'decrease' means reduce distractors relative to the source environment."
    ),
    "path_difficulty": (
        "'increase' means add traversal difficulty, obstacles, or tighter affordance geometry. "
        "'decrease' means simplify traversal."
    ),
    "view_difficulty": (
        "'increase' means make the target less centered or farther from the default initial view. "
        "'decrease' means make the target more centered and directly visible."
    ),
}

PRIMARY_FACTOR_CODES = ["R", "H", "O", "C", "P", "A"]
FACTOR_SPLIT_LABELS = ["clean", "train_id", "ood", "stress"]

PRIMARY_FACTOR_CODE_SEMANTICS: Dict[str, str] = {
    "R": "spawn-target distance",
    "H": "initial heading offset",
    "O": "visibility / occlusion",
    "C": "clutter / confusability",
    "P": "path complexity",
    "A": "action margin",
}


ALLOWED_EPISODE_STATUS = ["success", "failure", "ambiguous"]
ALLOWED_CONFIDENCE = ["low", "medium", "high"]
ALLOWED_PRIMARY_FAILURE_MODES = [
    "success",
    "target_acquisition_failure",
    "distractor_confusion",
    "path_failure",
    "visibility_recovery_failure",
    "final_actuation_failure",
    "timeout_unstable_recovery",
    "ambiguous",
]


GLOBAL_LAYOUT_CHANGE_TAGS = [
    "keep_layout",
    "make_target_visible_at_step0",
    "shift_target_off_center",
    "shift_target_farther",
    "partial_occluder",
    "add_similar_distractor",
    "add_low_obstacle",
    "add_medium_obstacle",
    "narrow_action_margin",
    "move_spawn_lateral",
    "move_spawn_back",
]


TASK_KEY_ALIASES: Dict[str, str] = {
    "hunt_fence": "hunt_sheep_right_fence",
    "hunt_cowonly": "hunt_cow_do_not_touch_sheep",
    "mine_emerald": "mine_emerald",
    "mine_coal": "mine_coal",
    "interact_chest": "interact_left_chest",
    "interact_house": "open_door_then_open_chest_in_house",
    "navigate_village": "approach_nearest_village",
    "navigate_ocean": "approach_ocean",
    "tool_fire": "set_fire_on_tree",
    "tool_lava": "use_bucket_get_lava",
    "place_minecart": "place_minecart_on_rail",
    "place_door": "place_oak_door_on_diamond_block",
}


TASK_ALLOWED_LAYOUT_CHANGE_TAGS: Dict[str, List[str]] = {
    "hunt_sheep_right_fence": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
        "add_medium_obstacle",
    ],
    "hunt_cow_do_not_touch_sheep": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
        "add_medium_obstacle",
    ],
    "mine_emerald": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "shift_target_farther",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
        "add_medium_obstacle",
        "narrow_action_margin",
        "move_spawn_lateral",
        "move_spawn_back",
    ],
    "mine_coal": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "shift_target_farther",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
        "add_medium_obstacle",
        "move_spawn_lateral",
        "move_spawn_back",
    ],
    "interact_left_chest": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
    ],
    "open_door_then_open_chest_in_house": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "move_spawn_lateral",
        "move_spawn_back",
        "add_low_obstacle",
        "add_medium_obstacle",
    ],
    "approach_nearest_village": [
        "keep_layout",
        "make_target_visible_at_step0",
        "move_spawn_lateral",
        "move_spawn_back",
        "partial_occluder",
        "add_low_obstacle",
        "add_medium_obstacle",
    ],
    "approach_ocean": [
        "keep_layout",
        "make_target_visible_at_step0",
        "move_spawn_lateral",
        "move_spawn_back",
        "partial_occluder",
        "add_low_obstacle",
    ],
    "set_fire_on_tree": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
    ],
    "use_bucket_get_lava": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
    ],
    "place_minecart_on_rail": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
    ],
    "place_oak_door_on_diamond_block": [
        "keep_layout",
        "make_target_visible_at_step0",
        "shift_target_off_center",
        "partial_occluder",
        "add_similar_distractor",
        "add_low_obstacle",
    ],
}


TASK_ALLOWED_FACTORS: Dict[str, List[str]] = {
    task_key: list(FACTOR_VALUE_SPACE.keys())
    for task_key in TASK_ALLOWED_LAYOUT_CHANGE_TAGS
}


TASK_PRIMARY_FACTOR_CODES: Dict[str, List[str]] = {
    "mine_coal": ["R", "H", "O", "C", "P"],
    "mine_emerald": ["R", "H", "O", "C", "P", "A"],
    "hunt_sheep_right_fence": ["R", "H", "O", "C", "P"],
    "hunt_cow_do_not_touch_sheep": ["R", "H", "O", "C", "P"],
    "interact_left_chest": ["R", "H", "O", "C", "P", "A"],
    "open_door_then_open_chest_in_house": ["R", "H", "O", "C", "P", "A"],
    "approach_nearest_village": ["R", "H", "O", "C", "P"],
    "approach_ocean": ["R", "H", "O", "C", "P"],
    "set_fire_on_tree": ["R", "H", "O", "C", "P", "A"],
    "use_bucket_get_lava": ["R", "H", "O", "C", "P", "A"],
    "place_minecart_on_rail": ["R", "H", "O", "C", "P", "A"],
    "place_oak_door_on_diamond_block": ["R", "H", "O", "C", "P", "A"],
}


def canonical_task_key(task_name: str) -> str:
    key = str(task_name).strip()
    return TASK_KEY_ALIASES.get(key, key)


def world_factor_schema_for_task(task_name: str) -> Dict:
    task_key = canonical_task_key(task_name)
    return {
        "task_key": task_key,
        "primary_factor_codes": TASK_PRIMARY_FACTOR_CODES.get(task_key, PRIMARY_FACTOR_CODES),
        "primary_factor_code_semantics": PRIMARY_FACTOR_CODE_SEMANTICS,
        "allowed_factors": {key: FACTOR_VALUE_SPACE[key] for key in TASK_ALLOWED_FACTORS.get(task_key, FACTOR_VALUE_SPACE.keys())},
        "factor_semantics": FACTOR_SEMANTICS,
        "allowed_layout_change_tags": TASK_ALLOWED_LAYOUT_CHANGE_TAGS.get(task_key, GLOBAL_LAYOUT_CHANGE_TAGS),
    }


def normalize_factor_value(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"keep", "increase", "decrease"} else "keep"


def normalize_layout_tags(task_name: str, tags) -> List[str]:
    task_key = canonical_task_key(task_name)
    allowed = set(TASK_ALLOWED_LAYOUT_CHANGE_TAGS.get(task_key, GLOBAL_LAYOUT_CHANGE_TAGS))
    normalized: List[str] = []
    for tag in tags or []:
        tag = str(tag).strip()
        if tag in allowed:
            normalized.append(tag)
    return sorted(set(normalized))


def normalize_worldgen_suggestions(task_name: str, suggestions: Dict) -> Dict:
    task_key = canonical_task_key(task_name)
    normalized = {
        "visibility": normalize_factor_value((suggestions or {}).get("visibility", "keep")),
        "distractors": normalize_factor_value((suggestions or {}).get("distractors", "keep")),
        "path_difficulty": normalize_factor_value((suggestions or {}).get("path_difficulty", "keep")),
        "view_difficulty": normalize_factor_value((suggestions or {}).get("view_difficulty", "keep")),
        "layout_changes": normalize_layout_tags(task_key, (suggestions or {}).get("layout_changes", [])),
        "notes": str((suggestions or {}).get("notes", "")).strip(),
    }
    if not normalized["layout_changes"]:
        normalized["layout_changes"] = ["keep_layout"]
    # Preserve task-specific or experimental worldgen override keys such as
    # exact mine occluder geometry controls used by fixed-bank and collect-plan
    # recipes. The normalized base keys above still control the generic factor
    # semantics; these extras are passed through for downstream worldgen code.
    for key, value in dict(suggestions or {}).items():
        if key in normalized:
            continue
        if isinstance(value, list):
            normalized[key] = list(value)
        elif isinstance(value, dict):
            normalized[key] = dict(value)
        else:
            normalized[key] = value
    return normalized


def normalize_episode_status(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in set(ALLOWED_EPISODE_STATUS) else "ambiguous"


def normalize_confidence(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in set(ALLOWED_CONFIDENCE) else "medium"


def normalize_primary_failure_mode(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in set(ALLOWED_PRIMARY_FAILURE_MODES) else "ambiguous"


def normalize_primary_factors(values) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        token = str(value or "").strip().upper()
        if token in PRIMARY_FACTOR_CODES:
            normalized.append(token)
    return sorted(set(normalized))


def normalize_severity(value) -> int:
    try:
        severity = int(value)
    except Exception:
        return 1
    return max(0, min(severity, 3))


def empty_factor_levels(task_name: str) -> Dict[str, int]:
    task_key = canonical_task_key(task_name)
    return {code: 0 for code in TASK_PRIMARY_FACTOR_CODES.get(task_key, PRIMARY_FACTOR_CODES)}


def normalize_factor_levels(task_name: str, values) -> Dict[str, int]:
    normalized = empty_factor_levels(task_name)
    if not isinstance(values, dict):
        return normalized
    for code in list(normalized.keys()):
        if code in values:
            normalized[code] = normalize_severity(values.get(code, 0))
    return normalized


def derive_factor_levels_from_review(
    task_name: str,
    primary_failure_mode: str,
    primary_factors,
    severity,
) -> Dict[str, int]:
    levels = empty_factor_levels(task_name)
    severity = 0 if str(primary_failure_mode) == "success" else max(1, normalize_severity(severity))
    for code in normalize_primary_factors(primary_factors):
        if code in levels:
            levels[code] = severity
    if str(primary_failure_mode) == "success":
        return levels
    if not any(levels.values()):
        fallback_by_bucket = {
            "target_acquisition_failure": ["R", "H"],
            "distractor_confusion": ["C"],
            "path_failure": ["P", "R"],
            "visibility_recovery_failure": ["O", "H"],
            "final_actuation_failure": ["A"],
            "timeout_unstable_recovery": ["H", "O"],
        }
        for code in fallback_by_bucket.get(str(primary_failure_mode), []):
            if code in levels:
                levels[code] = severity
    return levels


def factor_levels_to_worldgen_suggestions(task_name: str, factor_levels: Dict[str, int]) -> Dict:
    task_key = canonical_task_key(task_name)
    levels = normalize_factor_levels(task_key, factor_levels)
    suggestions = {
        "visibility": "increase" if levels.get("O", 0) > 0 else "keep",
        "distractors": "increase" if levels.get("C", 0) > 0 else "keep",
        "path_difficulty": "increase" if levels.get("P", 0) > 0 else "keep",
        "view_difficulty": "increase" if levels.get("H", 0) > 0 else "keep",
        "layout_changes": [],
        "notes": f"Derived from factor levels {levels}.",
    }
    if levels.get("R", 0) > 0:
        suggestions["layout_changes"].append("move_spawn_back")
        if levels["R"] >= 2:
            suggestions["layout_changes"].append("shift_target_farther")
    if levels.get("H", 0) > 0:
        suggestions["layout_changes"].extend(["move_spawn_lateral", "shift_target_off_center"])
    if levels.get("O", 0) > 0:
        suggestions["layout_changes"].append("partial_occluder")
    if levels.get("C", 0) > 0:
        suggestions["layout_changes"].append("add_similar_distractor")
    if levels.get("P", 0) > 0:
        suggestions["layout_changes"].append("add_medium_obstacle" if levels["P"] >= 2 else "add_low_obstacle")
    if levels.get("A", 0) > 0:
        suggestions["layout_changes"].append("narrow_action_margin")
    return normalize_worldgen_suggestions(task_key, suggestions)


def _ordered_factor_codes(task_name: str, preferred_codes=None) -> List[str]:
    task_key = canonical_task_key(task_name)
    allowed = list(TASK_PRIMARY_FACTOR_CODES.get(task_key, PRIMARY_FACTOR_CODES))
    ordered: List[str] = []
    for code in normalize_primary_factors(preferred_codes):
        if code in allowed and code not in ordered:
            ordered.append(code)
    for code in allowed:
        if code not in ordered:
            ordered.append(code)
    return ordered


def hard_factor_count(task_name: str, factor_levels: Dict[str, int]) -> int:
    levels = normalize_factor_levels(task_name, factor_levels)
    return sum(1 for value in levels.values() if int(value) == 2)


def classify_factor_split(task_name: str, factor_levels: Dict[str, int]) -> str:
    levels = normalize_factor_levels(task_name, factor_levels)
    if not any(int(value) > 0 for value in levels.values()):
        return "clean"
    if any(int(value) >= 3 for value in levels.values()):
        return "stress"
    return "train_id" if hard_factor_count(task_name, levels) <= 1 else "ood"


def project_factor_levels_to_split(
    task_name: str,
    factor_levels: Dict[str, int],
    split_label: str,
    preferred_codes=None,
) -> Dict[str, int]:
    levels = normalize_factor_levels(task_name, factor_levels)
    split = str(split_label or "").strip().lower()
    if split not in FACTOR_SPLIT_LABELS:
        return levels
    if split == "clean":
        return empty_factor_levels(task_name)

    ordered = _ordered_factor_codes(task_name, preferred_codes)
    nonzero_codes = [code for code in ordered if int(levels.get(code, 0)) > 0]

    if split == "train_id":
        projected = empty_factor_levels(task_name)
        hard_kept = False
        for code in ordered:
            value = min(2, int(levels.get(code, 0)))
            if value >= 2:
                if not hard_kept:
                    projected[code] = 2
                    hard_kept = True
                else:
                    projected[code] = 1
            else:
                projected[code] = value
        return projected

    if split == "ood":
        projected = empty_factor_levels(task_name)
        selected = list(nonzero_codes)
        if len(selected) < 2:
            selected = ordered[: max(2, len(selected) or 2)]
        hard_target = min(3, max(2, len(selected)))
        if len(selected) < hard_target:
            for code in ordered:
                if code not in selected:
                    selected.append(code)
                if len(selected) >= hard_target:
                    break
        for idx, code in enumerate(selected):
            projected[code] = 2 if idx < hard_target else 1
        return projected

    projected = empty_factor_levels(task_name)
    primary = nonzero_codes[0] if nonzero_codes else ordered[0]
    projected[primary] = 3
    if len(nonzero_codes) >= 2:
        projected[nonzero_codes[1]] = 2
    return projected


def sample_factor_levels_for_split(
    task_name: str,
    split_label: str,
    rng: random.Random,
    preferred_codes=None,
) -> Dict[str, int]:
    split = str(split_label or "").strip().lower()
    ordered = _ordered_factor_codes(task_name, preferred_codes)
    levels = empty_factor_levels(task_name)
    if split not in FACTOR_SPLIT_LABELS or split == "clean":
        return levels

    if split == "train_id":
        active_count = 1 if rng.random() < 0.6 else min(2, len(ordered))
        selected = list(ordered[:active_count])
        if active_count > 1 and rng.random() < 0.5:
            selected = rng.sample(ordered, active_count)
        hard_code = selected[0] if rng.random() < 0.5 else None
        for code in selected:
            levels[code] = 2 if code == hard_code else 1
        return normalize_factor_levels(task_name, levels)

    if split == "ood":
        hard_count = 2 if rng.random() < 0.7 else min(3, len(ordered))
        selected = list(ordered[:hard_count])
        if len(ordered) > hard_count:
            selected = rng.sample(ordered, hard_count)
        for code in selected:
            levels[code] = 2
        return normalize_factor_levels(task_name, levels)

    primary = ordered[0]
    if len(ordered) > 1 and rng.random() < 0.5:
        primary = rng.choice(ordered)
    levels[primary] = 3
    if len(ordered) > 1 and rng.random() < 0.7:
        secondary = next(code for code in ordered if code != primary)
        levels[secondary] = 2 if rng.random() < 0.7 else 1
    return normalize_factor_levels(task_name, levels)
