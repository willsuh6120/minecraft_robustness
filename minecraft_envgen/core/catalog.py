from __future__ import annotations

from .contracts import InterventionFamily


SKILL_TO_FAMILIES = {
    "collect_wood": {
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.SPAWN_CONTROL,
        InterventionFamily.SAFETY_SCAFFOLD,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "collect_stone": {
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.SAFETY_SCAFFOLD,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "collect_iron": {
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.TERRAIN_CONTROL,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "collect_diamond": {
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.TERRAIN_CONTROL,
        InterventionFamily.PRIVILEGED_HINT,
        InterventionFamily.SAFETY_SCAFFOLD,
    },
    "make_wood_pickaxe": {
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "make_stone_pickaxe": {
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "make_iron_pickaxe": {
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.RESOURCE_BOOST,
        InterventionFamily.PRIVILEGED_HINT,
    },
    "defeat_zombie": {
        InterventionFamily.MOB_PRESSURE,
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.TIME_CONTROL,
        InterventionFamily.SAFETY_SCAFFOLD,
    },
    "defeat_skeleton": {
        InterventionFamily.MOB_PRESSURE,
        InterventionFamily.INVENTORY_BOOTSTRAP,
        InterventionFamily.TIME_CONTROL,
        InterventionFamily.SAFETY_SCAFFOLD,
    },
}


TASK_CLASS_OVERRIDES = {
    "collect_wood": "harvest",
    "collect_stone": "harvest",
    "collect_iron": "harvest",
    "collect_diamond": "harvest",
    "defeat_zombie": "combat",
    "defeat_skeleton": "combat",
}


def infer_task_class(skill_label: str) -> str:
    if skill_label in TASK_CLASS_OVERRIDES:
        return TASK_CLASS_OVERRIDES[skill_label]
    if skill_label.startswith("collect_"):
        return "harvest"
    if skill_label.startswith("make_") or skill_label.startswith("craft_"):
        return "tech_tree"
    if skill_label.startswith("defeat_") or skill_label.startswith("kill_"):
        return "combat"
    if skill_label.startswith("survive_"):
        return "survival"
    return "creative"
