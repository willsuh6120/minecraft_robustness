from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


Cell = Tuple[int, int]
def _normalize_cell(value: Sequence[int | float]) -> Cell:
    if len(value) < 2:
        raise ValueError(f"Cell must contain at least two coordinates: {value}")
    return int(value[0]), int(value[1])


def _arena_params(
    *,
    target_local: Sequence[int | float],
    arena_settings: Mapping[str, object] | None,
) -> Dict[str, int]:
    arena = arena_settings if isinstance(arena_settings, Mapping) else {}
    target = _normalize_cell(target_local)
    return {
        "target_x": int(target[0]),
        "target_z": int(target[1]),
        "funnel_start_z": int(arena.get("funnel_start_z", 4) or 4),
        "half_width": int(arena.get("half_width", 4) or 4),
        "back_z": int(arena.get("back_z", -3) or -3),
    }


def _candidate_open_cells(
    *,
    obstacle_positions: Iterable[Cell],
    goal_z: int,
) -> Set[Cell]:
    blocked = {(_normalize_cell(cell)) for cell in obstacle_positions}
    cells: Set[Cell] = set()
    for z in range(0, int(goal_z) + 1):
        x_candidates = range(-1, 2) if z == int(goal_z) else range(-2, 3)
        for x in x_candidates:
            cell = (int(x), int(z))
            if cell in blocked:
                continue
            cells.add(cell)
    return cells


def _neighbors(cell: Cell, open_cells: Set[Cell], goal_z: int) -> List[Cell]:
    x, z = cell
    neighbors: List[Cell] = []
    for nxt in ((x - 1, z), (x + 1, z)):
        if nxt in open_cells:
            neighbors.append(nxt)
    if z < int(goal_z):
        forward = (x, z + 1)
        if forward in open_cells:
            neighbors.append(forward)
    return neighbors


def _collect_simple_path_cells(open_cells: Set[Cell], goal_z: int) -> Set[Cell]:
    start = (0, 0)
    if start not in open_cells:
        return set()

    sink = ("sink",)
    valid: Set[Cell] = set()
    seen_states: Set[Tuple[Cell | Tuple[str], frozenset[Cell]]] = set()

    def dfs(node: Cell | Tuple[str], visited: Set[Cell]) -> None:
        state = (node, frozenset(visited))
        if state in seen_states:
            return
        seen_states.add(state)

        if node == sink:
            valid.update(visited)
            return

        cell = node
        if int(cell[1]) == int(goal_z):
            dfs(sink, visited)

        for nxt in _neighbors(cell, open_cells, goal_z):
            if nxt in visited:
                continue
            visited.add(nxt)
            dfs(nxt, visited)
            visited.remove(nxt)

    dfs(start, {start})
    return valid


def compute_path_progress_zone_map(
    *,
    target_local: Sequence[int | float],
    obstacle_positions: Sequence[Sequence[int | float]],
    arena_settings: Mapping[str, object] | None,
) -> Dict[int, List[Cell]]:
    params = _arena_params(target_local=target_local, arena_settings=arena_settings)
    goal_z = int(params["funnel_start_z"])
    open_cells = _candidate_open_cells(
        obstacle_positions=[_normalize_cell(cell) for cell in obstacle_positions],
        goal_z=goal_z,
    )
    if (0, 0) not in open_cells:
        return {}
    valid_cells = _collect_simple_path_cells(open_cells, goal_z)

    zone_map: Dict[int, List[Cell]] = {}
    for z in range(1, goal_z + 1):
        row_cells = sorted(cell for cell in valid_cells if int(cell[1]) == z)
        if row_cells:
            zone_map[int(z)] = row_cells

    # Row z=2 is the main place where a simple forward path can still keep one
    # extra outer pre-turn cell in chicane-like layouts. Keep the row-2 reward
    # band within one lateral move of the cells that can advance directly into
    # the row-3 reward set.
    row2 = list(zone_map.get(2) or [])
    row3 = {cell for cell in zone_map.get(3) or []}
    if row2 and row3:
        row2_set = set(row2)
        anchors = {cell for cell in row2 if (int(cell[0]), 3) in row3}
        if anchors:
            queue = deque((cell, 0) for cell in anchors)
            distance: Dict[Cell, int] = {cell: 0 for cell in anchors}
            while queue:
                cell, dist = queue.popleft()
                for nxt in ((cell[0] - 1, 2), (cell[0] + 1, 2)):
                    if nxt not in row2_set or nxt in distance:
                        continue
                    distance[nxt] = int(dist) + 1
                    queue.append((nxt, int(dist) + 1))
            zone_map[2] = sorted(cell for cell in row2 if int(distance.get(cell, 99)) <= 1)
    return zone_map


def build_path_progress_reward_config(
    *,
    target_local: Sequence[int | float],
    obstacle_positions: Sequence[Sequence[int | float]],
    arena_settings: Mapping[str, object] | None,
    reward_per_zone: float = 0.25,
) -> Optional[Dict[str, object]]:
    params = _arena_params(target_local=target_local, arena_settings=arena_settings)
    goal_z = int(params["funnel_start_z"])
    if goal_z < 1:
        return None

    zone_map = compute_path_progress_zone_map(
        target_local=target_local,
        obstacle_positions=obstacle_positions,
        arena_settings=arena_settings,
    )
    if not zone_map:
        return None

    zones: List[Dict[str, object]] = []
    for row_z in range(1, goal_z + 1):
        cells = zone_map.get(row_z) or []
        if not cells:
            continue
        is_approach = int(row_z) == goal_z
        zones.append(
            {
                "id": "approach" if is_approach else f"z{int(row_z)}",
                "label": "A" if is_approach else str(int(row_z)),
                "row_z": int(row_z),
                "reward": float(reward_per_zone),
                "cells_local": [[int(x), int(z)] for x, z in cells],
            }
        )

    if not zones:
        return None

    return {
        "version": "path_progress_reward_v1",
        "origin_mode": "initial_player_pos",
        "reward_per_zone": float(reward_per_zone),
        "goal_row_z": int(goal_z),
        "zones": zones,
    }


def build_zone_label_lookup(reward_config: Mapping[str, object] | None) -> Dict[Cell, str]:
    lookup: Dict[Cell, str] = {}
    if not isinstance(reward_config, Mapping):
        return lookup
    for zone in reward_config.get("zones") or []:
        if not isinstance(zone, Mapping):
            continue
        label = str(zone.get("label") or "")
        for cell in zone.get("cells_local") or []:
            try:
                lookup[_normalize_cell(cell)] = label
            except Exception:
                continue
    return lookup
