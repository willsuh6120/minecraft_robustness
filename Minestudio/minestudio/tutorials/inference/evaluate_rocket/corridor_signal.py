"""
Corridor-commit phase analysis for P-maze episodes.

Reads trajectory.jsonl + goal_spec.json (and optionally worldgen_plan.json) co-located
with an episode_fragment.pt and computes geometry-aware per-step phase labels:

  lateral_offset    — signed perpendicular distance from start→target line (XZ plane)
  local_x / local_z — episode position in local arena coordinates
  lane_id           — "left" / "center" / "right" based on obstacle-derived thresholds
  dist_to_target    — XZ Euclidean distance to target
  in_approach       — agent past funnel_start_z (or within fallback radius)
  in_causal_segment — corridor-commit → approach_start window (loss-weighted region)

Geometry source priority:
  1. worldgen_plan.json found by walking up from episode_dir (training runs)
  2. task yaml via baked_goal_spec_path in goal_spec.json (calibration/probe runs)
  3. Heuristic fallback: lane_threshold=1.5 blocks, approach when dist < 2.5

Episode-specific winning corridor: we do NOT assume a world-level "correct" lane.
Some worlds have multiple successful routes.  We infer which corridor the agent
stably occupied just before reaching the approach zone in THIS episode, then
backtrack to find the last entry into that corridor that remained stable until approach.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── Tuneable constants ────────────────────────────────────────────────────────
_DEFAULT_LANE_THRESHOLD: float = 1.5   # fallback only (blocks)
_DEFAULT_APPROACH_RADIUS: float = 2.5  # fallback distance-based approach (blocks)
PRE_APPROACH_WINDOW: int = 12          # steps back from approach_start for winning lane
COMMIT_LOOK_AHEAD: int = 8             # stability look-ahead window
COMMIT_STABILITY_FRAC: float = 0.65   # fraction of window that must be winning lane
MIN_CAUSAL_SEGMENT: int = 4            # minimum accepted causal segment length


# ── Arena geometry ────────────────────────────────────────────────────────────

@dataclass
class ArenaGeometry:
    """Obstacle/funnel geometry derived from worldgen_plan.json or task yaml."""
    # Obstacle positions in LOCAL arena coords (integer x, z pairs)
    obstacle_positions: List[Tuple[int, int]] = field(default_factory=list)

    # Arena layout
    target_local_x: float = 0.0        # local x of target (usually 0)
    target_local_z: float = 6.0        # local z of target
    funnel_start_z: float = 4.0        # local z where the path narrows into approach
    half_width: float = 4.0            # arena half-width (blocks each side)

    # World-space origin (local 0,0 → these world coords)
    origin_x: float = 0.0
    origin_z: float = 0.0

    # Derived lane thresholds in LOCAL x-coords
    left_threshold: float = -0.5       # local_x < this → "left" lane
    right_threshold: float = 0.5       # local_x > this → "right" lane

    loaded: bool = False               # True = loaded from real geometry, False = fallback


def _derive_thresholds(obstacle_positions: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Derive left/right lane thresholds from obstacle x-coords.
    Left corridor: local_x < min(obs_x) - 0.5
    Right corridor: local_x > max(obs_x) + 0.5
    """
    if not obstacle_positions:
        return -0.5, 0.5
    xs = [p[0] for p in obstacle_positions]
    return float(min(xs)) - 0.5, float(max(xs)) + 0.5


def _load_geometry_from_worldgen_plan(plan_path: Path, target_world: Tuple[float, float]) -> Optional[ArenaGeometry]:
    """Load ArenaGeometry from a worldgen_plan.json file."""
    try:
        plans = json.loads(plan_path.read_text(encoding="utf-8"))
        if not plans:
            return None
        plan = plans[0]  # single block always has one plan entry
        wgs = plan.get("world_generation_suggestions") or {}

        obs_raw = wgs.get("mine_path_obstacle_positions") or []
        obs_positions = [(int(p[0]), int(p[1])) for p in obs_raw if len(p) >= 2]

        tl = wgs.get("mine_target_local") or [0, 6]
        target_local_x, target_local_z = float(tl[0]), float(tl[1])
        funnel_start_z = float(wgs.get("mine_arena_funnel_start_z") or 4.0)
        half_width = float(wgs.get("mine_arena_half_width") or 4.0)

        origin_x = target_world[0] - target_local_x
        origin_z = target_world[1] - target_local_z

        left_thr, right_thr = _derive_thresholds(obs_positions)

        return ArenaGeometry(
            obstacle_positions=obs_positions,
            target_local_x=target_local_x,
            target_local_z=target_local_z,
            funnel_start_z=funnel_start_z,
            half_width=half_width,
            origin_x=origin_x,
            origin_z=origin_z,
            left_threshold=left_thr,
            right_threshold=right_thr,
            loaded=True,
        )
    except Exception:
        return None


def _load_geometry_from_task_yaml(yaml_path: Path, target_world: Tuple[float, float]) -> Optional[ArenaGeometry]:
    """Load ArenaGeometry from a mine_coal.yaml task file (calibration/probe episodes)."""
    try:
        import yaml  # optional dependency
    except ImportError:
        return None
    try:
        with yaml_path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        layout = doc.get("procedural_layout") or {}
        arena = layout.get("arena_settings") or {}

        tl = layout.get("target_local") or [0, 6]
        target_local_x, target_local_z = float(tl[0]), float(tl[1])
        funnel_start_z = float(arena.get("funnel_start_z") or 4.0)
        half_width = float(arena.get("half_width") or 4.0)

        origin_x = target_world[0] - target_local_x
        origin_z = target_world[1] - target_local_z

        # obstacle positions aren't directly in the yaml; use empty for threshold derivation
        # → thresholds fall back to ±0.5 which won't be useful; prefer worldgen_plan path
        return ArenaGeometry(
            obstacle_positions=[],
            target_local_x=target_local_x,
            target_local_z=target_local_z,
            funnel_start_z=funnel_start_z,
            half_width=half_width,
            origin_x=origin_x,
            origin_z=origin_z,
            left_threshold=-0.5,
            right_threshold=0.5,
            loaded=True,
        )
    except Exception:
        return None


def load_arena_geometry(episode_dir: Path, target_world: Tuple[float, float]) -> ArenaGeometry:
    """
    Attempt to load ArenaGeometry for episode_dir using multiple strategies.

    Strategy 1: Walk up from episode_dir to find a parent with a worldgen/ subdir,
                 then glob for worldgen_plan.json (training runs).
    Strategy 2: Follow baked_goal_spec_path in goal_spec.json up to task yaml
                 (calibration/probe runs).
    Strategy 3: Heuristic fallback using start position to derive origin,
                 with hardcoded default funnel_start_z=4.
    """
    # ── Strategy 1: worldgen_plan.json in parent tree ─────────────────────────
    for i in range(12):
        parent = episode_dir.parents[i] if i < len(episode_dir.parents) else None
        if parent is None:
            break
        worldgen_dir = parent / "worldgen"
        if worldgen_dir.is_dir():
            plans = sorted(worldgen_dir.rglob("worldgen_plan.json"))
            if plans:
                geom = _load_geometry_from_worldgen_plan(plans[0], target_world)
                if geom is not None:
                    return geom
            break  # found worldgen/ but no useful plan; stop searching

    # ── Strategy 2: baked_goal_spec_path → worldgen_plan or task yaml ────────
    goal_spec_path = episode_dir / "goal_spec.json"
    if goal_spec_path.exists():
        try:
            goal = json.loads(goal_spec_path.read_text(encoding="utf-8"))
            baked = goal.get("baked_goal_spec_path")
            if baked:
                baked_p = Path(baked)
                for i in range(10):
                    parent = baked_p.parents[i] if i < len(baked_p.parents) else None
                    if parent is None:
                        break
                    # Prefer worldgen_plan.json (has obstacle positions)
                    plan = parent / "worldgen_plan.json"
                    if plan.exists():
                        geom = _load_geometry_from_worldgen_plan(plan, target_world)
                        if geom is not None and geom.obstacle_positions:
                            return geom
                    # Fall back to task yaml (no obstacle positions, but has funnel_start_z)
                    yaml_path = parent / "mine_coal.yaml"
                    if yaml_path.exists():
                        geom = _load_geometry_from_task_yaml(yaml_path, target_world)
                        if geom is not None and geom.obstacle_positions:
                            return geom
                        # yaml found but no obstacles — continue to strategy 2b
                        break
        except Exception:
            pass

    # ── Strategy 2b: eval_bank worldgen_plan via instance folder name ────────
    # Calibration baseline episodes: instance_NNN_<variant>/ → eval_bank/instance_NNN/worldgen_plan.json
    import re as _re
    for i in range(8):
        parent = episode_dir.parents[i] if i < len(episode_dir.parents) else None
        if parent is None:
            break
        m = _re.match(r"instance_(\d+)", parent.name)
        if m:
            inst_num = m.group(1)
            # Search upward for a worldgen_plan.json whose path contains /instance_{inst_num}/
            for j in range(i + 1, i + 7):
                ancestor = episode_dir.parents[j] if j < len(episode_dir.parents) else None
                if ancestor is None:
                    break
                # Find any worldgen_plan.json under this ancestor, filtered by instance number
                plans = [
                    p for p in ancestor.rglob("worldgen_plan.json")
                    if f"/instance_{inst_num}/" in str(p) or f"/instance_{inst_num.lstrip('0') or '0'}/" in str(p)
                ]
                if plans:
                    geom = _load_geometry_from_worldgen_plan(sorted(plans)[0], target_world)
                    if geom is not None:
                        return geom
                    break  # found plans but geometry failed; stop

    # ── Strategy 3: heuristic fallback ───────────────────────────────────────
    return ArenaGeometry(loaded=False)


# ── Phase dataclass ──────────────────────────────────────────────────────────

@dataclass
class EpisodePhases:
    # ── per-step arrays (length == sequence_length) ──────────────────────────
    lateral_offset: List[float]     # signed lateral from start→target line
    local_x: List[float]            # local arena x (or NaN if no geometry)
    local_z: List[float]            # local arena z (or NaN if no geometry)
    dist_to_target: List[float]
    lane_id: List[str]              # "left" / "center" / "right"
    in_approach: List[bool]
    in_causal_segment: List[bool]

    # ── episode-level metadata ───────────────────────────────────────────────
    approach_start: int             # index of first approach step; -1 = never reached
    winning_corridor: str           # "left" / "center" / "right" / "unknown"
    commit_step: int                # first step of causal segment (≥0)
    fallback_used: bool             # True → causal segment is all-False
    sequence_length: int
    geometry_loaded: bool           # True → real geometry was used

    def causal_mask(self) -> torch.Tensor:
        return torch.tensor(self.in_causal_segment, dtype=torch.bool)

    def to_per_step_records(self) -> List[Dict]:
        return [
            {
                "step": t,
                "lateral_offset": round(self.lateral_offset[t], 4),
                "local_x": round(self.local_x[t], 4),
                "local_z": round(self.local_z[t], 4),
                "dist_to_target": round(self.dist_to_target[t], 4),
                "lane_id": self.lane_id[t],
                "in_approach": self.in_approach[t],
                "in_causal_segment": self.in_causal_segment[t],
            }
            for t in range(self.sequence_length)
        ]

    def summary(self) -> Dict:
        return {
            "approach_start": self.approach_start,
            "winning_corridor": self.winning_corridor,
            "commit_step": self.commit_step,
            "causal_length": (
                max(0, (self.approach_start if self.approach_start >= 0 else self.sequence_length) - self.commit_step)
                if not self.fallback_used else 0
            ),
            "fallback_used": self.fallback_used,
            "sequence_length": self.sequence_length,
            "geometry_loaded": self.geometry_loaded,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_lane_geometry(local_x: float, geom: ArenaGeometry) -> str:
    if local_x < geom.left_threshold:
        return "left"
    if local_x > geom.right_threshold:
        return "right"
    return "center"


def _classify_lane_lateral(lat: float, threshold: float) -> str:
    if lat < -threshold:
        return "left"
    if lat > threshold:
        return "right"
    return "center"


def _infer_winning_corridor(lanes: List[str], approach_start: int, window: int) -> str:
    """Most frequent non-center lane in the pre-approach window."""
    if approach_start <= 0:
        return "unknown"
    seg = lanes[max(0, approach_start - window): approach_start]
    counts: Dict[str, int] = {}
    for lane in seg:
        counts[lane] = counts.get(lane, 0) + 1
    side = {k: v for k, v in counts.items() if k != "center" and v > 0}
    if side:
        return max(side, key=side.__getitem__)
    center_count = counts.get("center", 0)
    return "center" if center_count > 0 else "unknown"


def _find_commit_step(
    lanes: List[str],
    approach_start: int,
    winning_corridor: str,
    min_run: int,
) -> int:
    """
    Find the start of the last sustained run of winning_corridor in lanes[0:approach_start]
    that has at least min_run steps.  Returns 0 if no qualifying run exists.

    "Last sustained run" = the LATEST run of winning_corridor that has >=min_run consecutive
    steps before approach_start.  Handles the common case where the agent briefly drifts to
    center just before the funnel.
    """
    if winning_corridor in ("unknown", "center") or approach_start <= 0:
        return 0

    last_run_start: Optional[int] = None
    run_start: Optional[int] = None
    run_len: int = 0

    for t in range(approach_start):
        if lanes[t] == winning_corridor:
            if run_start is None:
                run_start = t
                run_len = 1
            else:
                run_len += 1
        else:
            if run_start is not None and run_len >= min_run:
                last_run_start = run_start
            run_start = None
            run_len = 0

    # Episode may end inside a winning run
    if run_start is not None and run_len >= min_run:
        last_run_start = run_start

    return last_run_start if last_run_start is not None else 0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_corridor_phases(
    positions: List[Tuple[float, float]],
    target: Tuple[float, float],
    geom: Optional[ArenaGeometry] = None,
    pre_approach_window: int = PRE_APPROACH_WINDOW,
    min_causal_segment: int = MIN_CAUSAL_SEGMENT,
) -> EpisodePhases:
    """
    Compute corridor phase labels for an episode.

    positions : list of (world_x, world_z) tuples, one per env step
    target    : (world_tx, world_tz) centre of the goal voxel
    geom      : ArenaGeometry (loaded from worldgen_plan or task yaml); if None or
                not loaded, falls back to start→target lateral heuristic
    """
    T = len(positions)
    tx, tz = target
    start_px, start_pz = positions[0]

    use_geometry = bool(geom is not None and geom.loaded and geom.obstacle_positions)

    # ── Direct-path lateral reference (always computed for annotation) ────────
    d_x, d_z = tx - start_px, tz - start_pz
    d_len = math.sqrt(d_x ** 2 + d_z ** 2)
    has_dir = d_len >= 1e-6
    if has_dir:
        td_x, td_z = d_x / d_len, d_z / d_len

    lateral_offsets: List[float] = []
    local_xs: List[float] = []
    local_zs: List[float] = []
    dists: List[float] = []
    lanes: List[str] = []
    in_approach_list: List[bool] = []

    for px, pz in positions:
        dist = math.sqrt((px - tx) ** 2 + (pz - tz) ** 2)
        dists.append(dist)

        # Lateral offset (start→target perpendicular)
        if has_dir:
            rel_x, rel_z = px - start_px, pz - start_pz
            lat = rel_x * td_z - rel_z * td_x
        else:
            lat = 0.0
        lateral_offsets.append(lat)

        if use_geometry:
            lx = px - geom.origin_x
            lz = pz - geom.origin_z
            local_xs.append(lx)
            local_zs.append(lz)
            lane = _classify_lane_geometry(lx, geom)
            approach = lz >= geom.funnel_start_z
        else:
            lx = lat  # reuse lateral for display when no geometry
            lz = float("nan")
            local_xs.append(lx)
            local_zs.append(lz)
            lane = _classify_lane_lateral(lat, _DEFAULT_LANE_THRESHOLD)
            approach = dist <= _DEFAULT_APPROACH_RADIUS

        lanes.append(lane)
        in_approach_list.append(approach)

    # approach_start: first step in approach zone
    approach_start = next((t for t in range(T) if in_approach_list[t]), -1)

    effective_approach = approach_start if approach_start >= 0 else T
    winning_corridor = _infer_winning_corridor(lanes, effective_approach, pre_approach_window)
    commit_step = _find_commit_step(
        lanes, effective_approach, winning_corridor, min_causal_segment
    )

    causal_len = effective_approach - commit_step
    fallback_used = (
        approach_start < 0
        or winning_corridor in ("unknown", "center")
        or causal_len < min_causal_segment
    )

    in_causal = (
        [commit_step <= t < effective_approach for t in range(T)]
        if not fallback_used
        else [False] * T
    )

    return EpisodePhases(
        lateral_offset=lateral_offsets,
        local_x=local_xs,
        local_z=local_zs,
        dist_to_target=dists,
        lane_id=lanes,
        in_approach=in_approach_list,
        in_causal_segment=in_causal,
        approach_start=approach_start,
        winning_corridor=winning_corridor,
        commit_step=commit_step,
        fallback_used=fallback_used,
        sequence_length=T,
        geometry_loaded=use_geometry,
    )


def load_episode_phases(
    fragment_path: str,
    sequence_length: int,
    pre_approach_window: int = PRE_APPROACH_WINDOW,
    min_causal_segment: int = MIN_CAUSAL_SEGMENT,
) -> Optional[EpisodePhases]:
    """
    Load trajectory.jsonl + goal_spec.json co-located with fragment_path and
    return EpisodePhases.  Returns None if required files are absent or malformed.
    """
    fpath = Path(str(fragment_path))
    ep_dir = fpath.parent
    traj_path = ep_dir / "trajectory.jsonl"
    goal_path = ep_dir / "goal_spec.json"
    if not traj_path.exists() or not goal_path.exists():
        return None
    try:
        goal = json.loads(goal_path.read_text(encoding="utf-8"))
        target_raw = goal.get("target_world_center")
        if not (isinstance(target_raw, (list, tuple)) and len(target_raw) >= 3):
            return None
        target = (float(target_raw[0]), float(target_raw[2]))

        positions: List[Tuple[float, float]] = []
        with traj_path.open(encoding="utf-8") as fh:
            for line in fh:
                pos = json.loads(line).get("player_pos") or {}
                positions.append((float(pos.get("x", target[0])), float(pos.get("z", target[1]))))

        if len(positions) < 2:
            return None

        positions = positions[:sequence_length]

        geom = load_arena_geometry(ep_dir, target)

        return compute_corridor_phases(
            positions, target, geom=geom,
            pre_approach_window=pre_approach_window,
            min_causal_segment=min_causal_segment,
        )
    except Exception:
        return None


def load_corridor_causal_mask(
    fragment_path: str,
    sequence_length: int,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Convenience wrapper: returns bool tensor [T] where True = causal segment.
    Returns None if phases cannot be loaded.
    """
    phases = load_episode_phases(fragment_path, sequence_length, **kwargs)
    if phases is None:
        return None
    mask = phases.causal_mask()
    if mask.shape[0] < sequence_length:
        pad = torch.zeros(sequence_length - mask.shape[0], dtype=torch.bool)
        mask = torch.cat([mask, pad])
    return mask[:sequence_length]
