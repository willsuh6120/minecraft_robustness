import math
from collections import Counter
from typing import Dict, List, Tuple


def _parse_buttons(summary: str) -> List[str]:
    if not summary or "buttons=" not in summary:
        return []
    try:
        button_part = summary.split("buttons=", 1)[1].split(",", 1)[0].strip()
    except Exception:
        return []
    if not button_part or button_part == "noop":
        return ["noop"] if button_part == "noop" else []
    return [token.strip() for token in button_part.split("+") if token.strip()]


def _count_nested_totals(rows: List[Dict], key: str) -> int:
    total = 0
    for row in rows:
        value = row.get(key, {}) or {}
        if isinstance(value, dict):
            for qty in value.values():
                try:
                    total += int(qty)
                except Exception:
                    continue
    return total


def _player_position(row: Dict) -> Tuple[float, float, float] | None:
    pos = row.get("player_pos") or {}
    try:
        return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    except Exception:
        return None


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def analyze_interaction_episode(result: Dict) -> Dict:
    reprompt_events = result.get("reprompt_events", []) or []
    trajectory_rows = result.get("trajectory_rows", []) or []
    category = str(result.get("category", ""))
    stop_reason = str(result.get("stop_reason", ""))
    auto_supported = bool(result.get("auto_eval_supported", False))
    auto_success = result.get("auto_success")

    no_point_events = sum(1 for event in reprompt_events if not event.get("points"))
    pointer_calls = max(1, int(result.get("pointer_calls", len(reprompt_events) or 1)))
    pointer_none_rate = no_point_events / float(pointer_calls)

    segment_areas = [int(event.get("segment_area", 0) or 0) for event in reprompt_events]
    raw_segment_areas = [int(event.get("segment_raw_area", 0) or 0) for event in reprompt_events]
    fallback_count = sum(1 for event in reprompt_events if bool(event.get("segment_used_fallback", False)))
    tiny_segment_count = sum(1 for area in raw_segment_areas if area > 0 and area < 300)
    tiny_segment_rate = tiny_segment_count / float(max(1, len(raw_segment_areas)))

    button_counts = Counter()
    for row in trajectory_rows:
        for button in _parse_buttons(str(row.get("last_action_summary", ""))):
            button_counts[button] += 1

    mined_total = _count_nested_totals(trajectory_rows, "mine_block")
    killed_total = _count_nested_totals(trajectory_rows, "kill_entity")
    used_total = _count_nested_totals(trajectory_rows, "use_item")
    crafted_total = _count_nested_totals(trajectory_rows, "craft_item")
    picked_total = _count_nested_totals(trajectory_rows, "pickup")

    first_pos = _player_position(trajectory_rows[0]) if trajectory_rows else None
    last_pos = _player_position(trajectory_rows[-1]) if trajectory_rows else None
    displacement = _distance(first_pos, last_pos) if first_pos and last_pos else None

    tags: List[str] = []
    primary_bucket = "unknown_failure"
    notes: List[str] = []

    if stop_reason == "exception":
        primary_bucket = "env_exception"
        tags.append("env_exception")
        notes.append("Episode ended with a runtime/reset exception.")
    elif auto_supported and auto_success is True:
        primary_bucket = "success"
        tags.append("success")
    elif pointer_none_rate >= 0.75:
        primary_bucket = "pointer_absent"
        tags.extend(["pointer_absent", "pointer_refresh_failed"])
        notes.append("Most reprompts returned no point.")
    elif tiny_segment_rate >= 0.5:
        primary_bucket = "segmentation_tiny"
        tags.extend(["segmentation_tiny", "mask_undersegmented"])
        notes.append("Most prompt masks were extremely small before fallback.")
    elif category in {"Tool", "Place", "Interact"} and button_counts.get("use", 0) == 0 and button_counts.get("attack", 0) > 0:
        primary_bucket = "control_mismatch"
        tags.extend(["control_mismatch", "attack_dominant_for_non_attack_task"])
        notes.append("Policy mostly attacked despite non-attack interaction type.")
    elif category in {"Mine", "Hunt"} and (button_counts.get("attack", 0) > 0) and mined_total == 0 and killed_total == 0:
        primary_bucket = "approach_or_affordance_failure"
        tags.extend(["attack_without_effect", "approach_or_affordance_failure"])
        notes.append("Attack actions occurred but no mining/kill event was recorded.")
    elif category == "Navigate" and displacement is not None and displacement < 3.0:
        primary_bucket = "navigation_stall"
        tags.extend(["navigation_stall", "low_displacement"])
        notes.append("Agent moved very little during a navigation task.")
    elif not auto_supported:
        primary_bucket = "manual_review_needed"
        tags.append("manual_review_needed")
    else:
        if auto_success is False:
            tags.append("auto_fail")
        notes.append("No dominant failure heuristic matched.")

    if pointer_none_rate > 0:
        tags.append("pointer_partial_none" if pointer_none_rate < 1.0 else "pointer_all_none")
    if fallback_count > 0:
        tags.append("sam_fallback_used")
    if button_counts.get("use", 0) > 0:
        tags.append("use_button_present")
    if button_counts.get("attack", 0) > 0:
        tags.append("attack_button_present")
    if mined_total > 0:
        tags.append("mine_event_present")
    if killed_total > 0:
        tags.append("kill_event_present")
    if used_total > 0:
        tags.append("use_item_event_present")
    if crafted_total > 0:
        tags.append("craft_event_present")
    if picked_total > 0:
        tags.append("pickup_event_present")

    return {
        "primary_bucket": primary_bucket,
        "tags": sorted(set(tags)),
        "pointer_none_rate": round(pointer_none_rate, 4),
        "reprompt_count": int(len(reprompt_events)),
        "no_point_events": int(no_point_events),
        "segment_area_min": int(min(segment_areas)) if segment_areas else 0,
        "segment_area_max": int(max(segment_areas)) if segment_areas else 0,
        "segment_raw_area_min": int(min(raw_segment_areas)) if raw_segment_areas else 0,
        "segment_raw_area_max": int(max(raw_segment_areas)) if raw_segment_areas else 0,
        "segment_fallback_count": int(fallback_count),
        "tiny_segment_rate": round(tiny_segment_rate, 4),
        "button_counts": dict(sorted(button_counts.items())),
        "mined_total": int(mined_total),
        "killed_total": int(killed_total),
        "used_total": int(used_total),
        "crafted_total": int(crafted_total),
        "picked_total": int(picked_total),
        "displacement": round(displacement, 4) if displacement is not None else None,
        "notes": " ".join(notes).strip(),
    }


def summarize_failure_buckets(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, Counter] = {}
    for row in rows:
        benchmark_name = str(row.get("benchmark_name", "unknown"))
        grouped.setdefault(benchmark_name, Counter())
        grouped[benchmark_name][str(row.get("primary_failure_bucket", "unknown"))] += 1

    summary = []
    for benchmark_name, counts in grouped.items():
        total = sum(counts.values())
        row = {
            "benchmark_name": benchmark_name,
            "episodes": int(total),
        }
        for bucket, count in sorted(counts.items()):
            row[f"bucket::{bucket}"] = int(count)
            row[f"bucket_rate::{bucket}"] = round(count / float(total), 4) if total else 0.0
        summary.append(row)
    return summary
