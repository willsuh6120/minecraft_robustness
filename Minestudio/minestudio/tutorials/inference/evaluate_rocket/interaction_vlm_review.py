import argparse
import base64
import cv2
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    ALLOWED_CONFIDENCE,
    ALLOWED_EPISODE_STATUS,
    ALLOWED_PRIMARY_FAILURE_MODES,
    PRIMARY_FACTOR_CODES,
    PRIMARY_FACTOR_CODE_SEMANTICS,
    canonical_task_key,
    derive_factor_levels_from_review,
    factor_levels_to_worldgen_suggestions,
    normalize_confidence,
    normalize_episode_status,
    normalize_primary_failure_mode,
    normalize_primary_factors,
    normalize_severity,
    world_factor_schema_for_task,
)


DEFAULT_MODEL = "gpt-4o"
DEFAULT_MID_FRAME_INTERVAL = 30


def derive_worldgen_suggestions_from_review(
    task_name: str,
    primary_failure_mode: str,
    primary_factors: List[str],
    severity: int,
) -> Dict:
    factor_levels = derive_factor_levels_from_review(
        task_name,
        primary_failure_mode=primary_failure_mode,
        primary_factors=primary_factors,
        severity=severity,
    )
    suggestions = factor_levels_to_worldgen_suggestions(task_name, factor_levels)
    suggestions["notes"] = (
        f"Derived from bucket={primary_failure_mode}, factors={sorted(primary_factors or [])}, "
        f"severity={int(severity)}, factor_levels={factor_levels}."
    )
    return suggestions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-mid-images", type=int, default=3)
    parser.add_argument("--max-reprompt-images", type=int, dest="max_mid_images", help=argparse.SUPPRESS)
    parser.add_argument("--mid-frame-interval", type=int, default=DEFAULT_MID_FRAME_INTERVAL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_image_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def find_episode_dirs(run_dir: Path, task_filter: Optional[List[str]]) -> List[Path]:
    task_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    episode_dirs: List[Path] = []
    for task_dir in task_dirs:
        if task_filter and task_dir.name not in task_filter:
            continue
        for seed_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            episode_dirs.append(seed_dir)
    return episode_dirs


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compact_trajectory_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "num_rows": 0,
            "first_action_summaries": [],
            "last_action_summaries": [],
            "use_item_totals": {},
            "mine_block_totals": {},
            "kill_entity_totals": {},
            "pickup_totals": {},
            "craft_item_totals": {},
        }

    def merge_counter(key: str) -> Dict[str, int]:
        merged = Counter()
        for row in rows:
            value = row.get(key) or {}
            if isinstance(value, dict):
                for item_key, qty in value.items():
                    try:
                        merged[str(item_key)] += int(qty)
                    except Exception:
                        continue
        return dict(sorted(merged.items()))

    return {
        "num_rows": len(rows),
        "first_action_summaries": [row.get("last_action_summary", "none") for row in rows[:5]],
        "last_action_summaries": [row.get("last_action_summary", "none") for row in rows[-5:]],
        "use_item_totals": merge_counter("use_item"),
        "mine_block_totals": merge_counter("mine_block"),
        "kill_entity_totals": merge_counter("kill_entity"),
        "pickup_totals": merge_counter("pickup"),
        "craft_item_totals": merge_counter("craft_item"),
        "initial_inventory": rows[0].get("inventory_totals", {}),
        "final_inventory": rows[-1].get("inventory_totals", {}),
    }


def horizontal_distance(player_pos: Dict, target_world_center: List[float]) -> float:
    return (
        (float(player_pos.get("x", 0.0)) - float(target_world_center[0])) ** 2
        + (float(player_pos.get("z", 0.0)) - float(target_world_center[2])) ** 2
    ) ** 0.5


def travel_span(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    start_pos = rows[0].get("player_pos") or {}
    start_x = float(start_pos.get("x", 0.0))
    start_z = float(start_pos.get("z", 0.0))
    best = 0.0
    for row in rows:
        player_pos = row.get("player_pos") or {}
        dx = float(player_pos.get("x", 0.0)) - start_x
        dz = float(player_pos.get("z", 0.0)) - start_z
        best = max(best, (dx * dx + dz * dz) ** 0.5)
    return best


def compute_world_shift_telemetry(result: Dict, rows: List[Dict]) -> Dict:
    goal_metadata = result.get("goal_metadata") or {}
    target_world_center = goal_metadata.get("target_world_center")
    auto_progress = result.get("auto_progress") or {}
    mined_count = float(auto_progress.get("mine_block", 0) or 0)
    inventory_delta = float(auto_progress.get("inventory_delta", 0) or 0)
    progress_score = mined_count + inventory_delta
    telemetry = {
        "start_distance_to_target": None,
        "min_distance_to_target": None,
        "end_distance_to_target": None,
        "distance_improvement": None,
        "travel_span": float(travel_span(rows)),
        "interaction_progress_score": float(progress_score),
        "mined_count": float(mined_count),
        "inventory_delta": float(inventory_delta),
        "reached_target_neighborhood": False,
    }
    if not (isinstance(target_world_center, list) and len(target_world_center) == 3):
        return telemetry
    distances = [
        horizontal_distance(row.get("player_pos") or {}, target_world_center)
        for row in rows
        if isinstance(row.get("player_pos"), dict)
    ]
    if not distances:
        return telemetry
    start_distance = float(distances[0])
    min_distance = float(min(distances))
    end_distance = float(distances[-1])
    telemetry.update(
        {
            "start_distance_to_target": start_distance,
            "min_distance_to_target": min_distance,
            "end_distance_to_target": end_distance,
            "distance_improvement": float(start_distance - min_distance),
            "reached_target_neighborhood": bool(min_distance <= 3.0),
        }
    )
    return telemetry


def first_existing(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def latest_matching(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def extract_video_step_frames(
    video_path: Optional[Path],
    output_dir: Path,
    steps: List[int],
) -> List[Tuple[str, Path]]:
    if video_path is None or not video_path.exists() or not steps:
        return []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        pairs: List[Tuple[str, Path]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        seen_indices = set()
        for step in steps:
            frame_index = max(0, min(int(step), frame_count - 1))
            if frame_index in seen_indices:
                continue
            seen_indices.add(frame_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            image_path = output_dir / f"mid_step_{int(step):03d}.png"
            cv2.imwrite(str(image_path), frame)
            pairs.append((f"mid_step_{int(step):03d}", image_path))
        return pairs
    finally:
        capture.release()


def select_episode_images(
    episode_dir: Path,
    result: Dict,
    max_mid_images: int,
    mid_frame_interval: int,
) -> List[Tuple[str, Path]]:
    images: List[Tuple[str, Path]] = []

    goal_overlay = first_existing(
        [
            episode_dir / "selected_pose" / "goal_mask_overlay.png",
            episode_dir / "goal_mask_overlay.png",
            episode_dir / "selected_pose" / "goal_bbox_overlay.png",
        ]
    )
    if goal_overlay is not None:
        images.append(("goal_mask_overlay", goal_overlay))

    start_image = first_existing(
        [
            episode_dir / "rollout_after_warmup_pov.png",
            episode_dir / "rollout_reset_pov.png",
        ]
    )
    if start_image is not None:
        images.append(("start_pov_after_warmup", start_image))

    num_steps = int(result.get("num_steps") or 0)
    if mid_frame_interval > 0 and num_steps > mid_frame_interval:
        requested_steps = list(range(mid_frame_interval, num_steps, mid_frame_interval))
        if max_mid_images > 0:
            requested_steps = requested_steps[:max_mid_images]
        raw_video = None
        for candidate in sorted(episode_dir.glob("*.mp4")):
            if candidate.name.endswith("_annotated.mp4"):
                continue
            raw_video = candidate
            break
        images.extend(
            extract_video_step_frames(
                raw_video,
                episode_dir / "vlm_review_assets",
                requested_steps,
            )
        )

    final_image = first_existing(
        [
            latest_matching(episode_dir, "*_final_frame_annotated.png") or Path("__missing__"),
            latest_matching(episode_dir, "*_final_frame.png") or Path("__missing__"),
        ]
    )
    if final_image is not None:
        images.append(("final_frame", final_image))

    return images


def parse_json_object(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("VLM review did not return valid JSON.")


def build_review_packet(
    result: Dict,
    trajectory_summary: Dict,
    telemetry: Dict,
    image_pairs: List[Tuple[str, Path]],
) -> Dict:
    task_name = result.get("task_config_name") or result.get("benchmark_name") or "unknown"
    factor_schema = world_factor_schema_for_task(task_name)
    review_schema = {
        "episode_status": "success | failure | ambiguous",
        "confidence": "low | medium | high",
        "bucket": (
            "success | target_acquisition_failure | distractor_confusion | "
            "path_failure | visibility_recovery_failure | final_actuation_failure | "
            "timeout_unstable_recovery | ambiguous"
        ),
        "primary_factors": PRIMARY_FACTOR_CODES,
        "severity": "integer 0..3",
        "evidence": ["short bullet evidence strings"],
        "trainable_with_rl": True,
        "exclude_from_rl_reason": "",
        "review_summary": "short paragraph",
    }

    text_payload = {
        "benchmark_name": result.get("benchmark_name"),
        "task_config_name": result.get("task_config_name"),
        "category": result.get("category"),
        "stop_reason": result.get("stop_reason"),
        "auto_eval_supported": result.get("auto_eval_supported"),
        "auto_success": result.get("auto_success"),
        "auto_metric": result.get("auto_metric"),
        "auto_reason": result.get("auto_reason"),
        "failure_analysis": result.get("failure_analysis", {}),
        "reprompt_events": result.get("reprompt_events", []),
        "trajectory_summary": trajectory_summary,
        "world_shift_telemetry": telemetry,
    }

    return {
        "task_name": task_name,
        "factor_schema": factor_schema,
        "review_schema": review_schema,
        "text_payload": text_payload,
        "images": [{"label": label, "path": str(path.resolve())} for label, path in image_pairs],
    }


def build_messages(packet: Dict) -> List[Dict]:
    factor_schema = packet["factor_schema"]
    review_schema = packet["review_schema"]
    text_payload = packet["text_payload"]
    image_pairs = [(item["label"], Path(item["path"])) for item in packet.get("images", [])]

    user_content: List[Dict] = [
        {
            "type": "text",
            "text": (
                "You are reviewing a Minecraft interaction-agent rollout for factorized world-shift training. "
                "The goal image and target segment are already fixed/generated for the episode. "
                "Do NOT diagnose errors as segmentation_tiny, tracking drift, pointer identity, or goal-mask quality. "
                "Assume the goal asset is correct unless there is explicit evidence of corruption. "
                "Your job is only to judge the rollout in terms of world-shift failure buckets and to suggest "
                "which factors are primarily responsible.\n\n"
                "Use the world_shift_telemetry values aggressively. "
                "If the start frame clearly shows the target or the telemetry shows the agent reached the target neighborhood, "
                "do not call it target_acquisition_failure. "
                "If the agent reaches the target neighborhood but still fails the task, prefer final_actuation_failure unless there is stronger evidence for path or recovery issues.\n\n"
                "Use these buckets:\n"
                "- target_acquisition_failure: agent never meaningfully acquires the target / target region from the start.\n"
                "- distractor_confusion: agent goes to a wrong but similar-looking object or landmark.\n"
                "- path_failure: agent knows roughly where to go but fails because approach path / obstacle / traversal is the problem.\n"
                "- visibility_recovery_failure: target becomes hidden or leaves view and the agent fails to recover it.\n"
                "- final_actuation_failure: agent reaches the correct target neighborhood but fails the last interaction.\n"
                "- timeout_unstable_recovery: behavior is indecisive, oscillatory, or times out without a cleaner bucket.\n\n"
                "Primary factor codes:\n"
                f"{json.dumps(PRIMARY_FACTOR_CODE_SEMANTICS, ensure_ascii=False, indent=2)}\n\n"
                "Return 1-3 primary factors only. "
                "Severity should be 0 for success, otherwise 1..3 where 3 means strongest stress signal.\n\n"
                "Prioritize world-shift diagnosis over low-level perception diagnosis. "
                "For mining tasks, if the goal target is visible enough and the agent reaches the ore neighborhood but does not mine it, "
                "prefer final_actuation_failure. "
                "Only use ambiguous if the episode evidence is genuinely insufficient.\n\n"
                f"Return strict JSON only with this schema:\n{json.dumps(review_schema, ensure_ascii=False, indent=2)}\n\n"
                f"Allowed episode_status values: {ALLOWED_EPISODE_STATUS}\n"
                f"Allowed confidence values: {ALLOWED_CONFIDENCE}\n"
                f"Allowed primary_failure_mode values: {ALLOWED_PRIMARY_FAILURE_MODES}\n"
                f"Task-specific factor schema:\n{json.dumps(factor_schema, ensure_ascii=False, indent=2)}\n\n"
                f"Episode payload:\n{json.dumps(text_payload, ensure_ascii=False, indent=2)}\n\n"
                "Image semantics:\n"
                "- goal_mask_overlay: the goal image with the target segment overlaid in red.\n"
                "- start_pov_after_warmup: the rollout start observation after warmup.\n"
                "- mid_step_XXX: raw rollout frames sampled every fixed number of steps.\n"
                "- final_frame: final rollout frame, usually annotated."
            ),
        }
    ]

    for label, image_path in image_pairs:
        user_content.append(
            {
                "type": "text",
                "text": f"Image label: {label}",
            }
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_file_base64(image_path)}"
                },
            }
        )

    return [
        {
            "role": "system",
            "content": (
                "You are a careful failure analyst for Minecraft world-shift robustness. "
                "Return strict JSON only. "
                "Do not invent factor names or layout tags outside the provided schema. "
                "Do not attribute failure to segmentation size, pointer identity, or tracking unless the inputs explicitly prove the goal asset is corrupted."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def contains_any_token(text: str, tokens: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in tokens)


def guardrail_primary_factors(
    task_name: str,
    primary_failure_mode: str,
    primary_factors: List[str],
    evidence: List[str],
    review_summary: str,
    telemetry: Dict,
) -> List[str]:
    allowed_codes = set(world_factor_schema_for_task(task_name).get("primary_factor_codes", PRIMARY_FACTOR_CODES))
    normalized = [code for code in normalize_primary_factors(primary_factors) if code in allowed_codes]
    if primary_failure_mode == "success":
        return []

    combined_text = " ".join([*(evidence or []), str(review_summary or "")]).lower()

    if primary_failure_mode == "final_actuation_failure":
        # For actuation failures, default to A only. Allow O/C only when the review
        # contains explicit evidence of occlusion or distractor confusion.
        adjusted: List[str] = []
        if "A" in allowed_codes:
            adjusted.append("A")
        occlusion_tokens = [
            "occluded",
            "occlusion",
            "obscured",
            "hidden",
            "not visible",
            "hard to see",
            "couldn't see",
            "cannot see",
            "lost sight",
            "dark",
            "dim",
            "foliage",
        ]
        confusion_tokens = [
            "distractor",
            "confus",
            "wrong target",
            "wrong object",
            "wrong ore",
            "similar-looking",
            "similar object",
            "mistook",
            "mistaken",
        ]
        reached_neighborhood = bool((telemetry or {}).get("reached_target_neighborhood"))
        if contains_any_token(combined_text, occlusion_tokens) and "O" in allowed_codes:
            adjusted.append("O")
        if contains_any_token(combined_text, confusion_tokens) and "C" in allowed_codes:
            adjusted.append("C")
        if reached_neighborhood or adjusted:
            return adjusted[:3]

    return normalized


def normalize_review_output(task_name: str, review: Dict, telemetry: Dict) -> Dict:
    normalized_evidence: List[str] = []
    for item in review.get("evidence", []) or []:
        item = str(item).strip()
        if item:
            normalized_evidence.append(item)

    primary_failure_mode = normalize_primary_failure_mode(
        review.get("bucket", review.get("primary_failure_mode"))
    )
    raw_primary_factors = normalize_primary_factors(review.get("primary_factors", []))
    review_summary = str(review.get("review_summary", "")).strip()
    primary_factors = guardrail_primary_factors(
        task_name=task_name,
        primary_failure_mode=primary_failure_mode,
        primary_factors=raw_primary_factors,
        evidence=normalized_evidence,
        review_summary=review_summary,
        telemetry=telemetry,
    )
    severity = normalize_severity(review.get("severity", 0 if primary_failure_mode == "success" else 1))
    normalized = {
        "episode_status": normalize_episode_status(review.get("episode_status")),
        "confidence": normalize_confidence(review.get("confidence")),
        "primary_failure_mode": primary_failure_mode,
        "primary_factors": primary_factors,
        "primary_factors_model_raw": raw_primary_factors,
        "severity": severity,
        "evidence": normalized_evidence,
        "world_generation_suggestions": derive_worldgen_suggestions_from_review(
            task_name,
            primary_failure_mode=primary_failure_mode,
            primary_factors=primary_factors,
            severity=severity,
        ),
        "trainable_with_rl": bool(review.get("trainable_with_rl", True)),
        "exclude_from_rl_reason": str(review.get("exclude_from_rl_reason", "")).strip(),
        "review_summary": review_summary,
    }
    return normalized


def review_episode(
    client: OpenAI,
    model_id: str,
    episode_dir: Path,
    max_mid_images: int,
    mid_frame_interval: int,
) -> Dict:
    result = load_json(episode_dir / "result.json")
    trajectory_rows = load_jsonl(episode_dir / "trajectory.jsonl")
    trajectory_summary = compact_trajectory_summary(trajectory_rows)
    telemetry = compute_world_shift_telemetry(result, trajectory_rows)
    image_pairs = select_episode_images(
        episode_dir,
        result=result,
        max_mid_images=max_mid_images,
        mid_frame_interval=mid_frame_interval,
    )
    packet = build_review_packet(result, trajectory_summary, telemetry, image_pairs)
    (episode_dir / "vlm_request_packet.json").write_text(
        json.dumps(packet, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    messages = build_messages(packet)

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
    )
    raw_text = completion.choices[0].message.content or ""
    parsed = parse_json_object(raw_text)
    normalized_review = normalize_review_output(result.get("task_config_name"), parsed, telemetry)
    return {
        "episode_dir": str(episode_dir.resolve()),
        "benchmark_name": result.get("benchmark_name"),
        "task_config_name": result.get("task_config_name"),
        "task_key": canonical_task_key(result.get("task_config_name")),
        "seed": result.get("seed"),
        "review_inputs": {
            "packet_path": str((episode_dir / "vlm_request_packet.json").resolve()),
            "image_labels": [item["label"] for item in packet.get("images", [])],
            "image_paths": [item["path"] for item in packet.get("images", [])],
        },
        "review": normalized_review,
        "raw_response": raw_text,
    }


def aggregate_worldgen_plan(review_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in review_rows:
        grouped[str(row.get("task_config_name", "unknown"))].append(row)

    summary_rows: List[Dict] = []
    for task_config_name, task_rows in sorted(grouped.items()):
        task_key = canonical_task_key(task_config_name)
        primary_modes = Counter()
        primary_factors = Counter()
        severity_values: List[int] = []
        visibility = Counter()
        distractors = Counter()
        path_difficulty = Counter()
        view_difficulty = Counter()
        layout_changes = Counter()
        trainable_votes = Counter()

        for row in task_rows:
            review = row.get("review", {}) or {}
            primary_modes[str(review.get("primary_failure_mode", "unknown"))] += 1
            for factor_code in review.get("primary_factors", []) or []:
                primary_factors[str(factor_code)] += 1
            try:
                severity_values.append(int(review.get("severity", 1)))
            except Exception:
                pass
            trainable_votes[bool(review.get("trainable_with_rl", True))] += 1
            suggestions = review.get("world_generation_suggestions", {}) or {}
            visibility[str(suggestions.get("visibility", "keep"))] += 1
            distractors[str(suggestions.get("distractors", "keep"))] += 1
            path_difficulty[str(suggestions.get("path_difficulty", "keep"))] += 1
            view_difficulty[str(suggestions.get("view_difficulty", "keep"))] += 1
            for item in suggestions.get("layout_changes", []) or []:
                layout_changes[str(item)] += 1

        summary_rows.append(
            {
                "task_config_name": task_config_name,
                "task_key": task_key,
                "episodes": len(task_rows),
                "primary_failure_mode_majority": primary_modes.most_common(1)[0][0] if primary_modes else "unknown",
                "primary_factors_majority": [factor for factor, _ in primary_factors.most_common(3)],
                "severity_majority": Counter(severity_values).most_common(1)[0][0] if severity_values else 1,
                "factor_levels": derive_factor_levels_from_review(
                    task_key,
                    primary_failure_mode=primary_modes.most_common(1)[0][0] if primary_modes else "unknown",
                    primary_factors=[factor for factor, _ in primary_factors.most_common(3)],
                    severity=Counter(severity_values).most_common(1)[0][0] if severity_values else 1,
                ),
                "trainable_with_rl_majority": trainable_votes.most_common(1)[0][0] if trainable_votes else True,
                "world_generation_suggestions": {
                    "visibility": visibility.most_common(1)[0][0] if visibility else "keep",
                    "distractors": distractors.most_common(1)[0][0] if distractors else "keep",
                    "path_difficulty": path_difficulty.most_common(1)[0][0] if path_difficulty else "keep",
                    "view_difficulty": view_difficulty.most_common(1)[0][0] if view_difficulty else "keep",
                    "layout_changes": [item for item, _ in layout_changes.most_common(5)] or ["keep_layout"],
                    "notes": f"Aggregated from {len(task_rows)} VLM-reviewed episode(s).",
                },
                "primary_failure_mode_counts": dict(primary_modes),
                "primary_factor_counts": dict(primary_factors),
                "severity_values": severity_values,
                "factor_vote_counts": {
                    "visibility": dict(visibility),
                    "distractors": dict(distractors),
                    "path_difficulty": dict(path_difficulty),
                    "view_difficulty": dict(view_difficulty),
                    "layout_changes": dict(layout_changes),
                },
                "factor_schema": world_factor_schema_for_task(task_key),
            }
        )
    return summary_rows


def main():
    args = parse_args()
    if OpenAI is None:
        raise ImportError(
            "openai package is required for interaction_vlm_review.py. "
            "Install it in the active environment or run this script in an environment that already has it."
        )
    run_dir = Path(args.run_dir)
    task_filter = [task.strip() for task in args.tasks.split(",") if task.strip()] or None
    review_dir = run_dir / "vlm_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.backend_url.rstrip("/"),
    )

    episode_dirs = find_episode_dirs(run_dir, task_filter=task_filter)
    review_rows: List[Dict] = []

    for episode_dir in episode_dirs:
        output_path = episode_dir / "vlm_review.json"
        if output_path.exists() and not args.overwrite:
            row = load_json(output_path)
        else:
            row = review_episode(
                client=client,
                model_id=args.model_id,
                episode_dir=episode_dir,
                max_mid_images=args.max_mid_images,
                mid_frame_interval=args.mid_frame_interval,
            )
            output_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        review_rows.append(row)
        print(json.dumps({"task_config_name": row["task_config_name"], "seed": row["seed"], "episode_status": row["review"].get("episode_status"), "primary_failure_mode": row["review"].get("primary_failure_mode")}, ensure_ascii=False))

    (review_dir / "vlm_reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    plan_rows = aggregate_worldgen_plan(review_rows)
    (review_dir / "worldgen_plan.json").write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved VLM reviews to {review_dir}")


if __name__ == "__main__":
    main()
