from __future__ import annotations

import json
import pickle
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import av
import numpy as np
from PIL import Image

from minecraft_envgen.core.contracts import FailureEvidence, FeedbackPacket, ResetMode, Split


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_as_jsonable(payload), handle, indent=2)


def _to_number(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(np.sum(value))
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _matching_event_total(info_path: str | Path, success_key: str, success_regex: str) -> tuple[bool, float, List[Dict[str, Any]]]:
    infos = _load_pickle(info_path)
    if not infos:
        return False, 0.0, infos
    last_info = infos[-1]
    event_bucket = last_info.get(success_key, {})
    if not hasattr(event_bucket, "items"):
        return False, 0.0, infos

    total = 0.0
    for event_name, value in event_bucket.items():
        if re.match(success_regex, event_name):
            total += _to_number(value)
    return True, total, infos


def summarize_episodes(
    episodes: Iterable[Mapping[str, Any]],
    success_key: str,
    success_regex: str,
    success_num: int = 1,
) -> Dict[str, Any]:
    episode_summaries: List[Dict[str, Any]] = []
    num_yes = 0
    total_matches = 0.0

    for index, episode in enumerate(episodes):
        valid, total, _ = _matching_event_total(episode["info_path"], success_key, success_regex)
        success = bool(valid and total >= success_num)
        if success:
            num_yes += 1
        total_matches += total
        episode_summaries.append(
            {
                "episode_index": index,
                "success": success,
                "matched_total": total,
                "info_path": episode["info_path"],
                "action_path": episode["action_path"],
                "video_path": episode.get("video_path"),
            }
        )

    num_episodes = len(episode_summaries)
    success_rate = (num_yes / num_episodes) if num_episodes else 0.0
    mean_matches = (total_matches / num_episodes) if num_episodes else 0.0
    return {
        "num_yes": num_yes,
        "num_episodes": num_episodes,
        "yes_rate": f"{success_rate * 100:.2f}%",
        "success_rate": success_rate,
        "mean_match_count": mean_matches,
        "episodes": episode_summaries,
    }


def _read_video_tail(video_path: str | Path, tail_frames: int) -> List[np.ndarray]:
    frames: deque[np.ndarray] = deque(maxlen=tail_frames)
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return list(frames)


def _write_video(frames: List[np.ndarray], output_path: Path, fps: int = 30) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_feedback_packet(
    episodes: Iterable[Mapping[str, Any]],
    run_id: str,
    output_dir: str | Path,
    split: str | Split,
    reset_mode: str | ResetMode,
    success_key: str,
    success_regex: str,
    success_num: int = 1,
    tail_window: int = 128,
    max_failure_episodes: int = 3,
    notes: str | None = None,
) -> FeedbackPacket:
    episodes = list(episodes)
    summary = summarize_episodes(episodes, success_key=success_key, success_regex=success_regex, success_num=success_num)

    output_root = Path(output_dir)
    evidence_dir = output_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    failure_evidence: List[FailureEvidence] = []
    event_traces: List[Dict[str, Any]] = []
    inventory_traces: List[Dict[str, Any]] = []
    privileged_traces: List[Dict[str, Any]] = []

    failed = [item for item in summary["episodes"] if not item["success"]][:max_failure_episodes]
    for failed_episode in failed:
        episode_index = failed_episode["episode_index"]
        infos = _load_pickle(failed_episode["info_path"])
        actions = _load_pickle(failed_episode["action_path"])
        start = max(0, len(infos) - tail_window)
        tail_infos = infos[start:]
        tail_actions = actions[max(0, len(actions) - tail_window) :]

        event_traces.append(
            {
                "episode_index": episode_index,
                "tail_start_step": start,
                "tail_end_step": len(infos) - 1,
                "infos": tail_infos,
                "actions": tail_actions,
            }
        )

        inventory_slice = []
        privileged_slice = []
        for step_offset, info in enumerate(tail_infos, start=start):
            if "inventory" in info:
                inventory_slice.append({"step": step_offset, "inventory": info["inventory"]})
            privileged_payload = {}
            for key in ("voxels", "location_stats", "health", "food_level", "player_pos"):
                if key in info:
                    privileged_payload[key] = info[key]
            if privileged_payload:
                privileged_slice.append({"step": step_offset, "payload": privileged_payload})

        if inventory_slice:
            inventory_traces.append({"episode_index": episode_index, "trace": inventory_slice})
        if privileged_slice:
            privileged_traces.append({"episode_index": episode_index, "trace": privileged_slice})

        video_path = failed_episode.get("video_path")
        if video_path:
            frames = _read_video_tail(video_path, tail_frames=tail_window)
            if frames:
                clip_path = evidence_dir / f"episode_{episode_index}_tail.mp4"
                frame_path = evidence_dir / f"episode_{episode_index}_last.png"
                _write_video(frames, clip_path)
                Image.fromarray(frames[-1]).save(frame_path)
                failure_evidence.append(
                    FailureEvidence(
                        modality="video_clip",
                        summary=(
                            f"Final {len(frames)} frames from failed episode {episode_index} "
                            f"for success target '{success_regex}'."
                        ),
                        uri=str(clip_path.resolve()),
                        metadata={"episode_index": episode_index, "tail_frames": len(frames)},
                    )
                )
                failure_evidence.append(
                    FailureEvidence(
                        modality="image_frame",
                        summary=f"Last frame of failed episode {episode_index}.",
                        uri=str(frame_path.resolve()),
                        metadata={"episode_index": episode_index},
                    )
                )

    if event_traces:
        event_path = evidence_dir / "event_trace.json"
        _write_json(event_path, event_traces)
        failure_evidence.append(
            FailureEvidence(
                modality="event_trace",
                summary=f"Tail action and info traces for {len(event_traces)} failed episodes.",
                uri=str(event_path.resolve()),
                metadata={"episodes": [item["episode_index"] for item in failed], "tail_window": tail_window},
            )
        )

    if inventory_traces:
        inventory_path = evidence_dir / "inventory_trace.json"
        _write_json(inventory_path, inventory_traces)
        failure_evidence.append(
            FailureEvidence(
                modality="inventory_trace",
                summary=f"Tail inventory traces for {len(inventory_traces)} failed episodes.",
                uri=str(inventory_path.resolve()),
                metadata={"episodes": [item["episode_index"] for item in failed]},
            )
        )

    if privileged_traces:
        privileged_path = evidence_dir / "privileged_trace.json"
        _write_json(privileged_path, privileged_traces)
        failure_evidence.append(
            FailureEvidence(
                modality="privileged_observation",
                summary=f"Tail privileged observations for {len(privileged_traces)} failed episodes.",
                uri=str(privileged_path.resolve()),
                metadata={"episodes": [item["episode_index"] for item in failed]},
            )
        )

    scalar_metrics = {
        "success_rate": summary["success_rate"],
        "success_count": float(summary["num_yes"]),
        "mean_match_count": summary["mean_match_count"],
    }
    verifier_notes = [
        "Packet built from MineStudio rollout artifacts (video, info, actions).",
        f"Success criterion: info['{success_key}'] matched against /{success_regex}/ >= {success_num}.",
    ]

    return FeedbackPacket(
        run_id=run_id,
        split=split,
        reset_mode=reset_mode,
        scalar_metrics=scalar_metrics,
        failure_evidence=failure_evidence,
        verifier_notes=verifier_notes,
        episode_count=summary["num_episodes"],
        notes=notes,
    )
