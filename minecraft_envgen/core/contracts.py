from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping


class EngineTarget(str, Enum):
    MINEDOJO = "minedojo"
    MINESTUDIO = "minestudio"


class Split(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


class ResetMode(str, Enum):
    FAST = "fast"
    CLEAN = "clean"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InterventionFamily(str, Enum):
    RESOURCE_BOOST = "resource_boost"
    INVENTORY_BOOTSTRAP = "inventory_bootstrap"
    SPAWN_CONTROL = "spawn_control"
    MOB_PRESSURE = "mob_pressure"
    TERRAIN_CONTROL = "terrain_control"
    TIME_CONTROL = "time_control"
    WEATHER_CONTROL = "weather_control"
    PRIVILEGED_HINT = "privileged_hint"
    SAFETY_SCAFFOLD = "safety_scaffold"


class EvidenceModality(str, Enum):
    TEXT = "text"
    EVENT_TRACE = "event_trace"
    INVENTORY_TRACE = "inventory_trace"
    VIDEO_CLIP = "video_clip"
    IMAGE_FRAME = "image_frame"
    PRIVILEGED_OBSERVATION = "privileged_observation"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


def _coerce_enum(enum_cls: type[Enum], value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_ready(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class FailureEvidence:
    modality: EvidenceModality | str
    summary: str
    uri: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = _coerce_enum(EvidenceModality, self.modality)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FailureEvidence":
        return cls(
            modality=raw["modality"],
            summary=raw["summary"],
            uri=raw.get("uri"),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(self)


@dataclass
class FeedbackPacket:
    run_id: str
    split: Split | str
    reset_mode: ResetMode | str
    scalar_metrics: Dict[str, float]
    failure_evidence: List[FailureEvidence] = field(default_factory=list)
    verifier_notes: List[str] = field(default_factory=list)
    episode_count: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        self.split = _coerce_enum(Split, self.split)
        self.reset_mode = _coerce_enum(ResetMode, self.reset_mode)
        self.failure_evidence = [
            item if isinstance(item, FailureEvidence) else FailureEvidence.from_dict(item)
            for item in self.failure_evidence
        ]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeedbackPacket":
        return cls(
            run_id=raw["run_id"],
            split=raw["split"],
            reset_mode=raw["reset_mode"],
            scalar_metrics=dict(raw.get("scalar_metrics", {})),
            failure_evidence=list(raw.get("failure_evidence", [])),
            verifier_notes=list(raw.get("verifier_notes", [])),
            episode_count=raw.get("episode_count"),
            notes=raw.get("notes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(self)


@dataclass
class CurriculumDraft:
    draft_id: str
    target_skill: str
    intervention_family: InterventionFamily | str
    difficulty: Difficulty | str
    task_class: str
    interventions: Dict[str, Any]
    initial_inventory: Dict[str, int] = field(default_factory=dict)
    privileged_observations: List[str] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intervention_family = _coerce_enum(InterventionFamily, self.intervention_family)
        self.difficulty = _coerce_enum(Difficulty, self.difficulty)
        self.initial_inventory = {str(k): int(v) for k, v in self.initial_inventory.items()}
        self.privileged_observations = [str(item) for item in self.privileged_observations]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CurriculumDraft":
        return cls(
            draft_id=raw["draft_id"],
            target_skill=raw["target_skill"],
            intervention_family=raw["intervention_family"],
            difficulty=raw["difficulty"],
            task_class=raw["task_class"],
            interventions=dict(raw.get("interventions", {})),
            initial_inventory=dict(raw.get("initial_inventory", {})),
            privileged_observations=list(raw.get("privileged_observations", [])),
            rationale=raw.get("rationale", ""),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(self)


@dataclass
class CompiledEnvironment:
    engine: EngineTarget | str
    split: Split | str
    reset_mode: ResetMode | str
    runtime: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.engine = _coerce_enum(EngineTarget, self.engine)
        self.split = _coerce_enum(Split, self.split)
        self.reset_mode = _coerce_enum(ResetMode, self.reset_mode)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(self)


@dataclass
class VerificationMessage:
    severity: Severity | str
    location: str
    message: str

    def __post_init__(self) -> None:
        self.severity = _coerce_enum(Severity, self.severity)

    def to_dict(self) -> Dict[str, str]:
        return _json_ready(self)


@dataclass
class VerificationReport:
    messages: List[VerificationMessage] = field(default_factory=list)

    def add(self, severity: Severity | str, location: str, message: str) -> None:
        self.messages.append(VerificationMessage(severity=severity, location=location, message=message))

    def extend(self, items: Iterable[VerificationMessage]) -> None:
        self.messages.extend(items)

    def has_errors(self) -> bool:
        return any(item.severity == Severity.ERROR for item in self.messages)

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {"messages": [_json_ready(item) for item in self.messages]}
