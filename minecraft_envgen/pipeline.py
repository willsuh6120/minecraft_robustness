from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from minecraft_envgen.adapters.minedojo import MineDojoAdapter
from minecraft_envgen.adapters.minestudio import MineStudioAdapter
from minecraft_envgen.core.contracts import (
    CompiledEnvironment,
    CurriculumDraft,
    FeedbackPacket,
    Split,
    VerificationReport,
)
from minecraft_envgen.core.prompting import PromptBuilder
from minecraft_envgen.verifiers.schema import AdapterSchemaVerifier, DraftSchemaVerifier
from minecraft_envgen.verifiers.semantic import SemanticVerifier


ADAPTERS = {
    "minedojo": MineDojoAdapter(),
    "minestudio": MineStudioAdapter(),
}


@dataclass
class CompilationOutcome:
    prompt: str
    report: VerificationReport
    compiled: CompiledEnvironment | None


def compile_curriculum(
    draft_data: Mapping[str, Any] | CurriculumDraft,
    adapter_name: str,
    feedback_data: Mapping[str, Any] | FeedbackPacket | None = None,
    split: str | Split | None = None,
    use_verifiers: bool = True,
) -> CompilationOutcome:
    if adapter_name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{adapter_name}'. Expected one of {sorted(ADAPTERS)}.")

    draft = draft_data if isinstance(draft_data, CurriculumDraft) else CurriculumDraft.from_dict(draft_data)
    feedback = None
    if feedback_data is not None:
        feedback = feedback_data if isinstance(feedback_data, FeedbackPacket) else FeedbackPacket.from_dict(feedback_data)

    chosen_split = split
    if chosen_split is None:
        chosen_split = feedback.split if feedback is not None else Split.TRAIN
    if not isinstance(chosen_split, Split):
        chosen_split = Split(chosen_split)

    prompt = PromptBuilder().build_update_prompt(draft, feedback)
    report = VerificationReport()

    if use_verifiers:
        report.extend(DraftSchemaVerifier().verify(draft))
        report.extend(SemanticVerifier().verify_draft(draft))
        if feedback is not None:
            report.extend(SemanticVerifier().verify_feedback(feedback))

    if use_verifiers and report.has_errors():
        return CompilationOutcome(prompt=prompt, report=report, compiled=None)

    compiled = ADAPTERS[adapter_name].compile(draft, chosen_split)
    if use_verifiers:
        report.extend(AdapterSchemaVerifier().verify(compiled))
        if report.has_errors():
            return CompilationOutcome(prompt=prompt, report=report, compiled=None)
    return CompilationOutcome(prompt=prompt, report=report, compiled=compiled)


def compile_for_all_adapters(
    draft_data: Mapping[str, Any] | CurriculumDraft,
    feedback_data: Mapping[str, Any] | FeedbackPacket | None = None,
    split: str | Split | None = None,
) -> Dict[str, CompilationOutcome]:
    return {
        adapter_name: compile_curriculum(
            draft_data=draft_data,
            adapter_name=adapter_name,
            feedback_data=feedback_data,
            split=split,
        )
        for adapter_name in ADAPTERS
    }
