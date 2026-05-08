from __future__ import annotations

from typing import List

from minecraft_envgen.core.catalog import SKILL_TO_FAMILIES, infer_task_class
from minecraft_envgen.core.contracts import (
    CurriculumDraft,
    FeedbackPacket,
    InterventionFamily,
    ResetMode,
    Severity,
    Split,
    VerificationMessage,
)


class SemanticVerifier:
    def verify_draft(self, draft: CurriculumDraft) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []
        inferred_task_class = infer_task_class(draft.target_skill)
        if draft.task_class != inferred_task_class:
            messages.append(
                VerificationMessage(
                    severity=Severity.WARNING,
                    location="draft.task_class",
                    message=(
                        f"task_class '{draft.task_class}' does not match the inferred class "
                        f"'{inferred_task_class}' for skill '{draft.target_skill}'."
                    ),
                )
            )

        allowed_families = SKILL_TO_FAMILIES.get(draft.target_skill)
        if allowed_families and draft.intervention_family not in allowed_families:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.intervention_family",
                    message=(
                        f"Intervention family '{draft.intervention_family.value}' is inconsistent "
                        f"with skill '{draft.target_skill}'."
                    ),
                )
            )

        if draft.intervention_family == InterventionFamily.RESOURCE_BOOST and "resource_multipliers" not in draft.interventions:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions.resource_multipliers",
                    message="resource_boost drafts must include resource_multipliers.",
                )
            )

        if draft.intervention_family == InterventionFamily.INVENTORY_BOOTSTRAP and not draft.initial_inventory:
            messages.append(
                VerificationMessage(
                    severity=Severity.WARNING,
                    location="draft.initial_inventory",
                    message="inventory_bootstrap usually requires non-empty initial_inventory.",
                )
            )

        if draft.intervention_family == InterventionFamily.MOB_PRESSURE and "mob_spawns" not in draft.interventions:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="draft.interventions.mob_spawns",
                    message="mob_pressure drafts must include mob_spawns.",
                )
            )

        if draft.intervention_family == InterventionFamily.PRIVILEGED_HINT and not draft.privileged_observations:
            messages.append(
                VerificationMessage(
                    severity=Severity.WARNING,
                    location="draft.privileged_observations",
                    message="privileged_hint drafts should specify privileged_observations.",
                )
            )

        return messages

    def verify_feedback(self, feedback: FeedbackPacket) -> List[VerificationMessage]:
        messages: List[VerificationMessage] = []

        if feedback.split == Split.EVAL and feedback.reset_mode != ResetMode.CLEAN:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="feedback.reset_mode",
                    message="Evaluation feedback must come from clean resets, not fast resets.",
                )
            )

        if not feedback.scalar_metrics:
            messages.append(
                VerificationMessage(
                    severity=Severity.ERROR,
                    location="feedback.scalar_metrics",
                    message="Feedback packet must contain scalar metrics.",
                )
            )

        low_performance = False
        for metric_name, value in feedback.scalar_metrics.items():
            if not isinstance(value, (int, float)):
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"feedback.scalar_metrics.{metric_name}",
                        message="Each scalar metric must be numeric.",
                    )
                )
                continue
            if value < 0:
                messages.append(
                    VerificationMessage(
                        severity=Severity.ERROR,
                        location=f"feedback.scalar_metrics.{metric_name}",
                        message="Scalar metrics must be non-negative.",
                    )
                )
            if (0.0 <= value <= 1.0 and value < 0.2) or (1.0 < value <= 100.0 and value < 20.0):
                low_performance = True

        if low_performance and not feedback.failure_evidence:
            messages.append(
                VerificationMessage(
                    severity=Severity.WARNING,
                    location="feedback.failure_evidence",
                    message="Low performance was reported without failure evidence. LLM updates may overfit to scalar noise.",
                )
            )

        for index, note in enumerate(feedback.verifier_notes):
            if not isinstance(note, str) or not note.strip():
                messages.append(
                    VerificationMessage(
                        severity=Severity.WARNING,
                        location=f"feedback.verifier_notes[{index}]",
                        message="Verifier notes should be non-empty strings.",
                    )
                )

        return messages
