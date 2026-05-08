from __future__ import annotations

import json

from .contracts import CurriculumDraft, FeedbackPacket


class PromptBuilder:
    def build_update_prompt(self, draft: CurriculumDraft, feedback: FeedbackPacket | None = None) -> str:
        sections = [
            "You are a curriculum designer for Minecraft training environments.",
            "Produce one JSON object that follows the exact contract shown below.",
            "Keep train/eval reset semantics explicit. Fast reset is acceptable for training, but evaluation must remain clean and uncontaminated.",
            "",
            "Output contract:",
            json.dumps(
                {
                    "draft_id": "string",
                    "target_skill": draft.target_skill,
                    "intervention_family": "enum",
                    "difficulty": "easy|medium|hard",
                    "task_class": draft.task_class,
                    "interventions": {"key": "value"},
                    "initial_inventory": {"item_name": 1},
                    "privileged_observations": ["inventory", "voxel"],
                    "rationale": "brief explanation tied to the evidence",
                    "metadata": {"task_id": "engine_specific_optional_binding"},
                },
                indent=2,
            ),
            "",
            "Current draft:",
            json.dumps(draft.to_dict(), indent=2),
        ]

        if feedback is not None:
            sections.extend(
                [
                    "",
                    "Feedback packet:",
                    json.dumps(feedback.to_dict(), indent=2),
                    "",
                    "Use the scalar metrics, failure evidence summaries, and verifier notes jointly.",
                    "If the metrics are low but evidence is missing, prefer conservative interventions and say that evidence coverage is incomplete.",
                    "Prefer lightweight privileged observations such as inventory by default. Only use voxel if the failure clearly requires local spatial geometry.",
                ]
            )

        sections.extend(
            [
                "",
                "Do not emit markdown fences. Return JSON only.",
            ]
        )
        return "\n".join(sections)
