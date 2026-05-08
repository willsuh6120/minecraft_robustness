from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any, Dict

from minecraft_envgen.core.contracts import (
    CompiledEnvironment,
    CurriculumDraft,
    EngineTarget,
    ResetMode,
    Split,
)


class EnvironmentAdapter(ABC):
    engine: EngineTarget
    dependency_name: str

    def dependency_status(self) -> Dict[str, Any]:
        return {
            "module": self.dependency_name,
            "installed": importlib.util.find_spec(self.dependency_name) is not None,
        }

    def default_reset_mode(self, split: Split) -> ResetMode:
        if split == Split.TRAIN:
            return ResetMode.FAST
        return ResetMode.CLEAN

    @abstractmethod
    def compile(self, draft: CurriculumDraft, split: Split) -> CompiledEnvironment:
        raise NotImplementedError
