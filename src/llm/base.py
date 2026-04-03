from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import FeedbackBundle


class BaseLLMClient(ABC):
    @abstractmethod
    async def summarize_feedback(self, feedback: FeedbackBundle) -> tuple[str, list[str]]:
        """Return a patch rationale and a list of suggestions."""
