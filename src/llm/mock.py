from __future__ import annotations

from collections import Counter

from core.models import FeedbackBundle
from llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    async def summarize_feedback(self, feedback: FeedbackBundle) -> tuple[str, list[str]]:
        counts = Counter(finding.category for finding in feedback.all_findings)
        if not counts:
            return (
                "No actionable findings were produced by review or debug steps.",
                ["No patch necessary."],
            )

        summary_parts = [f"{category}: {count}" for category, count in sorted(counts.items())]
        suggestions = [
            f"Prioritize {finding.rule_id} in {finding.path or 'repository root'}"
            for finding in feedback.all_findings[:5]
        ]
        return (
            "Feedback summary derived from deterministic mock reasoning: "
            + ", ".join(summary_parts),
            suggestions,
        )

