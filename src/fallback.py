"""Fail-safe fallback responses for guardrail violations."""

import logging
from typing import Dict, Optional

from .severity import GuardrailDecision, Severity

DEFAULT_FALLBACKS = {
    "block": "I'm sorry, but I can't help with that request. If you have another question, I'd be glad to help.",
    "off_topic": "I'm here to help with the topics I'm allowed to discuss. Let me know if you have a question in that area.",
    "unsupported": "I don't have enough verified information to answer that confidently.",
    "format_error": "I couldn't format that response as required. Let me try again with the correct format.",
}


class FallbackManager:
    """Selects fail-safe responses based on guardrail decisions."""

    def __init__(self, fallbacks: Optional[Dict[str, str]] = None):
        self.fallbacks = fallbacks or dict(DEFAULT_FALLBACKS)
        self.logger = logging.getLogger(__name__)

    def response_for(self, decision: GuardrailDecision,
                     input_text: Optional[str] = None) -> str:
        """Pick a fallback response for the worst violation."""
        worst = decision.max_severity() if decision.results else Severity.INFO
        first_violation = decision.violations()[0] if decision.violations() else None
        if first_violation:
            name = first_violation.rule_name
            if name == "jailbreak_detector":
                return self.fallbacks["block"]
            if name == "toxicity_classifier":
                return self.fallbacks["block"]
            if name == "pii_scanner":
                return self.fallbacks["block"] + " (Note: your input may contain personal data; avoid sharing it.)"
            if name == "topic_relevance":
                return self.fallbacks["off_topic"]
            if name == "nli_grounding":
                return self.fallbacks["unsupported"]
            if name == "format_validation":
                return self.fallbacks["format_error"]
        if worst.value >= Severity.HIGH.value:
            return self.fallbacks["block"]
        return self.fallbacks["unsupported"]
