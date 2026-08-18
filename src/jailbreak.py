"""Jailbreak prompt pattern detector."""

import logging
import re
from typing import Dict, List, Optional

from .severity import RuleResult, Severity

JAILBREAK_PATTERNS = [
    (r"\bignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions|rules|prompt|system)\b", "instruction_override"),
    (r"\byou\s+(are|now)\s+(a|an|no)\s+(different\s+)?(persona|character|model|gpt|dan)\b", "persona_swap"),
    (r"\bdan\b", "dan_mode"),
    (r"\b(act|behave)\s+as\s+(if\s+)?(you\s+are\s+)?(an\s+)?(unrestricted|jailbroken|without\s+filter)\b", "unrestricted_mode"),
    (r"\bbypass\s+(the\s+)?(safety|filter|guardrails|rules|alignment)\b", "bypass_attempt"),
    (r"\bsimulate\s+(a\s+)?(sandbox|jailbreak|unfiltered|no\s+restrictions)\b", "simulation_trick"),
    (r"\b(do\s+it\s+anyway|just\s+this\s+once|for\s+research\s+purposes)\b", "pressure_tactic"),
    (r"\brepeat\s+(everything|all)\s+(above|before)\s+(without|without\s+the)\s+(rules|filters)\b", "echo_exploit"),
    (r"\b(can\s+you|please|now)\s+tell\s+me\s+how\s+to\s+(make|build|create)\s+(a|an)?\s*(bomb|weapon|drug|explosive)\b", "harmful_request"),
]

FALLBACK_TRIGGERS = ["roleplay as", "pretend you are", "out of character"]


class JailbreakDetector:
    """Detects jailbreak and prompt-injection patterns."""

    def __init__(self, threshold: float = 0.3, patterns: Optional[List[tuple]] = None):
        self.patterns = patterns or JAILBREAK_PATTERNS
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def scan(self, text: str) -> Dict:
        """Scan text for jailbreak patterns. Returns hits + composite score."""
        lower = text.lower()
        hits = []
        for pattern, label in self.patterns:
            if re.search(pattern, lower):
                hits.append({"pattern": label, "regex": pattern})
        for trigger in FALLBACK_TRIGGERS:
            if trigger in lower:
                hits.append({"pattern": "roleplay_trick", "regex": trigger})
        score = min(len(hits) * 0.4, 1.0)
        return {"score": score, "hits": hits, "count": len(hits)}

    def check(self, text: str) -> RuleResult:
        """Run the jailbreak detection rule."""
        result = self.scan(text)
        score = result["score"]
        if score >= self.threshold:
            return RuleResult(
                rule_name="jailbreak_detector",
                passed=False,
                severity=Severity.CRITICAL if score >= 0.7 else Severity.HIGH,
                message=f"Jailbreak pattern detected: {[h['pattern'] for h in result['hits']]}",
                details=result,
                score=score,
            )
        return RuleResult(
            rule_name="jailbreak_detector",
            passed=True,
            severity=Severity.INFO,
            message="No jailbreak patterns detected.",
            details=result,
            score=score,
        )
