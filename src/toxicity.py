"""Toxicity classifier: lexicon-based with optional model backing."""

import logging
from typing import Dict, List, Optional

from .severity import RuleResult, Severity

TOXIC_TERMS = [
    "stupid", "idiot", "dumb", "hate", "hateful", "kill yourself", "shut up",
    "worthless", "trash", "garbage", "loser", "freak", "screw you", "f you",
    "moron", "pathetic", "disgusting", "repulsive", "sickening", "nasty",
]

HARASSMENT_PATTERNS = [
    r"\byou\s+(are\s+)?(a\s+)?(stupid|idiot|worthless|loser)\b",
    r"\b(go|get)\s+(to\s+)?(hell|die)\b",
    r"\bkill\s+yourself\b",
    r"\bfuck\s+(you|off|this|that)\b",
]


class ToxicityClassifier:
    """Classifies text toxicity with lexicon + pattern scoring.

    Uses a deterministic lexicon (offline) and can optionally delegate to
    a HuggingFace toxicity model via the ``model`` callable.
    """

    def __init__(self, model: Optional[object] = None, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def classify(self, text: str) -> Dict:
        """Return toxicity score (0-1) and flagged terms."""
        if self.model is not None:
            try:
                return self._model_score(text)
            except Exception as e:
                self.logger.warning(f"Model toxicity failed ({e}); falling back to lexicon")

        lower = text.lower()
        hits = [t for t in TOXIC_TERMS if t in lower]

        import re
        pattern_hits = []
        for pat in HARASSMENT_PATTERNS:
            if re.search(pat, lower):
                pattern_hits.append(pat)

        score = min(0.35 * len(hits) + 0.45 * len(pattern_hits), 1.0)
        if "hate" in lower or "kill" in lower:
            score = max(score, 0.8)
        if len(hits) >= 2:
            score = max(score, 0.7)
        return {"score": score, "terms": hits, "patterns": pattern_hits}

    def _model_score(self, text: str) -> Dict:
        out = self.model(text)
        return {"score": float(out.get("score", 0.0)), "terms": [], "patterns": []}

    def check(self, text: str) -> RuleResult:
        """Run the toxicity rule."""
        result = self.classify(text)
        score = result["score"]
        if score >= self.threshold:
            return RuleResult(
                rule_name="toxicity_classifier",
                passed=False,
                severity=Severity.HIGH if score >= 0.8 else Severity.MEDIUM,
                message=f"Toxic content detected (score {score:.2f}).",
                details=result,
                score=score,
            )
        return RuleResult(
            rule_name="toxicity_classifier",
            passed=True,
            severity=Severity.INFO,
            message=f"Toxicity check passed (score {score:.2f}).",
            details=result,
            score=score,
        )
