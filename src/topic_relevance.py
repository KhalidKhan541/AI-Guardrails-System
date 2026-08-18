"""Topic relevance checker: ensures inputs stay within allowed topics."""

import logging
from typing import Dict, List, Optional

from .severity import RuleResult, Severity

_GENERIC_TOPIC_WORDS = {
    "instructions", "information", "details", "things", "stuff", "ways",
    "tips", "guide", "tutorial", "about", "into", "with", "what", "how",
}


class TopicRelevanceChecker:
    """Checks whether an input is relevant to an allowed topic set.

    Uses token-overlap scoring with optional embedding-based similarity
    via a callable ``embedder(query) -> np.ndarray``.
    """

    def __init__(self, allowed_topics: Optional[List[str]] = None,
                 blocked_topics: Optional[List[str]] = None,
                 threshold: float = 0.2, embedder: Optional[object] = None):
        self.allowed_topics = allowed_topics or []
        self.blocked_topics = blocked_topics or []
        self.threshold = threshold
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)

    def check_blocked(self, text: str) -> Dict:
        """Check if text matches any blocked topic keyword."""
        lower = text.lower()
        hits = []
        for topic in self.blocked_topics:
            if topic.lower() in lower:
                hits.append(topic)
                continue
            # Match key tokens, tolerating inflections ("hack" ~ "hacking")
            tokens = set(self._clean_tokens(topic))
            text_tokens = set(self._clean_tokens(text))
            for tok in tokens:
                if len(tok) < 4 or tok in _GENERIC_TOPIC_WORDS:
                    continue
                if any(t.startswith(tok) or tok.startswith(t) for t in text_tokens):
                    hits.append(topic)
                    break
        return {"blocked": bool(hits), "hits": hits}

    def relevance(self, text: str) -> float:
        """Relevance score (0-1) of text against allowed topics."""
        if not self.allowed_topics:
            return 1.0
        if self.embedder is not None:
            try:
                return self._embedding_relevance(text)
            except Exception:
                pass
        return self._lexical_relevance(text)

    def _lexical_relevance(self, text: str) -> float:
        tokens = set(self._clean_tokens(text))
        if not tokens:
            return 0.0
        best = 0.0
        for topic in self.allowed_topics:
            topic_tokens = set(self._clean_tokens(topic))
            if not topic_tokens:
                continue
            overlap = len(tokens.intersection(topic_tokens))
            best = max(best, overlap / len(topic_tokens))
        return best

    @staticmethod
    def _clean_tokens(text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9']+", text.lower())

    def _embedding_relevance(self, text: str) -> float:
        import numpy as np
        q = self.embedder(text)
        scores = []
        for topic in self.allowed_topics:
            t = self.embedder(topic)
            q_n = q / (np.linalg.norm(q) + 1e-9)
            t_n = t / (np.linalg.norm(t) + 1e-9)
            scores.append(float(np.dot(q_n, t_n)))
        return max(scores) if scores else 0.0

    def check(self, text: str) -> RuleResult:
        """Run the topic relevance rule."""
        blocked = self.check_blocked(text)
        if blocked["blocked"]:
            return RuleResult(
                rule_name="topic_relevance",
                passed=False,
                severity=Severity.HIGH,
                message=f"Blocked topic detected: {blocked['hits']}",
                details=blocked,
                score=1.0,
            )
        score = self.relevance(text)
        if score < self.threshold:
            return RuleResult(
                rule_name="topic_relevance",
                passed=False,
                severity=Severity.MEDIUM,
                message=f"Input is off-topic (relevance {score:.2f}).",
                details={"score": score},
                score=score,
            )
        return RuleResult(
            rule_name="topic_relevance",
            passed=True,
            severity=Severity.INFO,
            message=f"Topic relevance passed (score {score:.2f}).",
            details={"score": score},
            score=score,
        )
