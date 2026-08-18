"""NLI-based hallucination scoring and factual grounding checks."""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .severity import RuleResult, Severity


class NLIClassifier:
    """Contradiction/entailment classifier with NLI model or fallback."""

    def __init__(self, model: Optional[object] = None):
        """``model``: callable(premise, hypothesis) -> {'label': 'entailment'|'neutral'|'contradiction', 'score': float}"""
        self.model = model
        self.logger = logging.getLogger(__name__)

    def predict(self, premise: str, hypothesis: str) -> Dict:
        """Classify the relationship between premise and hypothesis."""
        if self.model is not None:
            try:
                return self.model(premise, hypothesis)
            except Exception as e:
                self.logger.warning(f"NLI model failed ({e}); falling back to lexical")
        return self._lexical_predict(premise, hypothesis)

    def _lexical_predict(self, premise: str, hypothesis: str) -> Dict:
        """Deterministic lexical NLI fallback."""
        p_tokens = set(self._tokens(premise))
        h_tokens = set(self._tokens(hypothesis))
        if not h_tokens:
            return {"label": "neutral", "score": 0.5}
        overlap = len(p_tokens.intersection(h_tokens)) / len(h_tokens)
        negation_hits = ["not", "never", "no", "nothing", "none", "without", "cannot"]
        hyp_has_negation = any(n in hypothesis.lower().split() for n in negation_hits)
        pre_has_negation = any(n in premise.lower().split() for n in negation_hits)
        negations_differ = hyp_has_negation != pre_has_negation

        if overlap >= 0.6 and not negations_differ:
            return {"label": "entailment", "score": overlap}
        if overlap >= 0.35 and negations_differ:
            return {"label": "contradiction", "score": 1.0 - overlap}
        if overlap <= 0.1:
            return {"label": "neutral", "score": 0.5}
        return {"label": "neutral", "score": overlap}

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"[a-z']+", text.lower())]


class HallucinationScorer:
    """Scores generated claims against retrieved context (NLI grounding)."""

    def __init__(self, nli: Optional[NLIClassifier] = None,
                 entailment_threshold: float = 0.5):
        self.nli = nli or NLIClassifier()
        self.entailment_threshold = entailment_threshold

    def split_claims(self, answer: str) -> List[str]:
        """Split an answer into atomic claims (sentences)."""
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [p for p in parts if p]

    def score_answer(self, answer: str, contexts: List[str]) -> Dict:
        """Per-claim NLI scores against the concatenated context."""
        claims = self.split_claims(answer)
        context = " ".join(contexts)
        per_claim = []
        for claim in claims:
            result = self.nli.predict(context, claim)
            per_claim.append({"claim": claim, "label": result["label"],
                              "score": result["score"]})
        supported = sum(1 for c in per_claim if c["label"] == "entailment")
        contradicted = sum(1 for c in per_claim if c["label"] == "contradiction")
        n = len(per_claim)
        return {
            "num_claims": n,
            "supported": supported,
            "contradicted": contradicted,
            "hallucination_score": (contradicted / n) if n else 0.0,
            "grounding_score": (supported / n) if n else 1.0,
            "per_claim": per_claim,
        }

    def check(self, answer: str, contexts: List[str]) -> RuleResult:
        """Run the hallucination/grounding rule."""
        result = self.score_answer(answer, contexts)
        if result["num_claims"] == 0:
            return RuleResult(rule_name="nli_grounding", passed=True,
                              severity=Severity.INFO, message="Empty answer.",
                              details=result, score=0.0)
        if result["hallucination_score"] >= 0.5:
            return RuleResult(rule_name="nli_grounding", passed=False,
                              severity=Severity.HIGH,
                              message=f"Possible hallucination: {result['contradicted']}/{result['num_claims']} claims contradicted.",
                              details=result, score=result["hallucination_score"])
        if result["grounding_score"] < self.entailment_threshold:
            return RuleResult(rule_name="nli_grounding", passed=False,
                              severity=Severity.MEDIUM,
                              message=f"Weak factual grounding ({result['grounding_score']:.2f}).",
                              details=result, score=result["grounding_score"])
        return RuleResult(rule_name="nli_grounding", passed=True,
                          severity=Severity.INFO,
                          message=f"Answer grounded ({result['grounding_score']:.2f}).",
                          details=result, score=result["grounding_score"])
