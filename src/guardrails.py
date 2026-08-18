"""Two-layer guardrail orchestrator: input validation then output validation."""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from .audit_log import AuditEntry, AuditLogger
from .fallback import FallbackManager
from .rule_engine import Action, RuleEngine, RuleSpec
from .severity import GuardrailDecision, RuleResult, Severity


class GuardrailConfig:
    """Configuration for the two-layer guardrail pipeline."""

    def __init__(self,
                 allowed_topics: Optional[List[str]] = None,
                 blocked_topics: Optional[List[str]] = None,
                 output_schema: Optional[Dict] = None,
                 max_output_length: int = 8000,
                 rule_overrides: Optional[Dict[str, Dict]] = None):
        self.allowed_topics = allowed_topics or []
        self.blocked_topics = blocked_topics or []
        self.output_schema = output_schema
        self.max_output_length = max_output_length
        self.rule_overrides = rule_overrides or {}


class Guardrails:
    """Orchestrates input layer + output layer validation with audit logging."""

    def __init__(self, config: Optional[GuardrailConfig] = None,
                 audit_logger: Optional[AuditLogger] = None,
                 llm: Optional[Callable[[str], str]] = None,
                 rules: Optional[List[RuleSpec]] = None):
        self.config = config or GuardrailConfig()
        self.audit = audit_logger or AuditLogger()
        self.llm = llm or self._default_llm
        self.logger = logging.getLogger(__name__)

        from .toxicity import ToxicityClassifier
        from .jailbreak import JailbreakDetector
        from .pii_scanner import PIIScanner
        from .topic_relevance import TopicRelevanceChecker
        from .nli_grounding import HallucinationScorer
        from .format_validator import FormatValidator

        self.toxicity = ToxicityClassifier()
        self.jailbreak = JailbreakDetector()
        self.pii = PIIScanner()
        self.topic = TopicRelevanceChecker(
            allowed_topics=self.config.allowed_topics,
            blocked_topics=self.config.blocked_topics,
        )
        self.grounding = HallucinationScorer()
        self.format_validator = FormatValidator(
            schema=self.config.output_schema,
            max_length=self.config.max_output_length,
        )
        self.fallback = FallbackManager()

        self.engine = RuleEngine()
        for spec in (rules or self._default_rules()):
            self.engine.register(spec)

    def _default_rules(self) -> List[RuleSpec]:
        return [
            RuleSpec("toxicity_classifier", "input", lambda t: self.toxicity.check(t),
                     severity_floor=Severity.MEDIUM),
            RuleSpec("jailbreak_detector", "input", lambda t: self.jailbreak.check(t),
                     severity_floor=Severity.HIGH),
            RuleSpec("pii_scanner", "input", lambda t: self.pii.check(t),
                     severity_floor=Severity.MEDIUM),
            RuleSpec("topic_relevance", "input", lambda t: self.topic.check(t),
                     severity_floor=Severity.LOW),
            RuleSpec("nli_grounding", "output",
                     lambda o: self.grounding.check(o["answer"], o["contexts"]),
                     severity_floor=Severity.MEDIUM),
            RuleSpec("format_validation", "output",
                     lambda o: self.format_validator.check(o["raw_answer"]),
                     severity_floor=Severity.MEDIUM),
        ]

    def _default_llm(self, prompt: str) -> str:
        return ("I don't have enough verified information to answer that confidently. "
                "Please rephrase your question.")

    # -- pipeline -----------------------------------------------------------

    def protect_input(self, text: str) -> GuardrailDecision:
        """Validate user input through the input layer."""
        return self.engine.evaluate_layer("input", text)

    def protect_output(self, answer: str, contexts: List[str]) -> GuardrailDecision:
        """Validate model output through the output layer."""
        # Extract the semantic answer text from JSON envelopes so NLI grounding
        # scores the actual content, not the JSON syntax. The format validator
        # still receives the raw output to validate its structure.
        extracted = answer
        if self.config.output_schema:
            import json
            try:
                data = json.loads(answer)
                if isinstance(data, dict) and "answer" in data:
                    extracted = str(data["answer"])
            except (json.JSONDecodeError, TypeError):
                pass
        return self.engine.evaluate_layer("output", {
            "answer": extracted, "raw_answer": answer, "contexts": contexts,
        })

    def run(self, user_input: str, contexts: Optional[List[str]] = None,
            request_id: Optional[str] = None) -> Dict:
        """Full two-layer pipeline: input check -> generate -> output check -> fallback."""
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        contexts = contexts or []

        # Input layer
        input_decision = self.protect_input(user_input)
        final_action = self.engine.decide(input_decision)
        fallback_used = False

        if final_action in (Action.BLOCK, Action.FALLBACK):
            response = self.fallback.response_for(input_decision, user_input)
            fallback_used = True
            entry = AuditEntry(
                timestamp=self.audit.utcnow(), request_id=request_id,
                input_text=user_input, input_decision=input_decision,
                output_text=response, final_action="block" if final_action == Action.BLOCK else "fallback",
                fallback_used=True,
            )
            self.audit.record(entry)
            return {"response": response, "blocked": True,
                    "action": "block" if final_action == Action.BLOCK else "fallback",
                    "request_id": request_id, "input_decision": input_decision,
                    "output_decision": None}

        # Generate
        answer = self.llm(user_input)

        # Output layer
        output_decision = self.protect_output(answer, contexts)
        output_action = self.engine.decide(output_decision)

        if output_action in (Action.BLOCK, Action.FALLBACK):
            response = self.fallback.response_for(output_decision)
            fallback_used = True
            final_action = "fallback"
        elif output_action == Action.SANITIZE:
            response = self.pii.redact_text(answer)
            final_action = "sanitize"
        else:
            response = answer
            final_action = "pass"

        entry = AuditEntry(
            timestamp=self.audit.utcnow(), request_id=request_id,
            input_text=user_input, input_decision=input_decision,
            output_text=response, output_decision=output_decision,
            final_action=final_action, fallback_used=fallback_used,
        )
        self.audit.record(entry)

        return {
            "response": response, "blocked": final_action != "pass",
            "action": final_action, "request_id": request_id,
            "input_decision": input_decision, "output_decision": output_decision,
        }
