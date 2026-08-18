"""Configurable rule engine with severity levels and action policies."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .severity import RuleResult, Severity, GuardrailDecision


class Action(Enum):
    """Actions the engine can take on violations."""

    PASS = "pass"          # allow, no action
    WARN = "warn"          # allow but log warning
    SANITIZE = "sanitize"  # redact/modify before proceeding
    FALLBACK = "fallback"  # substitute a fail-safe response
    BLOCK = "block"        # reject the input/output entirely


@dataclass
class RuleSpec:
    """Declarative rule specification."""

    name: str
    layer: str                 # "input" | "output"
    fn: Callable[[Any], RuleResult]
    severity_floor: Severity = Severity.LOW
    action_on_violation: Action = Action.BLOCK
    enabled: bool = True


class RuleEngine:
    """Evaluates rules, applies severity-based actions, returns decisions."""

    def __init__(self, rules: Optional[List[RuleSpec]] = None):
        self.rules: List[RuleSpec] = rules or []
        self.logger = logging.getLogger(__name__)

    def register(self, spec: RuleSpec):
        self.rules.append(spec)
        return self

    def evaluate_layer(self, layer: str, payload: Any) -> GuardrailDecision:
        """Run all enabled rules for a layer against a payload."""
        decision = GuardrailDecision(layer=layer)
        for spec in self.rules:
            if not spec.enabled or spec.layer != layer:
                continue
            try:
                result = spec.fn(payload)
            except Exception as e:
                result = RuleResult(
                    rule_name=spec.name, passed=False,
                    severity=Severity.HIGH,
                    message=f"Rule '{spec.name}' crashed: {e}",
                    details={"exception": str(e)},
                )
            if result.severity.value < spec.severity_floor.value:
                result.severity = spec.severity_floor
            decision.results.append(result)
        return decision

    def decide(self, decision: GuardrailDecision,
               action_map: Optional[Dict[str, Action]] = None) -> Action:
        """Map the worst violation to an action.

        Only *failed* rules influence the action; passing rules never do,
        even when their severity floor raised their reported severity.
        """
        violations = decision.violations()
        if not violations:
            return Action.PASS
        worst = max((r.severity for r in violations), key=lambda s: s.value)
        if worst.value >= Severity.CRITICAL.value:
            return Action.BLOCK
        if worst.value >= Severity.HIGH.value:
            return Action.FALLBACK
        if worst.value >= Severity.MEDIUM.value:
            return Action.SANITIZE
        return Action.WARN

    def summarize(self, decision: GuardrailDecision) -> Dict:
        """Human-readable summary of a decision."""
        return {
            "layer": decision.layer,
            "action": self.decide(decision).value,
            "max_severity": decision.max_severity().label.lower(),
            "violations": [
                {"rule": r.rule_name, "severity": r.severity.label.lower(),
                 "message": r.message, "score": round(r.score, 3)}
                for r in decision.violations()
            ],
            "checks_run": len(decision.results),
        }
