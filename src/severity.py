"""Severity levels and rule result types for the guardrails engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    """Severity levels for guardrail violations."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self):
        return self.name.lower()

    @property
    def label(self) -> str:
        return self.name

    @classmethod
    def from_name(cls, name: str) -> "Severity":
        try:
            return cls[name.upper()]
        except KeyError:
            return cls.LOW


SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


@dataclass
class RuleResult:
    """Outcome of evaluating one guardrail rule."""

    rule_name: str
    passed: bool
    severity: Severity = Severity.LOW
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def blocked(self) -> bool:
        """Whether this rule result should block the pipeline."""
        return self.severity.value >= Severity.HIGH.value


@dataclass
class GuardrailDecision:
    """Aggregate decision for one layer (input or output)."""

    layer: str
    results: List[RuleResult] = field(default_factory=list)
    fallback_response: Optional[str] = None

    def max_severity(self) -> Severity:
        if not self.results:
            return Severity.INFO
        return max((r.severity for r in self.results), key=lambda s: s.value)

    def blocked(self) -> bool:
        return self.max_severity().value >= Severity.HIGH.value

    def violations(self) -> List[RuleResult]:
        return [r for r in self.results if not r.passed]
