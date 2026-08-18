"""PII (personally identifiable information) scanner using regex patterns."""

import logging
import re
from typing import Dict, List, Optional

from .severity import RuleResult, Severity

PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_us": r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date_of_birth": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "zipcode": r"\b\d{5}(?:-\d{4})?\b",
}


class PIIScanner:
    """Scans text for PII using regex patterns with severity mapping."""

    SEVERITY_BY_TYPE = {
        "ssn": Severity.CRITICAL,
        "credit_card": Severity.CRITICAL,
        "email": Severity.HIGH,
        "phone_us": Severity.HIGH,
        "ip_address": Severity.MEDIUM,
        "date_of_birth": Severity.MEDIUM,
        "zipcode": Severity.LOW,
    }

    def __init__(self, patterns: Optional[Dict[str, str]] = None, redact: bool = True):
        self.patterns = patterns or PII_PATTERNS
        self.redact = redact
        self.logger = logging.getLogger(__name__)

    def scan(self, text: str) -> Dict:
        """Find PII matches grouped by type."""
        findings = []
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": pii_type,
                    "value": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                })
        return {"findings": findings, "count": len(findings),
                "types": sorted({f["type"] for f in findings})}

    def redact_text(self, text: str) -> str:
        """Replace PII with [REDACTED:<type>] markers."""
        result = text
        for match in re.finditer("|".join(f"({p})" for p in self.patterns.values()), text):
            result = result[:match.start()] + "[REDACTED]" + result[match.end():]
        return result

    def check(self, text: str) -> RuleResult:
        """Run the PII scanning rule."""
        result = self.scan(text)
        if result["count"] > 0:
            worst = max((self.SEVERITY_BY_TYPE.get(f["type"], Severity.LOW)
                         for f in result["findings"]), key=lambda s: s.value)
            return RuleResult(
                rule_name="pii_scanner",
                passed=False,
                severity=worst,
                message=f"PII detected: {', '.join(result['types'])} "
                        f"({result['count']} finding(s)).",
                details=result,
                score=min(result["count"] * 0.3, 1.0),
            )
        return RuleResult(
            rule_name="pii_scanner",
            passed=True,
            severity=Severity.INFO,
            message="No PII detected.",
            details=result,
            score=0.0,
        )
