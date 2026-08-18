"""Full audit logging for every guardrail decision."""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .severity import GuardrailDecision, Severity


@dataclass
class AuditEntry:
    """One audited pipeline pass (input + output)."""

    timestamp: str
    request_id: str
    input_text: str
    input_decision: GuardrailDecision
    output_text: Optional[str] = None
    output_decision: Optional[GuardrailDecision] = None
    final_action: str = "pass"
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "input_text": self.input_text,
            "input_action": self.input_decision and self.input_decision.max_severity().label.lower(),
            "input_violations": [
                {"rule": r.rule_name, "severity": r.severity.label.lower(),
                 "message": r.message} for r in self.input_decision.violations()
            ],
            "output_text": self.output_text,
            "output_action": self.output_decision and self.output_decision.max_severity().label.lower(),
            "output_violations": [
                {"rule": r.rule_name, "severity": r.severity.label.lower(),
                 "message": r.message} for r in (self.output_decision.violations() if self.output_decision else [])
            ],
            "final_action": self.final_action,
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


class AuditLogger:
    """Writes audit entries to JSONL + CSV sinks."""

    def __init__(self, jsonl_path: Optional[str] = None, csv_path: Optional[str] = None):
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)
        self._entries: List[AuditEntry] = []

    @staticmethod
    def utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, entry: AuditEntry):
        self._entries.append(entry)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        if self.csv_path:
            self._append_csv(entry.to_dict())

    def _append_csv(self, data: Dict):
        exists = False
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                exists = bool(f.read(1))
        except FileNotFoundError:
            exists = False
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(data)

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    def summary(self) -> Dict:
        blocked = sum(1 for e in self._entries if e.final_action in ("block", "fallback"))
        return {
            "total_requests": len(self._entries),
            "blocked_or_fallback": blocked,
            "fallback_used": sum(1 for e in self._entries if e.fallback_used),
            "violations_by_rule": self._violation_counts(),
        }

    def _violation_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            for r in e.input_decision.violations():
                counts[r.rule_name] = counts.get(r.rule_name, 0) + 1
            if e.output_decision:
                for r in e.output_decision.violations():
                    counts[r.rule_name] = counts.get(r.rule_name, 0) + 1
        return counts
