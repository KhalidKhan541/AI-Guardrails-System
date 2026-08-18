"""Output format schema validation."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .severity import RuleResult, Severity


class FormatValidator:
    """Validates model output against a JSON schema or format spec."""

    def __init__(self, schema: Optional[Dict] = None, max_length: int = 8000,
                 allowed_languages: Optional[List[str]] = None):
        self.schema = schema or {}
        self.max_length = max_length
        self.allowed_languages = allowed_languages
        self.logger = logging.getLogger(__name__)

    def validate_json(self, text: str, schema: Optional[Dict] = None) -> Dict:
        """Parse and validate JSON output against a schema."""
        schema = schema or self.schema
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON: {e.msg}",
                    "line": e.lineno, "column": e.colno}
        if not schema:
            return {"valid": True, "data": data}
        errors = self._validate_against_schema(data, schema)
        return {"valid": not errors, "errors": errors, "data": data}

    def _validate_against_schema(self, data: Any, schema: Dict, path: str = "$") -> List[str]:
        errors = []
        expected_type = schema.get("type")
        if expected_type and expected_type != "any":
            if not self._type_matches(data, expected_type):
                errors.append(f"{path}: expected {expected_type}, got {type(data).__name__}")
                return errors
        if expected_type == "object":
            required = schema.get("required", [])
            props = schema.get("properties", {})
            if not isinstance(data, dict):
                errors.append(f"{path}: expected object")
                return errors
            for key in required:
                if key not in data:
                    errors.append(f"{path}.{key}: missing required field")
            for key, sub_schema in props.items():
                if key in data:
                    errors.extend(self._validate_against_schema(data[key], sub_schema, f"{path}.{key}"))
        elif expected_type == "array":
            item_schema = schema.get("items", {})
            for i, item in enumerate(data if isinstance(data, list) else []):
                errors.extend(self._validate_against_schema(item, item_schema, f"{path}[{i}]"))
        return errors

    @staticmethod
    def _type_matches(value: Any, expected: str) -> bool:
        mapping = {
            "string": str, "number": (int, float), "integer": int,
            "boolean": bool, "object": dict, "array": list, "null": type(None),
        }
        target = mapping.get(expected)
        return isinstance(value, target) if target else True

    def check_text(self, text: str) -> Dict:
        """Validate basic text constraints (length, language)."""
        errors = []
        if len(text) > self.max_length:
            errors.append(f"Output exceeds max length ({len(text)} > {self.max_length})")
        if self.allowed_languages:
            detected = self._detect_language(text)
            if detected not in self.allowed_languages:
                errors.append(f"Output language '{detected}' not allowed")
        return {"valid": not errors, "errors": errors}

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
            return "cjk"
        return "latin"

    def check(self, text: str) -> RuleResult:
        """Run format validation rules."""
        errors = []
        # Try JSON if schema is configured
        if self.schema:
            result = self.validate_json(text)
            if not result["valid"]:
                errors.append(result.get("error", "schema violation"))
        text_result = self.check_text(text)
        errors.extend(text_result["errors"])
        if errors:
            return RuleResult(
                rule_name="format_validation",
                passed=False,
                severity=Severity.MEDIUM,
                message="; ".join(errors[:3]),
                details={"errors": errors},
                score=min(len(errors) * 0.3, 1.0),
            )
        return RuleResult(
            rule_name="format_validation",
            passed=True,
            severity=Severity.INFO,
            message="Output format valid.",
            details={},
            score=0.0,
        )
