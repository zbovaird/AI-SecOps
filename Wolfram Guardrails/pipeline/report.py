"""Human-readable and JSON reporting."""

from __future__ import annotations

import json

from llm.schema import GuardrailResult
from utils.logging import should_store_prompt_text


def format_json(result: GuardrailResult) -> str:
    return json.dumps(
        result.to_report_dict(include_prompt=should_store_prompt_text()),
        indent=2,
        ensure_ascii=True,
    )


def format_human(result: GuardrailResult) -> str:
    lines = [f"Decision: {result.decision.value}"]
    if result.symbolic is not None:
        rules = ", ".join(result.symbolic.triggeredRules) or "none"
        lines.append(f"Triggered rules: {rules}")
        if result.symbolic.reason:
            lines.append(f"Reason: {result.symbolic.reason}")
    if result.parse_error:
        lines.append(f"Parse error: {result.parse_error}")
    return "\n".join(lines)
