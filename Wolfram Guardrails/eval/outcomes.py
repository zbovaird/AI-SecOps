"""Outcome tracking helpers for guardrail comparison rows."""

from __future__ import annotations

from typing import Any

from eval.disagreement_analyzer import classify_failure
from eval.policy_trace import attach_policy_trace


def build_comparison_row(
    *,
    item: dict[str, Any],
    english_decision,
    wolfram_decision,
    english_ok: bool,
    wolfram_ok: bool,
    english_latency_ms: float,
    wolfram_latency_ms: float,
    semantic,
) -> dict[str, Any]:
    """Build a fully tracked comparison row for one dataset item."""
    row: dict[str, Any] = {
        "id": item["id"],
        "prompt": item["prompt"],
        "category": item.get("category", "unknown"),
        "expectedDecision": item["expectedDecision"],
        "englishDecision": english_decision.decision.value,
        "wolframDecision": wolfram_decision.decision.value,
        "reasons": {
            "english": english_decision.reason,
            "wolfram": wolfram_decision.reason,
        },
        "triggeredRules": {
            "english": english_decision.triggeredRules,
            "wolfram": wolfram_decision.triggeredRules,
        },
        "englishReason": english_decision.reason,
        "wolframReason": wolfram_decision.reason,
        "englishTriggeredRules": english_decision.triggeredRules,
        "wolframTriggeredRules": wolfram_decision.triggeredRules,
        "englishParseOk": english_ok,
        "wolframParseOk": wolfram_ok,
        "englishLatencyMs": english_latency_ms,
        "wolframLatencyMs": wolfram_latency_ms,
        "semanticParse": semantic.model_dump() if semantic else None,
    }
    return attach_policy_trace(enrich_outcome_fields(row))


def enrich_outcome_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Add correctness flags and normalized nested fields to a comparison row."""
    if "reasons" not in row:
        row["reasons"] = {
            "english": row.get("englishReason", ""),
            "wolfram": row.get("wolframReason", ""),
        }
    if "triggeredRules" not in row:
        row["triggeredRules"] = {
            "english": row.get("englishTriggeredRules", []),
            "wolfram": row.get("wolframTriggeredRules", []),
        }

    row["englishCorrect"] = row["englishDecision"] == row["expectedDecision"]
    row["wolframCorrect"] = row["wolframDecision"] == row["expectedDecision"]
    row["evaluatorsAgree"] = row["englishDecision"] == row["wolframDecision"]
    row["failureType"] = classify_failure(row)
    if "policyTrace" not in row:
        attach_policy_trace(row)
    return row
