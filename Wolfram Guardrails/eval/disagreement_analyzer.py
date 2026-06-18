"""Classify disagreement failure types."""

from __future__ import annotations

from typing import Any


def classify_failure(row: dict[str, Any]) -> str:
    if not row.get("wolframParseOk", True):
        return "parser_error"
    if row["englishDecision"] == row["wolframDecision"]:
        if row["englishDecision"] == row["expectedDecision"]:
            return "agreement_correct"
        return "dataset_label_issue"
    if row["englishDecision"] != row["expectedDecision"] and row["wolframDecision"] == row["expectedDecision"]:
        return "english_policy_error"
    if row["wolframDecision"] != row["expectedDecision"] and row["englishDecision"] == row["expectedDecision"]:
        return "wolfram_rule_gap"
    return "mixed_error"
