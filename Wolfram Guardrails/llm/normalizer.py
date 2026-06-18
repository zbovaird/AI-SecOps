"""Canonicalize semantic parse labels."""

from __future__ import annotations

from llm.schema import SemanticParse


INTENT_ALIASES = {
    "summary": "summarization",
    "summarize": "summarization",
    "summarise": "summarization",
    "bypass": "bypass_policy",
    "jailbreak": "policy_evasion",
    "ignore_safeguards": "bypass_policy",
}


def _canonicalize_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return INTENT_ALIASES.get(normalized, normalized)


def normalize_semantic(parse: SemanticParse) -> SemanticParse:
    intent = _canonicalize_token(parse.intent)
    target = _canonicalize_token(parse.target) if parse.target else ""
    techniques: list[str] = []
    seen: set[str] = set()
    for technique in parse.techniques:
        canonical = _canonicalize_token(technique)
        if canonical and canonical not in seen:
            seen.add(canonical)
            techniques.append(canonical)

    risk = max(0.0, min(1.0, float(parse.risk)))
    notes = parse.notes.strip()

    return SemanticParse(
        intent=intent,
        target=target,
        techniques=techniques,
        risk=risk,
        notes=notes,
    )
