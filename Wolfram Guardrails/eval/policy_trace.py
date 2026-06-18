"""Build per-prompt policy translation traces for comparison reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guardrails.policy import (
    DANGEROUS_INTENTS,
    HIGH_RISK_TECHNIQUES,
    SENSITIVE_TARGETS,
)
from llm.schema import SemanticParse, parse_semantic_dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGLISH_POLICY_FILE = "config/english_policy_prompt.txt"
PARSER_PROMPT_FILE = "config/parser_prompt.txt"
WOLFRAM_RULES_FILE = "guardrails/rules.wl"

CONCEPT_MAPPING = [
    {
        "englishPolicyConcept": "BLOCK policy evasion / jailbreaks",
        "semanticJsonField": "intent in {policy_evasion, bypass_policy, ...}",
        "wolframRule": "DangerousIntentQ → dangerous_intent",
    },
    {
        "englishPolicyConcept": "BLOCK credential theft / malware / exploits",
        "semanticJsonField": "intent or techniques[]",
        "wolframRule": "DangerousIntentQ / HighRiskTechniqueQ",
    },
    {
        "englishPolicyConcept": "BLOCK ignoring safeguards",
        "semanticJsonField": "notes contains 'ignoring safeguard'",
        "wolframRule": "safeguard_override_notes",
    },
    {
        "englishPolicyConcept": "REVIEW sensitive targets / credentials / PII",
        "semanticJsonField": "target in {credential, password, personal_data, ...}",
        "wolframRule": "SensitiveTargetQ → sensitive_target",
    },
    {
        "englishPolicyConcept": "REVIEW ambiguous or elevated-risk requests",
        "semanticJsonField": "risk >= 0.5 or ambiguous notes",
        "wolframRule": "elevated_risk / ambiguous_notes",
    },
    {
        "englishPolicyConcept": "BLOCK extremely high parser-assessed risk",
        "semanticJsonField": "risk >= 0.85",
        "wolframRule": "high_parser_risk",
    },
]


def build_policy_trace(row: dict[str, Any]) -> dict[str, Any]:
    """Document English policy path vs Wolfram semantic→rules path for one comparison row."""
    semantic_raw = row.get("semanticParse")
    semantic = parse_semantic_dict(semantic_raw) if semantic_raw else None

    return {
        "englishPath": {
            "policyPromptFile": ENGLISH_POLICY_FILE,
            "description": "LLM reads raw user prompt + English policy prose and returns decision JSON.",
            "input": row.get("prompt", ""),
            "output": {
                "decision": row.get("englishDecision"),
                "reason": row.get("englishReason") or row.get("reasons", {}).get("english", ""),
                "triggeredRules": row.get("englishTriggeredRules")
                or row.get("triggeredRules", {}).get("english", []),
            },
        },
        "wolframPath": {
            "parserPromptFile": PARSER_PROMPT_FILE,
            "rulesFile": WOLFRAM_RULES_FILE,
            "description": "LLM parses prompt to semantic JSON; Wolfram symbolic rules evaluate JSON (not English policy prose).",
            "semanticJson": semantic_raw,
            "ruleEvaluation": explain_wolfram_rule_hits(semantic) if semantic else [],
            "output": {
                "decision": row.get("wolframDecision"),
                "reason": row.get("wolframReason") or row.get("reasons", {}).get("wolfram", ""),
                "triggeredRules": row.get("wolframTriggeredRules")
                or row.get("triggeredRules", {}).get("wolfram", []),
            },
        },
        "conceptMapping": CONCEPT_MAPPING,
    }


def explain_wolfram_rule_hits(semantic: SemanticParse) -> list[dict[str, str]]:
    """Explain which Wolfram rules would match a semantic parse (mirrors rules.wl / policy.py)."""
    hits: list[dict[str, str]] = []

    if semantic.intent in DANGEROUS_INTENTS:
        hits.append(
            {
                "rule": "dangerous_intent",
                "predicate": "DangerousIntentQ",
                "matchedBecause": f'intent="{semantic.intent}" is in $DangerousIntents',
            }
        )
    if "ignoring safeguard" in semantic.notes.lower():
        hits.append(
            {
                "rule": "safeguard_override_notes",
                "predicate": "ContainsIgnoringCaseQ[notes, \"ignoring safeguard\"]",
                "matchedBecause": "notes indicate safeguard override",
            }
        )
    for technique in semantic.techniques:
        if technique in HIGH_RISK_TECHNIQUES:
            hits.append(
                {
                    "rule": "high_risk_technique",
                    "predicate": "HighRiskTechniqueQ",
                    "matchedBecause": f'technique="{technique}" is in $HighRiskTechniques',
                }
            )
            break
    if semantic.risk >= 0.85:
        hits.append(
            {
                "rule": "high_parser_risk",
                "predicate": "risk >= 0.85",
                "matchedBecause": f"parser risk={semantic.risk}",
            }
        )
    if semantic.target in SENSITIVE_TARGETS:
        hits.append(
            {
                "rule": "sensitive_target",
                "predicate": "SensitiveTargetQ",
                "matchedBecause": f'target="{semantic.target}" is in $SensitiveTargets',
            }
        )
    if semantic.risk >= 0.5:
        hits.append(
            {
                "rule": "elevated_risk",
                "predicate": "risk >= 0.5",
                "matchedBecause": f"parser risk={semantic.risk}",
            }
        )
    if "ambiguous" in semantic.notes.lower():
        hits.append(
            {
                "rule": "ambiguous_notes",
                "predicate": "ContainsIgnoringCaseQ[notes, \"ambiguous\"]",
                "matchedBecause": "notes marked ambiguous",
            }
        )

    if not hits:
        hits.append(
            {
                "rule": "none",
                "predicate": "EvaluateGuardrails default",
                "matchedBecause": "no block/review predicates matched → ALLOW",
            }
        )
    return hits


def attach_policy_trace(row: dict[str, Any]) -> dict[str, Any]:
    row["policyTrace"] = build_policy_trace(row)
    return row
