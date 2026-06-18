"""Python mirror of guardrails/rules.wl for fallback and Colab scoring."""

from __future__ import annotations

from llm.schema import Decision, SemanticParse, SymbolicDecision

DANGEROUS_INTENTS = {
    "policy_evasion",
    "bypass_policy",
    "malware_creation",
    "credential_theft",
    "harmful_instruction",
    "exploit_execution",
}

HIGH_RISK_TECHNIQUES = {
    "instruction_override",
    "exfiltration",
    "credential_harvesting",
    "privilege_escalation",
    "malware",
    "stealth",
    "evasion",
}

SENSITIVE_TARGETS = {
    "credential",
    "password",
    "private_key",
    "token",
    "production_system",
    "personal_data",
}


def evaluate_guardrails(semantic: SemanticParse) -> SymbolicDecision:
    triggered: list[str] = []
    decision = Decision.ALLOW
    reason = "No policy violations detected."
    severity = 0.0

    if semantic.intent in DANGEROUS_INTENTS:
        triggered.append("dangerous_intent")
        decision = Decision.BLOCK
        reason = "Dangerous intent detected."
        severity = max(severity, 0.9)

    if "ignoring safeguard" in semantic.notes.lower():
        triggered.append("safeguard_override_notes")
        decision = Decision.BLOCK
        reason = "Notes indicate safeguard override."
        severity = max(severity, 0.9)

    if any(technique in HIGH_RISK_TECHNIQUES for technique in semantic.techniques):
        triggered.append("high_risk_technique")
        decision = Decision.BLOCK
        reason = "High-risk technique detected."
        severity = max(severity, 0.85)

    if semantic.risk >= 0.85:
        triggered.append("high_parser_risk")
        decision = Decision.BLOCK
        reason = "Parser risk score exceeds block threshold."
        severity = max(severity, semantic.risk)

    if decision == Decision.ALLOW and semantic.target in SENSITIVE_TARGETS:
        triggered.append("sensitive_target")
        decision = Decision.REVIEW
        reason = "Sensitive target requires review."
        severity = max(severity, 0.6)

    if decision == Decision.ALLOW and semantic.risk >= 0.5:
        triggered.append("elevated_risk")
        decision = Decision.REVIEW
        reason = "Elevated parser risk requires review."
        severity = max(severity, semantic.risk)

    if decision == Decision.ALLOW and "ambiguous" in semantic.notes.lower():
        triggered.append("ambiguous_notes")
        decision = Decision.REVIEW
        reason = "Ambiguous notes require review."
        severity = max(severity, 0.5)

    return SymbolicDecision(
        decision=decision,
        reason=reason,
        triggeredRules=triggered,
        severity=severity,
    )
