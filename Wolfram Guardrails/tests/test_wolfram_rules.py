"""Wolfram rules parity tests using Python mirror."""

from guardrails.policy import evaluate_guardrails
from llm.schema import Decision, SemanticParse


def test_wolfram_mirror_blocks_high_risk_technique() -> None:
    semantic = SemanticParse(
        intent="general_request",
        techniques=["instruction_override"],
        risk=0.2,
    )
    decision = evaluate_guardrails(semantic)
    assert decision.decision == Decision.BLOCK
    assert "high_risk_technique" in decision.triggeredRules
