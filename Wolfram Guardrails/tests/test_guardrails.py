"""Python policy mirror tests."""

from guardrails.policy import evaluate_guardrails
from llm.schema import Decision, SemanticParse


def test_block_dangerous_intent() -> None:
    semantic = SemanticParse(intent="bypass_policy", notes="ignoring safeguards", risk=0.5)
    decision = evaluate_guardrails(semantic)
    assert decision.decision == Decision.BLOCK
    assert "dangerous_intent" in decision.triggeredRules


def test_review_sensitive_target() -> None:
    semantic = SemanticParse(intent="credential_access", target="credential", risk=0.4)
    decision = evaluate_guardrails(semantic)
    assert decision.decision == Decision.REVIEW
    assert "sensitive_target" in decision.triggeredRules
