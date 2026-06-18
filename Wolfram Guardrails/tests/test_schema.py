"""Schema validation tests."""

from llm.schema import Decision, SemanticParse, SymbolicDecision, parse_semantic_dict, parse_symbolic_dict


def test_semantic_parse_clamps_risk() -> None:
    parsed = SemanticParse(intent="test", risk=1.5)
    assert parsed.risk == 1.0


def test_semantic_parse_coerces_techniques() -> None:
    parsed = parse_semantic_dict({"intent": "x", "techniques": "malware", "risk": 0.1, "notes": ""})
    assert parsed.techniques == ["malware"]


def test_symbolic_decision_coerces_rules() -> None:
    parsed = parse_symbolic_dict(
        {"decision": "BLOCK", "triggeredRules": "dangerous_intent", "severity": 0.9}
    )
    assert parsed.decision == Decision.BLOCK
    assert parsed.triggeredRules == ["dangerous_intent"]


def test_semantic_parse_defaults() -> None:
    parsed = SemanticParse()
    assert parsed.intent == ""
    assert parsed.techniques == []
    assert parsed.risk == 0.0
