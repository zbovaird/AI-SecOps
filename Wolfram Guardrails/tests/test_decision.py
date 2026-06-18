"""Decision precedence tests."""

from llm.schema import Decision, EmbeddingRisk, SymbolicDecision
from pipeline.decision import merge_decisions


def test_parser_error_routes_to_review() -> None:
    decision = merge_decisions(
        symbolic=SymbolicDecision(decision=Decision.ALLOW),
        embedding=None,
        parse_error="invalid json",
    )
    assert decision == Decision.REVIEW


def test_symbolic_block_wins_over_embedding() -> None:
    decision = merge_decisions(
        symbolic=SymbolicDecision(decision=Decision.BLOCK),
        embedding=EmbeddingRisk(score=0.1, enabled=True),
    )
    assert decision == Decision.BLOCK
