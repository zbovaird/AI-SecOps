"""Final decision precedence logic."""

from __future__ import annotations

from llm.schema import Decision, EmbeddingRisk, GuardrailResult, SemanticParse, SymbolicDecision


DECISION_RANK = {
    Decision.ALLOW: 0,
    Decision.REWRITE: 1,
    Decision.REVIEW: 2,
    Decision.BLOCK: 3,
}


def merge_decisions(
    *,
    symbolic: SymbolicDecision | None,
    embedding: EmbeddingRisk | None,
    parse_error: str | None = None,
) -> Decision:
    if parse_error:
        return Decision.REVIEW

    decision = symbolic.decision if symbolic else Decision.REVIEW

    if embedding and embedding.enabled:
        if embedding.score >= 0.82:
            decision = _max_decision(decision, Decision.BLOCK)
        elif embedding.score >= 0.62:
            decision = _max_decision(decision, Decision.REVIEW)

    return decision


def build_result(
    *,
    prompt: str,
    semantic: SemanticParse | None,
    symbolic: SymbolicDecision | None,
    embedding: EmbeddingRisk | None,
    parse_error: str | None = None,
) -> GuardrailResult:
    decision = merge_decisions(symbolic=symbolic, embedding=embedding, parse_error=parse_error)
    return GuardrailResult(
        decision=decision,
        semantic=semantic,
        symbolic=symbolic,
        embedding=embedding,
        parse_error=parse_error,
        prompt=prompt,
    )


def _max_decision(current: Decision, candidate: Decision) -> Decision:
    return candidate if DECISION_RANK[candidate] > DECISION_RANK[current] else current
