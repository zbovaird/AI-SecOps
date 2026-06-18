"""End-to-end guardrail orchestration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from guardrails.engine import GuardrailEngine
from llm.parser import MockSemanticParser, OllamaSemanticParser, ParserError, SemanticParser
from llm.schema import Decision, GuardrailResult, SymbolicDecision
from pipeline.decision import build_result
from risk.embedding_scorer import EmbeddingRiskScorer
from utils.logging import get_logger

logger = get_logger(__name__)


class GuardrailOrchestrator:
    def __init__(
        self,
        *,
        parser: SemanticParser | None = None,
        engine: GuardrailEngine | None = None,
        embedding_scorer: EmbeddingRiskScorer | None = None,
        use_mock_parser: bool = False,
    ) -> None:
        settings = _load_settings()
        ollama_cfg = settings.get("ollama", {})
        self.parser = parser or (
            MockSemanticParser() if use_mock_parser else OllamaSemanticParser(
                base_url=os.getenv("OLLAMA_BASE_URL", ollama_cfg.get("base_url")),
                model=os.getenv("OLLAMA_MODEL", ollama_cfg.get("model")),
                temperature=float(ollama_cfg.get("temperature", 0)),
                timeout_seconds=float(ollama_cfg.get("timeout_seconds", 30)),
            )
        )
        self.engine = engine or GuardrailEngine()
        self.embedding_scorer = embedding_scorer or EmbeddingRiskScorer()

    def evaluate(self, prompt: str) -> GuardrailResult:
        parse_error = None
        semantic = None
        symbolic = None

        try:
            semantic = self.parser.parse(prompt)
            symbolic = self.engine.evaluate(semantic)
        except ParserError as exc:
            logger.info("Parser error routed to REVIEW: %s", exc)
            parse_error = str(exc)
            symbolic = SymbolicDecision(
                decision=Decision.REVIEW,
                reason="Parser failure",
                triggeredRules=["parser_failure"],
                severity=0.5,
            )
        finally:
            self.engine.close()

        embedding = self.embedding_scorer.score(semantic) if semantic else None
        return build_result(
            prompt=prompt,
            semantic=semantic,
            symbolic=symbolic,
            embedding=embedding,
            parse_error=parse_error,
        )


def _load_settings() -> dict:
    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if not settings_path.exists():
        return {}
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
