"""Wolfram guardrail evaluator wrapper."""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from guardrails.engine import GuardrailEngine
from guardrails.policy import evaluate_guardrails
from llm.parser import MockSemanticParser, OllamaSemanticParser, SemanticParser
from llm.schema import Decision, SemanticParse, SymbolicDecision


class WolframGuardrailEvaluator:
    def __init__(
        self,
        *,
        parser: SemanticParser | None = None,
        engine: GuardrailEngine | None = None,
        use_mock: bool = False,
        use_mirror_only: bool = False,
    ) -> None:
        settings = _load_settings()
        ollama_cfg = settings.get("ollama", {})
        if parser is not None:
            self.parser = parser
        elif use_mock:
            self.parser = MockSemanticParser()
        else:
            self.parser = OllamaSemanticParser(
                base_url=os.getenv("OLLAMA_BASE_URL", ollama_cfg.get("base_url")),
                model=os.getenv("OLLAMA_MODEL", ollama_cfg.get("model")),
                temperature=float(ollama_cfg.get("temperature", 0)),
                timeout_seconds=float(ollama_cfg.get("timeout_seconds", 120)),
            )
        self.engine = engine or GuardrailEngine(require_kernel=False)
        self.use_mirror_only = use_mirror_only

    def evaluate(self, prompt: str) -> tuple[SemanticParse | None, SymbolicDecision, bool, float]:
        start = time.perf_counter()
        parse_ok = True
        semantic: SemanticParse | None = None
        try:
            semantic = self.parser.parse(prompt)
        except Exception:
            parse_ok = False
            elapsed = time.perf_counter() - start
            return None, SymbolicDecision(
                decision=Decision.REVIEW,
                reason="Parser failure",
                triggeredRules=["parser_failure"],
                severity=0.5,
            ), parse_ok, elapsed

        if self.use_mirror_only:
            symbolic = evaluate_guardrails(semantic)
        else:
            symbolic = self.engine.evaluate(semantic)
            self.engine.close()

        elapsed = time.perf_counter() - start
        return semantic, symbolic, parse_ok, elapsed


def _load_settings() -> dict:
    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if not settings_path.exists():
        return {}
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
