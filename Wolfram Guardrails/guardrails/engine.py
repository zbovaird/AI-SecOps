"""Guardrail engine with Wolfram primary and Python fallback."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from guardrails.client import WolframGuardrailClient, WolframUnavailableError
from guardrails.policy import evaluate_guardrails
from llm.schema import SemanticParse, SymbolicDecision, parse_symbolic_dict


class GuardrailEngine:
    def __init__(
        self,
        *,
        kernel_path: str | None = None,
        rules_path: str | Path | None = None,
        require_kernel: bool | None = None,
        timeout_seconds: float = 10.0,
        client: WolframGuardrailClient | None = None,
    ) -> None:
        settings = _load_settings()
        wolfram_cfg = settings.get("wolfram", {})
        self.require_kernel = (
            require_kernel
            if require_kernel is not None
            else _env_bool("WOLFRAM_REQUIRE_KERNEL", wolfram_cfg.get("require_kernel", False))
        )
        self.client = client or WolframGuardrailClient(
            kernel_path=kernel_path or os.getenv("WOLFRAM_KERNEL_PATH") or wolfram_cfg.get("kernel_path"),
            rules_path=rules_path or wolfram_cfg.get("rules_path"),
            timeout_seconds=timeout_seconds,
        )

    def evaluate(self, semantic: SemanticParse) -> SymbolicDecision:
        try:
            result = self.client.evaluate(semantic)
            return parse_symbolic_dict(result)
        except WolframUnavailableError:
            if self.require_kernel:
                raise
            return evaluate_guardrails(semantic)

    def close(self) -> None:
        self.client.close()


def _load_settings() -> dict:
    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if not settings_path.exists():
        return {}
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
