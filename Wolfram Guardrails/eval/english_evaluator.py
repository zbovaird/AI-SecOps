"""English natural-language policy evaluator."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from llm.repair import POLICY_SCHEMA_HINT, build_repair_prompt, extract_json_object
from llm.schema import Decision, SymbolicDecision, parse_symbolic_dict


class EnglishEvaluatorError(Exception):
    """Raised when English guardrail evaluation fails."""


class EnglishGuardrailEvaluator:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        policy_prompt_path: str | Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.policy_prompt = self._load_policy_prompt(policy_prompt_path)

    def _load_policy_prompt(self, policy_prompt_path: str | Path | None) -> str:
        if policy_prompt_path is not None:
            return Path(policy_prompt_path).read_text(encoding="utf-8")
        default_path = Path(__file__).resolve().parent.parent / "config" / "english_policy_prompt.txt"
        return default_path.read_text(encoding="utf-8")

    def evaluate(self, prompt: str) -> tuple[SymbolicDecision, bool]:
        raw = self._call_ollama(prompt)
        parsed, ok = self._parse_policy_output(raw)
        if not ok:
            repair = build_repair_prompt(raw, POLICY_SCHEMA_HINT)
            raw = self._call_ollama(repair)
            parsed, ok = self._parse_policy_output(raw)
        if parsed is None:
            raise EnglishEvaluatorError("Could not parse English policy JSON")
        return parsed, ok

    def _parse_policy_output(self, raw: str) -> tuple[SymbolicDecision | None, bool]:
        data = extract_json_object(raw)
        if data is None:
            return None, False
        try:
            return parse_symbolic_dict(data), True
        except Exception:
            return None, False

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.policy_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self._client is not None:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        content = (data.get("message") or {}).get("content")
        if not content:
            raise EnglishEvaluatorError("Empty Ollama response")
        return str(content)


class MockEnglishGuardrailEvaluator:
    """Deterministic English evaluator for offline comparison runs."""

    def evaluate(self, prompt: str) -> tuple[SymbolicDecision, bool]:
        lowered = prompt.lower()
        if any(token in lowered for token in ("ignore safeguard", "bypass", "jailbreak", "developer mode", "override safety")):
            return SymbolicDecision(
                decision=Decision.BLOCK,
                reason="Policy evasion detected",
                triggeredRules=["english_policy_block"],
                severity=0.9,
            ), True
        if "password" in lowered or "social security" in lowered:
            return SymbolicDecision(
                decision=Decision.REVIEW,
                reason="Sensitive target",
                triggeredRules=["english_policy_review"],
                severity=0.6,
            ), True
        if "summarize" in lowered or "explain" in lowered and "sql injection" in lowered:
            return SymbolicDecision(
                decision=Decision.ALLOW,
                reason="Benign request",
                triggeredRules=[],
                severity=0.1,
            ), True
        if "malware" in lowered or "exploit" in lowered or "steal" in lowered:
            return SymbolicDecision(
                decision=Decision.BLOCK,
                reason="Harmful request",
                triggeredRules=["english_policy_block"],
                severity=0.85,
            ), True
        return SymbolicDecision(
            decision=Decision.REVIEW,
            reason="Ambiguous request",
            triggeredRules=["english_policy_review"],
            severity=0.5,
        ), True
