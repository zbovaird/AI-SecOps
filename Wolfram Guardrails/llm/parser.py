"""Semantic parsers backed by Ollama or mocks."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import error, request

import httpx

from llm.normalizer import normalize_semantic
from llm.repair import SEMANTIC_SCHEMA_HINT, build_repair_prompt, extract_json_object
from llm.schema import SemanticParse, parse_semantic_dict


class ParserError(Exception):
    """Raised when semantic parsing fails."""


class SemanticParser(ABC):
    @abstractmethod
    def parse(self, prompt: str) -> SemanticParse:
        raise NotImplementedError


class OllamaSemanticParser(SemanticParser):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        parser_prompt_path: str | Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.parser_prompt = self._load_parser_prompt(parser_prompt_path)

    def _load_parser_prompt(self, parser_prompt_path: str | Path | None) -> str:
        if parser_prompt_path is not None:
            return Path(parser_prompt_path).read_text(encoding="utf-8")
        default_path = Path(__file__).resolve().parent.parent / "config" / "parser_prompt.txt"
        return default_path.read_text(encoding="utf-8")

    def validate_raw_output(self, raw_output: str) -> SemanticParse:
        data = extract_json_object(raw_output)
        if data is None:
            raise ParserError("Could not extract semantic JSON from model output")
        parsed = parse_semantic_dict(data)
        return normalize_semantic(parsed)

    def parse(self, prompt: str) -> SemanticParse:
        raw = self._call_ollama(prompt)
        try:
            return self.validate_raw_output(raw)
        except ParserError:
            repair_prompt = build_repair_prompt(raw, SEMANTIC_SCHEMA_HINT)
            repaired = self._call_ollama(repair_prompt)
            return self.validate_raw_output(repaired)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.parser_prompt},
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
            data = self._urllib_chat(payload)
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise ParserError("Ollama returned empty response")
        return str(content)

    def _urllib_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise ParserError(f"Ollama request failed: {exc}") from exc


class MockSemanticParser(SemanticParser):
    """Deterministic parser for tests without live Ollama."""

    def __init__(self, responses: dict[str, SemanticParse] | None = None) -> None:
        self.responses = responses or {}

    def parse(self, prompt: str) -> SemanticParse:
        lowered = prompt.lower()
        if prompt in self.responses:
            return normalize_semantic(self.responses[prompt])

        if any(token in lowered for token in ("ignore safeguard", "bypass policy", "jailbreak", "developer mode", "override safety", "disregard")):
            return SemanticParse(
                intent="bypass_policy",
                target="",
                techniques=["instruction_override"] if "developer mode" in lowered else [],
                risk=0.5,
                notes="ignoring safeguards",
            )
        if "malware" in lowered or "trojan" in lowered:
            return SemanticParse(
                intent="malware_creation",
                target="",
                techniques=["malware"],
                risk=0.9,
                notes="malware creation request",
            )
        if "exploit" in lowered and "cve" in lowered:
            return SemanticParse(
                intent="exploit_execution",
                target="production_system",
                techniques=["exploit"],
                risk=0.9,
                notes="operational exploit request",
            )
        if "steal" in lowered and "password" in lowered:
            return SemanticParse(
                intent="credential_theft",
                target="credential",
                techniques=["credential_harvesting"],
                risk=0.9,
                notes="credential theft request",
            )
        if "credential" in lowered or "password" in lowered:
            return SemanticParse(
                intent="credential_access",
                target="credential",
                techniques=[],
                risk=0.6,
                notes="sensitive credential context",
            )
        if "social security" in lowered or "employee" in lowered and "database" in lowered:
            return SemanticParse(
                intent="data_access",
                target="personal_data",
                techniques=[],
                risk=0.7,
                notes="sensitive personal data request",
            )
        if "mfa" in lowered or "attacker" in lowered:
            return SemanticParse(
                intent="security_education",
                target="",
                techniques=[],
                risk=0.55,
                notes="ambiguous dual-use security discussion",
            )
        if "system:" in lowered or "hidden system prompt" in lowered:
            return SemanticParse(
                intent="policy_evasion",
                target="",
                techniques=["instruction_override"],
                risk=0.85,
                notes="prompt injection pattern",
            )
        if "summarize" in lowered or "summary" in lowered:
            return SemanticParse(
                intent="summarization",
                target="",
                techniques=[],
                risk=0.1,
                notes="benign request",
            )
        if "sql injection" in lowered or "security concept" in lowered:
            return SemanticParse(
                intent="security_education",
                target="",
                techniques=[],
                risk=0.2,
                notes="high-level educational discussion",
            )
        return SemanticParse(
            intent="general_request",
            target="",
            techniques=[],
            risk=0.3,
            notes="",
        )
