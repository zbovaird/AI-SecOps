"""Wolfram Language session bridge."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from guardrails.adapter import semantic_to_json
from llm.schema import SemanticParse, parse_symbolic_dict

DEFAULT_KERNEL_PATHS = [
    "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
    "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel",
]


class WolframUnavailableError(Exception):
    """Raised when Wolfram Engine cannot be used."""


class WolframGuardrailClient:
    def __init__(
        self,
        *,
        kernel_path: str | None = None,
        rules_path: str | Path | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.kernel_path = kernel_path or os.getenv("WOLFRAM_KERNEL_PATH") or self._discover_kernel()
        self.rules_path = Path(rules_path or Path(__file__).resolve().parent / "rules.wl")
        self.timeout_seconds = timeout_seconds
        self._session = None
        self._rules_loaded = False

    @staticmethod
    def _discover_kernel() -> str | None:
        for candidate in DEFAULT_KERNEL_PATHS:
            if Path(candidate).exists():
                return candidate
        return None

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.kernel_path or not Path(self.kernel_path).exists():
            raise WolframUnavailableError("Wolfram kernel path not found")

        try:
            from wolframclient.evaluation import WolframLanguageSession
        except ImportError as exc:
            raise WolframUnavailableError("wolframclient is not installed") from exc

        try:
            self._session = WolframLanguageSession(kernel=self.kernel_path)
            self._session.start()
            self._load_rules()
        except Exception as exc:
            self._session = None
            raise WolframUnavailableError(f"WolframLanguageSession failed: {exc}") from exc

    def _load_rules(self) -> None:
        if self._session is None or self._rules_loaded:
            return
        rules_text = self.rules_path.read_text(encoding="utf-8")
        self._session.evaluate(rules_text, timeout=self.timeout_seconds)
        self._rules_loaded = True

    def _evaluate_via_wolframscript(self, semantic: SemanticParse) -> dict[str, object]:
        rules_text = self.rules_path.read_text(encoding="utf-8")
        payload = semantic_to_json(semantic)
        code = (
            f"{rules_text}\n"
            f'EvaluateGuardrails[ImportString[{json.dumps(payload)}, "RawJSON"]]'
        )
        env = os.environ.copy()
        if self.kernel_path:
            env["WOLFRAMSCRIPT_KERNELPATH"] = self.kernel_path
        try:
            completed = subprocess.run(
                ["wolframscript", "-code", code],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise WolframUnavailableError(f"wolframscript evaluation failed: {exc}") from exc

        output = completed.stdout.strip()
        if not output:
            raise WolframUnavailableError("wolframscript returned empty output")
        return parse_symbolic_dict(json.loads(output)).model_dump()

    def evaluate(self, semantic: SemanticParse) -> dict[str, object]:
        try:
            self._ensure_session()
            assert self._session is not None
            payload = semantic_to_json(semantic)
            expression = f'EvaluateGuardrails[ImportString[{json.dumps(payload)}, "RawJSON"]]'
            raw = self._session.evaluate(expression, timeout=self.timeout_seconds)
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                data = json.loads(str(raw))
            return parse_symbolic_dict(data).model_dump()
        except WolframUnavailableError:
            return self._evaluate_via_wolframscript(semantic)

    def close(self) -> None:
        if self._session is not None:
            self._session.terminate()
            self._session = None
            self._rules_loaded = False

    def __enter__(self) -> "WolframGuardrailClient":
        self._ensure_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
