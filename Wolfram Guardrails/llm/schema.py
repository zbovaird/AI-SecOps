"""Semantic parse and guardrail result schema models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Decision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    REWRITE = "REWRITE"


class SemanticParse(BaseModel):
    intent: str = ""
    target: str = ""
    techniques: list[str] = Field(default_factory=list)
    risk: float = 0.0
    notes: str = ""

    @field_validator("risk")
    @classmethod
    def clamp_risk(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("techniques", mode="before")
    @classmethod
    def coerce_techniques(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]


class SymbolicDecision(BaseModel):
    decision: Decision
    reason: str = ""
    triggeredRules: list[str] = Field(default_factory=list)
    severity: float = 0.0

    @field_validator("decision", mode="before")
    @classmethod
    def coerce_decision(cls, value: Any) -> Decision:
        if isinstance(value, Decision):
            return value
        normalized = str(value).strip().upper()
        return Decision(normalized)

    @field_validator("triggeredRules", mode="before")
    @classmethod
    def coerce_rules(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    @field_validator("severity")
    @classmethod
    def clamp_severity(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class EmbeddingRisk(BaseModel):
    score: float = 0.0
    reason: str = "Embedding risk scoring disabled."
    enabled: bool = False


class GuardrailResult(BaseModel):
    decision: Decision
    semantic: SemanticParse | None = None
    symbolic: SymbolicDecision | None = None
    embedding: EmbeddingRisk | None = None
    parse_error: str | None = None
    prompt: str | None = None

    def to_report_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"decision": self.decision.value}
        if self.semantic is not None:
            payload["semantic"] = self.semantic.model_dump()
        if self.symbolic is not None:
            payload["symbolic"] = self.symbolic.model_dump()
        if self.embedding is not None and self.embedding.enabled:
            payload["embedding"] = self.embedding.model_dump()
        if self.parse_error:
            payload["parse_error"] = self.parse_error
        if include_prompt and self.prompt is not None:
            payload["prompt"] = self.prompt
        return payload


def parse_semantic_dict(data: dict[str, Any]) -> SemanticParse:
    return SemanticParse.model_validate(data)


def parse_symbolic_dict(data: dict[str, Any]) -> SymbolicDecision:
    return SymbolicDecision.model_validate(data)
