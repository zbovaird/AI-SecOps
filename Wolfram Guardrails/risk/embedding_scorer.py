"""Optional embedding-based risk scorer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from llm.schema import EmbeddingRisk, SemanticParse


class EmbeddingRiskScorer:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model_name: str | None = None,
        examples_path: str | Path | None = None,
        review_threshold: float = 0.62,
        block_threshold: float = 0.82,
    ) -> None:
        settings = _load_settings()
        embedding_cfg = settings.get("embeddings", {})
        self.enabled = (
            enabled
            if enabled is not None
            else _env_bool("EMBEDDINGS_ENABLED", embedding_cfg.get("enabled", False))
        )
        self.model_name = model_name or embedding_cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.examples_path = Path(
            examples_path or embedding_cfg.get("examples_path", "datasets/risk_examples.jsonl")
        )
        self.review_threshold = review_threshold or embedding_cfg.get("review_threshold", 0.62)
        self.block_threshold = block_threshold or embedding_cfg.get("block_threshold", 0.82)
        self._model = None
        self._example_embeddings = None

    def score(self, semantic: SemanticParse) -> EmbeddingRisk:
        if not self.enabled:
            return EmbeddingRisk(score=0.0, reason="Embedding risk scoring disabled.", enabled=False)

        try:
            import numpy as np
        except ImportError:
            return EmbeddingRisk(
                score=0.0,
                reason="Embedding dependencies not installed.",
                enabled=False,
            )

        model = self._load_model()
        examples = self._load_examples()
        if not examples:
            return EmbeddingRisk(score=0.0, reason="No embedding examples loaded.", enabled=True)

        text = " ".join(
            part
            for part in [semantic.intent, semantic.target, semantic.notes, *semantic.techniques]
            if part
        )
        query_embedding = model.encode([text])[0]
        example_embeddings = self._encode_examples(model, examples)
        similarities = [
            float(np.dot(query_embedding, example_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(example_emb)))
            for example_emb in example_embeddings
        ]
        score = max(similarities) if similarities else 0.0
        reason = "Embedding similarity against risk examples."
        return EmbeddingRisk(score=score, reason=reason, enabled=True)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_examples(self) -> list[str]:
        if not self.examples_path.exists():
            return []
        examples: list[str] = []
        with self.examples_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(str(row.get("text", "")))
        return [item for item in examples if item]

    def _encode_examples(self, model, examples: list[str]):
        if self._example_embeddings is None:
            self._example_embeddings = model.encode(examples)
        return self._example_embeddings


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
