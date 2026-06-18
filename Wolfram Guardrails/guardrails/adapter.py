"""Convert SemanticParse to JSON for Wolfram evaluation."""

from __future__ import annotations

import json

from llm.schema import SemanticParse


def semantic_to_json(semantic: SemanticParse) -> str:
    return json.dumps(semantic.model_dump(), ensure_ascii=True)


def semantic_to_dict(semantic: SemanticParse) -> dict[str, object]:
    return semantic.model_dump()
