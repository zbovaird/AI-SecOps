"""Validate Python policy mirror against real Wolfram rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guardrails.engine import GuardrailEngine
from guardrails.policy import evaluate_guardrails
from llm.schema import SemanticParse, parse_semantic_dict


def validate_mirror_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare policy.py mirror vs real Wolfram rules for rows with semantic parses."""
    engine = GuardrailEngine(require_kernel=False)
    parity_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            semantic_raw = row.get("semanticParse")
            if not semantic_raw:
                continue
            semantic = parse_semantic_dict(semantic_raw)
            mirror = evaluate_guardrails(semantic)
            real = engine.evaluate(semantic)
            agrees = mirror.decision == real.decision
            parity_rows.append(
                {
                    "id": row["id"],
                    "mirrorDecision": mirror.decision.value,
                    "realDecision": real.decision.value,
                    "agrees": agrees,
                    "mirrorRules": mirror.triggeredRules,
                    "realRules": real.triggeredRules,
                }
            )
    finally:
        engine.close()

    checked = len(parity_rows)
    agreements = sum(1 for item in parity_rows if item["agrees"])
    return {
        "checked": checked,
        "agreementRate": agreements / checked if checked else 0.0,
        "mismatches": [item for item in parity_rows if not item["agrees"]],
        "details": parity_rows,
    }


def write_parity_report(output_dir: Path, parity: dict[str, Any]) -> Path:
    path = output_dir / "mirror_parity.json"
    path.write_text(json.dumps(parity, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
