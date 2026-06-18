"""Adversarial dataset shape tests."""

import json
from pathlib import Path

VALID_DECISIONS = {"ALLOW", "BLOCK", "REVIEW", "REWRITE"}


def test_adversarial_dataset_rows() -> None:
    dataset_path = Path("datasets/adversarial_prompts.jsonl")
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 4
    for row in rows:
        assert "id" in row
        assert "prompt" in row
        assert row["expectedDecision"] in VALID_DECISIONS
