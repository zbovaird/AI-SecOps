#!/usr/bin/env python3
"""Batch-evaluate prompts from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.orchestrator import GuardrailOrchestrator
from pipeline.report import format_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/adversarial_prompts.jsonl")
    parser.add_argument("--output", default="results/batch_results.jsonl")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orchestrator = GuardrailOrchestrator(use_mock_parser=args.mock)
    with dataset_path.open(encoding="utf-8") as handle, output_path.open("w", encoding="utf-8") as out:
        for line in handle:
            row = json.loads(line)
            result = orchestrator.evaluate(row["prompt"])
            payload = {
                "id": row["id"],
                "expectedDecision": row["expectedDecision"],
                "result": json.loads(format_json(result)),
            }
            out.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
