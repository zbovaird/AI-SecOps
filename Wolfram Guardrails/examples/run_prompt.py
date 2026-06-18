#!/usr/bin/env python3
"""Run a single prompt through the guardrail pipeline."""

from __future__ import annotations

import argparse

from pipeline.orchestrator import GuardrailOrchestrator
from pipeline.report import format_human, format_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    orchestrator = GuardrailOrchestrator(use_mock_parser=args.mock)
    result = orchestrator.evaluate(args.prompt)
    print(format_json(result) if args.json else format_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
