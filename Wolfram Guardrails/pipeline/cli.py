"""Command-line interface for guardrail evaluation."""

from __future__ import annotations

import argparse
import sys

from pipeline.orchestrator import GuardrailOrchestrator
from pipeline.report import format_human, format_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a prompt with Wolfram guardrails.")
    parser.add_argument("prompt", help="User prompt to evaluate")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--mock", action="store_true", help="Use mock semantic parser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    orchestrator = GuardrailOrchestrator(use_mock_parser=args.mock)
    result = orchestrator.evaluate(args.prompt)
    if args.json:
        print(format_json(result))
    else:
        print(format_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
