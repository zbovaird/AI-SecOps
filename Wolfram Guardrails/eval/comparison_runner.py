"""Run English vs Wolfram guardrail comparison."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from eval.english_evaluator import (
    EnglishEvaluatorError,
    EnglishGuardrailEvaluator,
    MockEnglishGuardrailEvaluator,
)
from eval.metrics import compute_metrics
from eval.outcomes import build_comparison_row
from eval.parity_validator import validate_mirror_parity, write_parity_report
from eval.report_generator import generate_report
from eval.wolfram_evaluator import WolframGuardrailEvaluator
from llm.schema import Decision, SymbolicDecision
from utils.logging import get_logger

logger = get_logger(__name__)


def _build_evaluators(*, use_mock: bool, use_mirror_only: bool):
    if use_mock:
        return (
            MockEnglishGuardrailEvaluator(),
            WolframGuardrailEvaluator(use_mock=True, use_mirror_only=use_mirror_only),
        )
    return (
        EnglishGuardrailEvaluator(timeout_seconds=120.0),
        WolframGuardrailEvaluator(use_mock=False, use_mirror_only=use_mirror_only),
    )


def run_comparison(
    dataset_path: Path,
    output_dir: Path,
    *,
    use_mock: bool = True,
    use_mirror_only: bool = False,
    validate_parity: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    english_evaluator, wolfram_evaluator = _build_evaluators(
        use_mock=use_mock,
        use_mirror_only=use_mirror_only,
    )

    rows: list[dict] = []
    english_results = []
    wolfram_results = []
    disagreements = []

    with dataset_path.open(encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    for index, item in enumerate(dataset, start=1):
        logger.info("Evaluating %s (%d/%d)", item["id"], index, len(dataset))

        english_start = time.perf_counter()
        try:
            english_decision, english_ok = english_evaluator.evaluate(item["prompt"])
        except EnglishEvaluatorError as exc:
            logger.info("English evaluator error for %s: %s", item["id"], exc)
            english_decision = SymbolicDecision(
                decision=Decision.REVIEW,
                reason="English policy parse failure",
                triggeredRules=["english_parse_failure"],
                severity=0.5,
            )
            english_ok = False
        english_latency_ms = round((time.perf_counter() - english_start) * 1000, 2)

        semantic, wolfram_decision, wolfram_ok, wolfram_latency = wolfram_evaluator.evaluate(
            item["prompt"]
        )

        row = build_comparison_row(
            item=item,
            english_decision=english_decision,
            wolfram_decision=wolfram_decision,
            english_ok=english_ok,
            wolfram_ok=wolfram_ok,
            english_latency_ms=english_latency_ms,
            wolfram_latency_ms=round(wolfram_latency * 1000, 2),
            semantic=semantic,
        )
        rows.append(row)
        english_results.append({**row, "evaluator": "english"})
        wolfram_results.append({**row, "evaluator": "wolfram"})
        if row["englishDecision"] != row["wolframDecision"]:
            disagreements.append(row)

    summary = compute_metrics(rows)
    summary["models"] = {
        "english": "mock" if use_mock else "ollama/llama3",
        "wolfram": (
            "mock-parser+mirror"
            if use_mock and use_mirror_only
            else "mock-parser+wolfram"
            if use_mock
            else "ollama-parser+mirror"
            if use_mirror_only
            else "ollama-parser+wolfram"
        ),
    }
    summary["dataset"] = str(dataset_path)
    summary["mode"] = "mock" if use_mock else "live"
    summary["useMirrorOnly"] = use_mirror_only

    if validate_parity and not use_mirror_only:
        parity = validate_mirror_parity(rows)
        summary["mirrorParity"] = {
            "checked": parity["checked"],
            "agreementRate": parity["agreementRate"],
            "mismatchCount": len(parity["mismatches"]),
        }
        write_parity_report(output_dir, parity)

    _write_jsonl(output_dir / "english_guardrail_results.jsonl", english_results)
    _write_jsonl(output_dir / "wolfram_guardrail_results.jsonl", wolfram_results)
    _write_jsonl(output_dir / "disagreements.jsonl", disagreements)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_metrics_csv(output_dir / "metrics.csv", summary)
    generate_report(output_dir)
    return summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_metrics_csv(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluator", "metric", "value"])
        for evaluator in ("english", "wolfram"):
            for metric, value in summary[evaluator].items():
                writer.writerow([evaluator, metric, value])
        writer.writerow(["comparison", "agreementRate", summary["agreementRate"]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare English and Wolfram guardrails.")
    parser.add_argument("--dataset", default="datasets/evaluation_prompts.jsonl")
    parser.add_argument("--output", default="results")
    parser.add_argument("--live", action="store_true", help="Use live Ollama instead of mocks")
    parser.add_argument(
        "--mirror-only",
        action="store_true",
        help="Use Python mirror for Wolfram rules (Colab bulk scoring)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip mirror vs real Wolfram parity validation",
    )
    args = parser.parse_args(argv)

    summary = run_comparison(
        Path(args.dataset),
        Path(args.output),
        use_mock=not args.live,
        use_mirror_only=args.mirror_only,
        validate_parity=not args.skip_parity,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
