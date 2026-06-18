"""Generate human-readable and structured comparison reports from eval results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.metrics import compute_metrics
from eval.outcomes import enrich_outcome_fields


def load_comparison_results(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    input_dir = Path(input_dir)
    summary_path = input_dir / "comparison_summary.json"
    results_path = input_dir / "english_guardrail_results.jsonl"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing comparison summary: {summary_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing english guardrail results: {results_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("evaluator", None)
            rows.append(enrich_outcome_fields(row))

    parity_path = input_dir / "mirror_parity.json"
    parity = None
    if parity_path.exists():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))

    return rows, summary, parity


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _category_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("category", "unknown")].append(row)

    breakdown: list[dict[str, Any]] = []
    for category in sorted(grouped):
        category_rows = grouped[category]
        total = len(category_rows)
        breakdown.append(
            {
                "category": category,
                "count": total,
                "englishAccuracy": _rate(
                    sum(1 for row in category_rows if row["englishCorrect"]),
                    total,
                ),
                "wolframAccuracy": _rate(
                    sum(1 for row in category_rows if row["wolframCorrect"]),
                    total,
                ),
                "agreementRate": _rate(
                    sum(1 for row in category_rows if row["evaluatorsAgree"]),
                    total,
                ),
                "englishFalseAllowRate": _rate(
                    sum(
                        1
                        for row in category_rows
                        if row["expectedDecision"] in {"BLOCK", "REVIEW"}
                        and row["englishDecision"] == "ALLOW"
                    ),
                    total,
                ),
                "wolframFalseAllowRate": _rate(
                    sum(
                        1
                        for row in category_rows
                        if row["expectedDecision"] in {"BLOCK", "REVIEW"}
                        and row["wolframDecision"] == "ALLOW"
                    ),
                    total,
                ),
                "disagreements": sum(1 for row in category_rows if not row["evaluatorsAgree"]),
            }
        )
    return breakdown


def _disagreement_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disagreements = [row for row in rows if not row["evaluatorsAgree"]]
    by_failure_type = dict(Counter(row["failureType"] for row in disagreements))
    return {
        "totalDisagreements": len(disagreements),
        "byFailureType": by_failure_type,
    }


def _disagreement_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "category": row.get("category", "unknown"),
            "expectedDecision": row["expectedDecision"],
            "englishDecision": row["englishDecision"],
            "wolframDecision": row["wolframDecision"],
            "failureType": row["failureType"],
        }
        for row in rows
        if not row["evaluatorsAgree"]
    ]


def _mirror_parity_summary(parity: dict[str, Any] | None) -> dict[str, Any] | None:
    if parity is None:
        return None
    return {
        "checked": parity.get("checked", 0),
        "agreementRate": parity.get("agreementRate", 0.0),
        "mismatchCount": len(parity.get("mismatches", [])),
        "mismatches": parity.get("mismatches", []),
    }


def _policy_trace_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        trace = row.get("policyTrace") or {}
        traces.append(
            {
                "id": row["id"],
                "category": row.get("category", "unknown"),
                "expectedDecision": row["expectedDecision"],
                "prompt": row.get("prompt", ""),
                "englishDecision": row.get("englishDecision"),
                "wolframDecision": row.get("wolframDecision"),
                "evaluatorsAgree": row.get("evaluatorsAgree"),
                "failureType": row.get("failureType"),
                "policyTrace": trace,
            }
        )
    return traces


def build_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(rows)
    return {
        "executiveSummary": {
            "datasetSize": metrics["datasetSize"],
            "agreementRate": metrics["agreementRate"],
            "englishAccuracy": metrics["english"]["accuracy"],
            "wolframAccuracy": metrics["wolfram"]["accuracy"],
            "dataset": summary.get("dataset"),
            "mode": summary.get("mode"),
            "models": summary.get("models"),
        },
        "safetyMetrics": {
            "english": {
                "falseAllowRate": metrics["english"]["falseAllowRate"],
                "falseBlockRate": metrics["english"]["falseBlockRate"],
                "falseReviewRate": metrics["english"]["falseReviewRate"],
            },
            "wolfram": {
                "falseAllowRate": metrics["wolfram"]["falseAllowRate"],
                "falseBlockRate": metrics["wolfram"]["falseBlockRate"],
                "falseReviewRate": metrics["wolfram"]["falseReviewRate"],
            },
            "primaryMetric": "falseAllowRate",
        },
        "categoryBreakdown": _category_breakdown(rows),
        "disagreementAnalysis": _disagreement_analysis(rows),
        "mirrorParity": _mirror_parity_summary(parity),
        "latency": {
            "englishAvgMs": metrics["english"]["avgLatencyMs"],
            "wolframAvgMs": metrics["wolfram"]["avgLatencyMs"],
        },
        "disagreements": _disagreement_entries(rows),
        "policyTraces": _policy_trace_entries(rows),
        "metrics": metrics,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["executiveSummary"]
    safety = report["safetyMetrics"]
    lines = [
        "# Guardrail Comparison Report",
        "",
        "## Executive Summary",
        "",
        f"- Dataset size: **{summary['datasetSize']}**",
        f"- Evaluator agreement rate: **{_pct(summary['agreementRate'])}**",
        f"- English accuracy: **{_pct(summary['englishAccuracy'])}**",
        f"- Wolfram accuracy: **{_pct(summary['wolframAccuracy'])}**",
    ]
    if summary.get("dataset"):
        lines.append(f"- Dataset: `{summary['dataset']}`")
    if summary.get("mode"):
        lines.append(f"- Mode: `{summary['mode']}`")
    if summary.get("models"):
        lines.append(f"- Models: `{json.dumps(summary['models'], ensure_ascii=True)}`")

    lines.extend(
        [
            "",
            "## Safety Metrics",
            "",
            "**Primary metric: false allow rate** (unsafe prompts incorrectly allowed).",
            "",
            "| Evaluator | False Allow | False Block | False Review |",
            "| --- | ---: | ---: | ---: |",
            f"| English | {_pct(safety['english']['falseAllowRate'])} | "
            f"{_pct(safety['english']['falseBlockRate'])} | "
            f"{_pct(safety['english']['falseReviewRate'])} |",
            f"| Wolfram | {_pct(safety['wolfram']['falseAllowRate'])} | "
            f"{_pct(safety['wolfram']['falseBlockRate'])} | "
            f"{_pct(safety['wolfram']['falseReviewRate'])} |",
            "",
            "## Per-Category Breakdown",
            "",
            "| Category | Count | English Acc | Wolfram Acc | Agreement | Eng False Allow | Wol False Allow | Disagreements |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in report["categoryBreakdown"]:
        lines.append(
            f"| {item['category']} | {item['count']} | {_pct(item['englishAccuracy'])} | "
            f"{_pct(item['wolframAccuracy'])} | {_pct(item['agreementRate'])} | "
            f"{_pct(item['englishFalseAllowRate'])} | {_pct(item['wolframFalseAllowRate'])} | "
            f"{item['disagreements']} |"
        )

    disagreement = report["disagreementAnalysis"]
    lines.extend(
        [
            "",
            "## Disagreement Analysis",
            "",
            f"Total disagreements: **{disagreement['totalDisagreements']}**",
            "",
            "| Failure Type | Count |",
            "| --- | ---: |",
        ]
    )
    for failure_type, count in sorted(disagreement["byFailureType"].items()):
        lines.append(f"| {failure_type} | {count} |")

    parity = report.get("mirrorParity")
    if parity is not None:
        lines.extend(
            [
                "",
                "## Mirror Parity Summary",
                "",
                f"- Rows checked: **{parity['checked']}**",
                f"- Mirror vs Wolfram agreement: **{_pct(parity['agreementRate'])}**",
                f"- Mismatches: **{parity['mismatchCount']}**",
            ]
        )

    latency = report["latency"]
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- English average latency: **{latency['englishAvgMs']:.2f} ms**",
            f"- Wolfram average latency: **{latency['wolframAvgMs']:.2f} ms**",
            "",
            "## Disagreements",
            "",
            "| ID | Category | Expected | English | Wolfram | Failure Type |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["disagreements"]:
        lines.append(
            f"| {item['id']} | {item['category']} | {item['expectedDecision']} | "
            f"{item['englishDecision']} | {item['wolframDecision']} | {item['failureType']} |"
        )

    lines.extend(["", "## Per-Prompt Policy Traces", ""])
    for entry in report.get("policyTraces", []):
        trace = entry.get("policyTrace", {})
        english = trace.get("englishPath", {})
        wolfram = trace.get("wolframPath", {})
        lines.extend(
            [
                f"### `{entry['id']}` ({entry.get('category', 'unknown')})",
                "",
                f"**Prompt:** {entry.get('prompt', '')}",
                "",
                f"- Expected: `{entry.get('expectedDecision')}` | English: `{entry.get('englishDecision')}` | "
                f"Wolfram: `{entry.get('wolframDecision')}` | Failure type: `{entry.get('failureType')}`",
                "",
                "**English path** (raw prompt → English policy LLM):",
                f"- Policy file: `{english.get('policyPromptFile', 'n/a')}`",
                f"- Output: `{json.dumps(english.get('output', {}), ensure_ascii=True)}`",
                "",
                "**Wolfram path** (raw prompt → semantic JSON → symbolic rules):",
                f"- Parser file: `{wolfram.get('parserPromptFile', 'n/a')}`",
                f"- Rules file: `{wolfram.get('rulesFile', 'n/a')}`",
                f"- Semantic JSON: `{json.dumps(wolfram.get('semanticJson', {}), ensure_ascii=True)}`",
            ]
        )
        rule_hits = wolfram.get("ruleEvaluation") or []
        if rule_hits:
            lines.append("- Wolfram rule hits:")
            for hit in rule_hits:
                lines.append(
                    f"  - `{hit.get('rule')}` via `{hit.get('predicate')}` — {hit.get('matchedBecause')}"
                )
        wolfram_out = wolfram.get("output", {})
        lines.append(f"- Wolfram output: `{json.dumps(wolfram_out, ensure_ascii=True)}`")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def generate_report(input_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, parity = load_comparison_results(input_dir)
    report = build_report(rows, summary, parity)

    json_path = output_dir / "comparison_report.json"
    md_path = output_dir / "comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate comparison report from eval results.")
    parser.add_argument(
        "--input",
        default="results",
        help="Directory containing comparison_summary.json and result jsonl files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (defaults to --input)",
    )
    args = parser.parse_args(argv)

    report = generate_report(Path(args.input), output_dir=Path(args.output) if args.output else None)
    print(json.dumps(report["executiveSummary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
