"""Export disagreement subsets for latent-space analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


COHORT_RULES = {
    "english_false_allow": lambda row: (
        row.get("expectedDecision") in {"BLOCK", "REVIEW"} and row.get("englishDecision") == "ALLOW"
    ),
    "english_under_block": lambda row: (
        row.get("expectedDecision") == "BLOCK" and row.get("englishDecision") in {"ALLOW", "REVIEW"}
    ),
    "wolfram_over_review": lambda row: (
        row.get("expectedDecision") == "ALLOW" and row.get("wolframDecision") == "REVIEW"
    ),
    "english_policy_error": lambda row: row.get("failureType") == "english_policy_error",
    "wolfram_rule_gap": lambda row: row.get("failureType") == "wolfram_rule_gap",
    "both_wrong": lambda row: not row.get("englishCorrect", True) and not row.get("wolframCorrect", True),
    "controls_agreement_correct": lambda row: (
        row.get("evaluatorsAgree", False)
        and row.get("englishCorrect", False)
        and row.get("wolframCorrect", False)
    ),
    "controls_agreement": lambda row: row.get("evaluatorsAgree", False),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_cohorts(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cohorts: dict[str, list[dict[str, Any]]] = {name: [] for name in COHORT_RULES}
    for row in rows:
        for name, predicate in COHORT_RULES.items():
            if predicate(row):
                cohorts[name].append(_export_record(row, cohort=name))
    return cohorts


def _export_record(row: dict[str, Any], *, cohort: str) -> dict[str, Any]:
    semantic = row.get("semanticParse") or {}
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "category": row.get("category", "unknown"),
        "expectedDecision": row["expectedDecision"],
        "englishDecision": row.get("englishDecision"),
        "wolframDecision": row.get("wolframDecision"),
        "failureType": row.get("failureType"),
        "cohort": cohort,
        "semanticRisk": semantic.get("risk"),
        "semanticIntent": semantic.get("intent"),
        "englishCorrect": row.get("englishCorrect"),
        "wolframCorrect": row.get("wolframCorrect"),
    }


def export_latent_space_inputs(
    results_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir or results_dir / "latent_space")
    output_dir.mkdir(parents=True, exist_ok=True)

    english_rows = load_jsonl(results_dir / "english_guardrail_results.jsonl")
    disagreement_rows = load_jsonl(results_dir / "disagreements.jsonl")
    source_rows = english_rows or disagreement_rows
    if not source_rows and (results_dir / "comparison_summary.json").exists():
        source_rows = disagreement_rows

    cohorts = classify_cohorts(source_rows)
    summary = {
        "sourceResultsDir": str(results_dir),
        "totalRows": len(source_rows),
        "disagreementCount": len(disagreement_rows),
        "cohortCounts": {name: len(items) for name, items in cohorts.items()},
        "analysisPriority": _priority_cohorts(cohorts),
    }

    for name, items in cohorts.items():
        if items:
            _write_jsonl(output_dir / f"cohort_{name}.jsonl", items)

    all_export = [item for items in cohorts.values() for item in items]
    _write_jsonl(output_dir / "all_cohorts.jsonl", all_export)
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _priority_cohorts(cohorts: dict[str, list[dict[str, Any]]]) -> list[str]:
    priority: list[str] = []
    for name in (
        "english_false_allow",
        "english_under_block",
        "english_policy_error",
        "wolfram_rule_gap",
        "wolfram_over_review",
        "both_wrong",
    ):
        if cohorts.get(name):
            priority.append(name)
    return priority


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
