"""Generate Phase 6b latent-space follow-up report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_latent_space_report(
    *,
    export_summary: dict[str, Any],
    embedding_probe: dict[str, Any],
    comparison_report_path: Path | None = None,
) -> dict[str, Any]:
    comparison_context = {}
    if comparison_report_path and comparison_report_path.exists():
        comparison_context = json.loads(comparison_report_path.read_text(encoding="utf-8"))

    report = {
        "phase": "6b",
        "title": "Latent-Space Follow-Up on Guardrail Disagreements",
        "exportSummary": export_summary,
        "embeddingProbe": embedding_probe,
        "comparisonContext": {
            "agreementRate": comparison_context.get("executiveSummary", {}).get("agreementRate"),
            "englishFalseAllowRate": comparison_context.get("safetyMetrics", {})
            .get("english", {})
            .get("falseAllowRate"),
            "wolframFalseAllowRate": comparison_context.get("safetyMetrics", {})
            .get("wolfram", {})
            .get("falseAllowRate"),
        },
        "recommendedColabAnalysis": {
            "notebook": "notebooks/latent_space_disagreements.ipynb",
            "frameworkPath": "../AI SecOps/latent_space_framework",
            "metrics": [
                "compositional_kappa",
                "vulnerability_basins",
                "gradient_sensitivity",
                "singular_value_collapse",
                "cka_similarity",
            ],
            "priorityCohorts": export_summary.get("analysisPriority", []),
        },
        "hypotheses": _build_hypotheses(export_summary, embedding_probe),
    }
    return report


def write_latent_space_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latent_space_report.json"
    md_path = output_dir / "latent_space_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return md_path, json_path


def _build_hypotheses(export_summary: dict[str, Any], embedding_probe: dict[str, Any]) -> list[str]:
    hypotheses: list[str] = []
    counts = export_summary.get("cohortCounts", {})
    if counts.get("english_false_allow", 0) > 0:
        hypotheses.append(
            "English false allows may correlate with prompts that embed similarly to benign controls "
            "while semantic parser assigns high risk — inspect parser-to-policy gap."
        )
    if counts.get("english_policy_error", 0) > 0:
        hypotheses.append(
            "English LLM policy judge may show instruction-following drift: high-risk semantic parses "
            "still classified ALLOW/REVIEW. Latent-space gradient attacks may reveal steerable basins."
        )
    if counts.get("wolfram_rule_gap", 0) > 0:
        hypotheses.append(
            "Wolfram over-review on benign educational prompts suggests parser risk inflation; "
            "compare activation stability against true benign controls."
        )
    for finding in embedding_probe.get("pairwiseFindings", []):
        if finding.get("embeddingConfusionSignal"):
            hypotheses.append(
                f"Embedding probe: {finding['pair']} shows cross-cohort similarity "
                f"({finding['meanCrossSimilarity']:.3f}) exceeding control cohesion."
            )
    if not hypotheses:
        hypotheses.append("No high-priority cohorts exported; re-run comparison to populate disagreements.")
    return hypotheses


def _render_markdown(report: dict[str, Any]) -> str:
    export = report["exportSummary"]
    probe = report["embeddingProbe"]
    ctx = report["comparisonContext"]
    lines = [
        "# Latent-Space Follow-Up Report (Phase 6b)",
        "",
        "## Context from Phase 6 Comparison",
        "",
        f"- Agreement rate: **{ctx.get('agreementRate', 'n/a')}**",
        f"- English false allow rate: **{ctx.get('englishFalseAllowRate', 'n/a')}**",
        f"- Wolfram false allow rate: **{ctx.get('wolframFalseAllowRate', 'n/a')}**",
        "",
        "## Exported Cohorts",
        "",
        "| Cohort | Count |",
        "| --- | ---: |",
    ]
    for name, count in export.get("cohortCounts", {}).items():
        if count:
            lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "**Priority cohorts for GPU latent-space analysis:**", ""])
    for name in export.get("analysisPriority", []):
        lines.append(f"- `{name}`")

    lines.extend(["", "## Local Embedding Probe", ""])
    if not probe.get("enabled"):
        lines.append(f"Not run: {probe.get('reason', 'unknown')}")
    else:
        lines.append(f"Model: `{probe.get('model', 'n/a')}`")
        lines.extend(["", "| Pair | Cross Sim | Confusion Signal |", "| --- | ---: | --- |"])
        for finding in probe.get("pairwiseFindings", []):
            signal = "yes" if finding.get("embeddingConfusionSignal") else "no"
            lines.append(
                f"| `{finding['pair']}` | {finding['meanCrossSimilarity']:.3f} | {signal} |"
            )

    lines.extend(["", "## Hypotheses for Colab / GPU Analysis", ""])
    for item in report.get("hypotheses", []):
        lines.append(f"- {item}")

    rec = report.get("recommendedColabAnalysis", {})
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            f"Open `{rec.get('notebook', 'notebooks/latent_space_disagreements.ipynb')}` in Colab with GPU runtime.",
            f"Framework reference: `{rec.get('frameworkPath', '../AI SecOps/latent_space_framework')}`",
            "",
            "Target metrics:",
        ]
    )
    for metric in rec.get("metrics", []):
        lines.append(f"- {metric}")

    return "\n".join(lines) + "\n"
