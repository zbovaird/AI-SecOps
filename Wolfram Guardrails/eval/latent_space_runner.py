"""Phase 6b: export disagreement cohorts and run local latent-space follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.latent_space.embedding_probe import run_embedding_probe
from eval.latent_space.export import export_latent_space_inputs
from eval.latent_space.report import generate_latent_space_report, write_latent_space_report


def run_phase6b(
    results_dir: Path,
    *,
    skip_embedding_probe: bool = False,
) -> dict:
    results_dir = Path(results_dir)
    latent_dir = results_dir / "latent_space"

    export_summary = export_latent_space_inputs(results_dir, latent_dir)
    embedding_probe = (
        {"enabled": False, "reason": "Skipped by flag"}
        if skip_embedding_probe
        else run_embedding_probe(latent_dir)
    )

    report = generate_latent_space_report(
        export_summary=export_summary,
        embedding_probe=embedding_probe,
        comparison_report_path=results_dir / "comparison_report.json",
    )
    md_path, json_path = write_latent_space_report(latent_dir, report)

    return {
        "exportSummary": export_summary,
        "embeddingProbe": embedding_probe,
        "reportMarkdown": str(md_path),
        "reportJson": str(json_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6b latent-space follow-up on disagreements.")
    parser.add_argument("--results", default="results", help="Directory with Phase 6 comparison artifacts")
    parser.add_argument("--skip-embedding-probe", action="store_true")
    args = parser.parse_args(argv)

    payload = run_phase6b(Path(args.results), skip_embedding_probe=args.skip_embedding_probe)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
