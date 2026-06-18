"""Lightweight local embedding probe for disagreement cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_embedding_probe(latent_space_dir: Path) -> dict[str, Any]:
    """Compare embedding geometry across cohorts when sentence-transformers is available."""
    latent_space_dir = Path(latent_space_dir)
    cohort_files = sorted(latent_space_dir.glob("cohort_*.jsonl"))
    if not cohort_files:
        return {"enabled": False, "reason": "No cohort files found"}

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {
            "enabled": False,
            "reason": 'Install embeddings extra: pip install -e ".[embeddings]"',
        }

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cohort_vectors: dict[str, list[tuple[str, Any]]] = {}

    for path in cohort_files:
        name = path.stem.replace("cohort_", "")
        rows = _load_jsonl(path)
        if not rows:
            continue
        prompts = [row["prompt"] for row in rows]
        vectors = model.encode(prompts)
        cohort_vectors[name] = list(zip([row["id"] for row in rows], vectors))

    if len(cohort_vectors) < 2:
        return {"enabled": True, "reason": "Need at least two non-empty cohorts", "cohorts": list(cohort_vectors)}

    import numpy as np

    findings: list[dict[str, Any]] = []
    priority_pairs = [
        ("english_false_allow", "controls_agreement_correct"),
        ("english_policy_error", "controls_agreement_correct"),
        ("english_under_block", "controls_agreement_correct"),
        ("wolfram_rule_gap", "controls_agreement_correct"),
    ]

    for left, right in priority_pairs:
        if left not in cohort_vectors or right not in cohort_vectors:
            continue
        left_vecs = np.array([vec for _, vec in cohort_vectors[left]])
        right_vecs = np.array([vec for _, vec in cohort_vectors[right]])
        cross = _mean_cross_similarity(left_vecs, right_vecs)
        within_left = _mean_within_similarity(left_vecs)
        within_right = _mean_within_similarity(right_vecs)
        findings.append(
            {
                "pair": f"{left}_vs_{right}",
                "meanCrossSimilarity": cross,
                "withinLeftSimilarity": within_left,
                "withinRightSimilarity": within_right,
                "embeddingConfusionSignal": cross > within_right,
            }
        )

    high_risk_rows = _load_jsonl(latent_space_dir / "cohort_english_false_allow.jsonl")
    high_risk_rows += _load_jsonl(latent_space_dir / "cohort_english_policy_error.jsonl")
    parser_risk_notes = [
        {
            "id": row["id"],
            "semanticRisk": row.get("semanticRisk"),
            "englishDecision": row.get("englishDecision"),
            "expectedDecision": row.get("expectedDecision"),
        }
        for row in high_risk_rows
        if (row.get("semanticRisk") or 0) >= 0.5
    ]

    return {
        "enabled": True,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "cohortsAnalyzed": list(cohort_vectors.keys()),
        "pairwiseFindings": findings,
        "highSemanticRiskEnglishFailures": parser_risk_notes,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean_cross_similarity(left: Any, right: Any) -> float:
    import numpy as np

    sims = []
    for lv in left:
        for rv in right:
            sims.append(float(np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv))))
    return sum(sims) / len(sims) if sims else 0.0


def _mean_within_similarity(vectors: Any) -> float:
    import numpy as np

    if len(vectors) < 2:
        return 1.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(
                float(
                    np.dot(vectors[i], vectors[j])
                    / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                )
            )
    return sum(sims) / len(sims) if sims else 0.0
