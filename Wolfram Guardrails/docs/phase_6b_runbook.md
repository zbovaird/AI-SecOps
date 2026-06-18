# Phase 6b Finish Runbook

Phase 6b is **not complete** until GPU latent-space analysis runs on disagreement cohorts.
Local export + report generation is done; embedding probe and Colab analysis are not.

## Current status

| Step | Status | Artifact |
|------|--------|----------|
| Phase 6 live comparison | Done | `results/comparison_summary.json` |
| Cohort export | Done (refresh after each Phase 6 run) | `results/latent_space/cohort_*.jsonl` |
| Local embedding probe | **Not run** | needs `pip install -e ".[embeddings]"` |
| Colab GPU analysis | **Not run** | `notebooks/latent_space_disagreements.ipynb` |
| Join-back to repo | Pending | `results/latent_space/colab_output/` |

## Do not reuse unchanged

- `latent_space_framework/notebooks/latent_space_redteaming.ipynb` — built for Gemma-2-2b-it + 30–50 generic red-team prompts
- `latentspacetestingresults/` — reference methodology only; wrong model and wrong prompt set for Wolfram guardrail disagreements

## Finish checklist

### 1. Refresh exports (local, ~seconds)

```bash
cd '/Users/zachbovaird/Documents/GitHub/Wolfram Guardrails'
.venv/bin/python -m eval.latent_space_runner --results results
```

Re-run whenever Phase 6 comparison results change.

### 2. Local embedding probe (optional pre-GPU, ~minutes)

PyTorch / sentence-transformers may not install on Python 3.13. Options:

```bash
# Option A: Python 3.12 venv
python3.12 -m venv .venv312 && source .venv312/bin/activate
pip install -e ".[embeddings]"
python -m eval.latent_space_runner --results results
```

Tests the hypothesis that false-allow prompts embed near benign controls.

### 3. Colab GPU run (main 6b deliverable)

Open `notebooks/latent_space_disagreements.ipynb` in Colab with **GPU runtime**.

Upload to `/content/wolfram_latent_space/`:

- `cohort_controls_agreement_correct.jsonl`
- `cohort_english_false_allow.jsonl`
- `cohort_english_under_block.jsonl`
- `cohort_english_policy_error.jsonl`
- `cohort_wolfram_rule_gap.jsonl`
- `cohort_wolfram_over_review.jsonl`
- `export_summary.json`

Notebook clones `AI-SecOps` for `redteam_kit` and loads **Llama 3 8B Instruct** (same family as Ollama `llama3` English judge).

Outputs:

- `phase0_baseline.json`
- `cohort_cka_vs_controls.json`
- `priority_gradient_attacks.json`

Download to `results/latent_space/colab_output/` in the repo.

### 4. Expanded dataset (48 prompts)

`datasets/evaluation_prompts.jsonl` now has **4 prompts per category** (48 total). Re-run Phase 6 live before Colab:

```bash
.venv/bin/python -m eval.comparison_runner --live --dataset datasets/evaluation_prompts.jsonl --output results
.venv/bin/python -m eval.latent_space_runner --results results
```

Expect ~30–60 min for live Phase 6 on 48 prompts (Ollama + Wolfram per row).

### 5. Dual-path latent analysis

| Path | Neural? | Colab analysis |
|------|---------|----------------|
| English guardrail | Yes — Llama + `english_policy_prompt.txt` | CKA vs controls, gradient attacks (`redteam_kit`) |
| Parser → semantic JSON | Yes — Llama + `parser_prompt.txt` | Parser-path CKA, English-vs-parser divergence |
| Wolfram `rules.wl` | **No** — symbolic predicates on JSON | JSON embedding proxy + rule labels from Phase 6 only |

You cannot observe latent space *inside* Wolfram Language rule evaluation. The Wolfram path's neural component is the **parser** that produces semantic JSON.

## What each path explains

| Failure type | Latent analysis can explain | Cannot explain |
|--------------|----------------------------|----------------|
| `english_false_allow` | Judge steerability, embedding similarity to benign controls | Wolfram rules |
| `english_policy_error` | Policy judge drift under gradient pressure | Symbolic rule gaps |
| `wolfram_rule_gap` / `wolfram_over_review` | Parser risk inflation (shared Ollama model) | Wolfram `rules.wl` thresholds |

Wolfram symbolic decisions are deterministic given semantic JSON. Latent analysis targets the **shared LLM parser** and **English judge**, not Wolfram Engine.

## AI-SecOps assets to reuse

| Asset | Path | Role |
|-------|------|------|
| Analysis engine | `latent_space_framework/redteam_kit/` | `GradientAttackEngine`, `CKAAnalyzer`, hooks |
| Reference notebook | `latent_space_framework/notebooks/latent_space_redteaming.ipynb` | Full metric definitions (Cells 26–32) |
| Prior results | `latentspacetestingresults/` | Example output schema only |

## New notebook (created)

`notebooks/latent_space_disagreements.ipynb` — cohort-aware wrapper; does not replace the full red-team notebook.
