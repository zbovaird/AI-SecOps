# Latent-Space Follow-Up Report (Phase 6b)

## Context from Phase 6 Comparison

- Agreement rate: **0.4166666666666667**
- English false allow rate: **0.16666666666666666**
- Wolfram false allow rate: **0.0**

## Exported Cohorts

| Cohort | Count |
| --- | ---: |
| `english_false_allow` | 2 |
| `english_under_block` | 3 |
| `wolfram_over_review` | 1 |
| `english_policy_error` | 6 |
| `wolfram_rule_gap` | 1 |
| `controls_agreement_correct` | 5 |
| `controls_agreement` | 5 |

**Priority cohorts for GPU latent-space analysis:**

- `english_false_allow`
- `english_under_block`
- `english_policy_error`
- `wolfram_rule_gap`
- `wolfram_over_review`

## Local Embedding Probe

Not run: Install embeddings extra: pip install -e ".[embeddings]"

## Hypotheses for Colab / GPU Analysis

- English false allows may correlate with prompts that embed similarly to benign controls while semantic parser assigns high risk — inspect parser-to-policy gap.
- English LLM policy judge may show instruction-following drift: high-risk semantic parses still classified ALLOW/REVIEW. Latent-space gradient attacks may reveal steerable basins.
- Wolfram over-review on benign educational prompts suggests parser risk inflation; compare activation stability against true benign controls.

## Recommended Next Step

Open `notebooks/latent_space_disagreements.ipynb` in Colab with GPU runtime.
Framework reference: `../AI SecOps/latent_space_framework`

Target metrics:
- compositional_kappa
- vulnerability_basins
- gradient_sensitivity
- singular_value_collapse
- cka_similarity
