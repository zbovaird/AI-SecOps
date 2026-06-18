# Latent-Space Follow-Up Report (Phase 6b)

## Context from Phase 6 Comparison

- Agreement rate: **0.4791666666666667**
- English false allow rate: **0.3333333333333333**
- Wolfram false allow rate: **0.16666666666666666**

## Exported Cohorts

| Cohort | Count |
| --- | ---: |
| `english_false_allow` | 16 |
| `english_under_block` | 12 |
| `wolfram_over_review` | 2 |
| `english_policy_error` | 15 |
| `wolfram_rule_gap` | 6 |
| `both_wrong` | 8 |
| `controls_agreement_correct` | 19 |
| `controls_agreement` | 23 |

**Priority cohorts for GPU latent-space analysis:**

- `english_false_allow`
- `english_under_block`
- `english_policy_error`
- `wolfram_rule_gap`
- `wolfram_over_review`
- `both_wrong`

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
