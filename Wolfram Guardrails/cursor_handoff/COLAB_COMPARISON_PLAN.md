# Colab Comparison Plan

Compare English guardrails (LLM policy judge) vs Wolfram guardrails (semantic JSON + symbolic rules) over the same prompt dataset.

## Metrics

- Accuracy, false allow/block/review rates
- BLOCK/REVIEW precision-recall
- Agreement rate between English and Wolfram
- Latency and JSON parse failure rate

## Colab Constraint

Use Python mirror (`guardrails/policy.py`) for bulk Colab scoring. Validate locally with real Wolfram Engine.

## Output Artifacts

```
results/english_guardrail_results.jsonl
results/wolfram_guardrail_results.jsonl
results/comparison_summary.json
results/disagreements.jsonl
results/metrics.csv
```
