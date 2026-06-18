# Project Architecture

## Pipeline

```
user prompt -> pipeline/orchestrator.py
  -> llm/parser.py (Ollama)
  -> llm/schema.py validation
  -> llm/normalizer.py
  -> guardrails/client.py -> guardrails/rules.wl
  -> risk/embedding_scorer.py (optional)
  -> pipeline/decision.py
  -> pipeline/report.py
```

## Key Modules

- `llm/schema.py` — SemanticParse, SymbolicDecision, GuardrailResult
- `guardrails/rules.wl` — DangerousIntentQ, SensitiveTargetQ, EvaluateGuardrails
- `guardrails/policy.py` — Python mirror for fallback/Colab
- `pipeline/orchestrator.py` — end-to-end flow
- `eval/comparison_runner.py` — English vs Wolfram evaluation

## Tests

Six test files, 13 tests total. Mocks avoid requiring live Ollama/Wolfram in CI.
