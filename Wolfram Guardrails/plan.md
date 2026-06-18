# Wolfram Guardrails Plan

See the full rebuild plan in Cursor or refer to `cursor_handoff/` for detailed architecture and comparison design.

## Phases

1. **Core pipeline** — Ollama parser → semantic JSON → Wolfram rules → decision
2. **Comparison eval** — English LLM policy vs Wolfram symbolic rules over shared dataset

## Success Criteria

- `pytest -q` → 13 passed
- CLI blocks policy-evasion prompts with `dangerous_intent`
- Comparison produces `results/comparison_summary.json` and `results/disagreements.jsonl`
