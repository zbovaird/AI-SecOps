# Cursor Rebuild Prompt

Rebuild the Python project `Wolfram Guardrails` from the design in the AI-SecOps notebook.

Pipeline: prompt → Ollama semantic parser → strict JSON → Wolfram Language rules → optional embedding scorer → ALLOW/BLOCK/REVIEW/REWRITE.

The LLM is not the final policy authority. Wolfram Language rules in `guardrails/rules.wl` are the source of truth.

See `LOCAL_RUNTIME_AND_CONFIG.md`, `PROJECT_ARCHITECTURE.md`, and `COLAB_COMPARISON_PLAN.md` in this folder.
