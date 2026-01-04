# CKA Transferability Analysis Notebook

## Quick Start

1. Click the button below to open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zbovaird/AI-SecOps/blob/framework-v3/transferability_framework/notebooks/transferability_analysis.ipynb)

2. Set your HuggingFace token (required for gated models like Gemma and Llama)
3. Optionally set OpenAI/Anthropic API keys for closed model testing
4. Run Phase 1 (CKA) first - this works without API keys
5. Run Phase 2 (Attacks) and Phase 3 (Correlation) for full analysis

## What This Does

### Phase 1: Open Model CKA
- Loads open-weight models one at a time (memory efficient)
- Extracts hidden states for a set of prompts
- Computes Centered Kernel Alignment (CKA) between all model pairs
- Identifies which models have similar internal representations

### Phase 2: Attack Testing
- Runs 30 standard adversarial prompts against all models
- Tests: jailbreaks, prompt injections, roleplay, encoding attacks
- Classifies responses as: refused, complied, or unclear
- Computes per-model and per-category success rates

### Phase 3: Correlation Analysis
- Correlates CKA similarity with attack transferability
- Finds the best surrogate model for closed targets (GPT-4, Claude)
- Predicts attack success based on surrogate model results
- Generates recommendations for red teaming

## Models Tested

**Open-Weight (HuggingFace):**
- google/gemma-2-2b-it
- mistralai/Ministral-3-8B-Instruct-2512
- meta-llama/Llama-3.2-3B-Instruct
- microsoft/phi-2
- Qwen/Qwen2.5-3B-Instruct

**Closed-Weight (API):**
- OpenAI GPT-4
- Anthropic Claude 3.5 Sonnet

## Outputs

All results are saved to Google Drive at `/content/drive/MyDrive/transferability_results/`:
- `cka_matrix.json` - CKA similarity scores between open models
- `attack_results.json` - Attack success rates per model
- `transferability_report.json` - Full correlation analysis and recommendations

## Red Team Applications

1. **Surrogate Selection**: Find which open model best predicts closed model behavior
2. **Attack Development**: Test attacks on open models first, then transfer to closed
3. **Coverage Analysis**: Identify attack categories that transfer well
4. **Prediction**: Estimate attack success on closed models without testing directly
