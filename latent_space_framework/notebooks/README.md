# Latent Space Red Teaming - Colab Notebooks

## Quick Start

### Option 1: Open Directly from GitHub

Click this link to open in Colab:

[**Open Latent Space Red Teaming Notebook**](https://colab.research.google.com/github/zbovaird/AI-SecOps/blob/framework-v1/latent_space_framework/notebooks/latent_space_redteaming.ipynb)

### Option 2: Manual Upload

1. Download `latent_space_redteaming.ipynb`
2. Go to [Google Colab](https://colab.research.google.com/)
3. File → Upload notebook
4. Select the downloaded file

## Before Running

1. **Set GPU Runtime**: Runtime → Change runtime type → **GPU** (T4 or better)
2. **HuggingFace Token**: Have your token ready for gated models (Gemma, Llama)
3. **Time**: Full analysis takes 30-60 minutes

## What the Notebook Does

### Phase 0: Baseline Characterization
- Computes baseline metrics (κ, σ_min, σ_max) for benign prompts
- Establishes percentile-based thresholds

### Phase 1: Target Identification
- Classifies layers as steerable, chaotic, collapsed, or stable
- Computes compositional kappa for full MLP blocks
- Identifies high-κ targets for attacks

### Phase 2-4: Attack Execution
- Gradient attacks (FGSM, PGD, BIM, MIM) on embeddings
- Three-way evaluation (semantic, policy, quality deltas)
- Jacobian tracking per attack

### Phase 5: Reproducibility Testing
- Tests across different seeds
- Tests across prompt paraphrases

### Final Export
- Saves all results to Google Drive
- JSON/CSV formats for analysis

## GPU Requirements

| GPU | VRAM | Works? |
|-----|------|--------|
| T4 | 16GB | ✅ Yes (recommended minimum) |
| V100 | 16GB | ✅ Yes |
| A100 | 40GB | ✅ Best |
| L4 | 24GB | ✅ Yes |

## Output Files

Results are saved to Google Drive at:
```
/content/drive/MyDrive/redteam_results/
├── phase1_targets.json           # Identified vulnerable layers
├── gradient_attack_results.json  # Attack outcomes
├── complete_analysis.json        # Full analysis data
└── layer_summary.csv             # Per-layer metrics
```

## Troubleshooting

**"CUDA out of memory"**: 
- Restart runtime and try again
- Reduce batch size in gradient attack cells

**"Module not found"**:
- Ensure Cell 2 (clone repo) ran successfully
- Restart runtime if needed

**"HuggingFace authentication failed"**:
- Run the huggingface-cli login cell
- Use a valid token with model access
