# Latent Space Red Teaming Framework (v1)

Internal/structural red teaming for LLMs through latent space analysis.

## What This Framework Does

Unlike behavioral testing (prompt-based attacks), this framework analyzes the **internal structure** of LLMs to find vulnerabilities:

- **Jacobian Analysis**: Computes condition numbers (κ) to find ill-conditioned layers
- **Gradient Attacks**: FGSM, PGD, BIM, MIM attacks on embedding space
- **Vulnerability Basin Detection**: Identifies structurally weak regions
- **Singular Value Analysis**: Tracks σ_min, σ_max for collapse/amplification detection
- **CKA Similarity**: Cross-layer activation correlation analysis

## When to Use This vs. Behavioral Testing

| Use Case | This Framework | Behavioral (framework-v2) |
|----------|---------------|---------------------------|
| Find practical bypasses | ❌ | ✅ |
| Understand WHY model fails | ✅ | ❌ |
| Research model internals | ✅ | ❌ |
| Test before deployment | ❌ | ✅ |
| Build defenses | ✅ | ❌ |

**Best approach**: Run behavioral testing first, then use this framework to understand the structural reasons behind any failures found.

## Quick Start (Google Colab)

1. Open the notebook directly in Colab:
   
   [Open in Colab](https://colab.research.google.com/github/zbovaird/AI-SecOps/blob/framework-v1/latent_space_framework/notebooks/latent_space_redteaming.ipynb)

2. Set runtime to GPU (Runtime → Change runtime type → GPU)

3. Run all cells

## Structure

```
latent_space_framework/
├── notebooks/
│   └── latent_space_redteaming.ipynb  # Main Colab notebook
├── redteam_kit/                        # Analysis modules
│   └── core/modules/
│       ├── gradient_attacks.py         # FGSM, PGD, BIM, MIM
│       ├── latent_space_analysis.py    # Jacobian computation
│       ├── cka_analysis.py             # CKA similarity
│       ├── collapse_induction.py       # Collapse detection
│       └── ...
└── README.md
```

## Key Outputs

- **Compositional Kappa (κ_comp)**: Condition number of MLP blocks - high values indicate steerable layers
- **Vulnerability Basins**: Layers with low entropy, high rank deficiency
- **Gradient Attack Results**: Per-layer response to embedding perturbations
- **Singular Value Distribution**: σ_min/σ_max trends across layers

## Requirements

- GPU with 16GB+ VRAM (T4 minimum, A100 recommended)
- HuggingFace account (for gated models like Gemma)
- ~30-60 minutes runtime for full analysis

## Related

- **framework-v2**: Behavioral red teaming (decode fragility, multi-turn drift, etc.)
- Combined usage provides both "what fails" (v2) and "why it fails" (v1)
