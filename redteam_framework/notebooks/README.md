# Red Team Framework - Colab Notebooks

## Quick Start

### Option 1: Run from GitHub

1. Open the notebook in Colab: [Open in Colab](https://colab.research.google.com/github/zbovaird/AI-SecOps/blob/framework-v2/redteam_framework/notebooks/redteam_framework_colab.ipynb)

2. Or manually:
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub tab
   - Enter: `https://github.com/zbovaird/AI-SecOps`
   - Select branch: `framework-v2`
   - Select: `redteam_framework/notebooks/redteam_framework_colab.ipynb`

### Option 2: Upload Directly

1. Download `redteam_framework_colab.ipynb`
2. Go to [Google Colab](https://colab.research.google.com/)
3. File → Upload notebook
4. Select the downloaded file

## Before Running

1. **Set GPU Runtime**: Runtime → Change runtime type → GPU (T4 or better)
2. **HuggingFace Token**: Have your token ready for gated models (Gemma, Llama, etc.)
3. **Google Drive**: Mount will be requested for saving results

## Notebook Contents

| Cell | Description |
|------|-------------|
| 1-5 | Setup: Install deps, clone repo, authenticate |
| 6 | Configuration: Set model and experiment options |
| 7 | Load Model |
| 8 | Define Prompts |
| 9 | Decode Fragility Experiment |
| 10 | Logit Lens Experiment |
| 11 | Multi-turn Drift Experiment |
| 12 | Attention Routing Experiment |
| 13 | KV Cache Experiment |
| 14 | Combined Report |
| 15 | List Saved Files |
| 16 | Cross-Model Benchmark (Optional) |

## Configuration Options

In Cell 6, you can configure:

```python
MODEL_ID = "google/gemma-2-2b-it"  # Model to test
QUICK_MODE = False  # Faster but less thorough

# Enable/disable experiments
RUN_DECODE_FRAGILITY = True
RUN_LOGIT_LENS = True
RUN_MULTITURN_DRIFT = True
RUN_ATTENTION_ROUTING = True
RUN_KV_CACHE = True
```

## Results

Results are saved to Google Drive at:
```
/content/drive/MyDrive/redteam_framework_results/
├── fragility_report.json
├── logit_lens_report.json
├── drift_report.json
├── attention_report.json
├── kv_cache_report.json
└── combined_report.json
```

## Recommended GPUs

| GPU | Memory | Recommended Max Model Size |
|-----|--------|---------------------------|
| T4 | 16GB | 7B params (float16/bfloat16) |
| V100 | 16GB | 7B params |
| A100 | 40GB | 13B params |
| L4 | 24GB | 8B params |

For Gemma-2-2B, a T4 is sufficient.
