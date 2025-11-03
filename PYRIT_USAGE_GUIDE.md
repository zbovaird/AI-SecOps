# PyRIT Usage Guide

**Version**: 0.9.0  
**Last Updated**: Based on actual installation exploration

## Interface Types

PyRIT is **NOT a command-line tool**. It's a Python library that can be used in several ways:

### 1. Python Scripts (Most Common)
PyRIT is primarily used programmatically in Python scripts.

```python
# Example script
from pyrit.models import PromptRequestPiece
from pyrit.memory import DuckDBMemory
from pyrit.score import SelfAskGeneralScorer

# Create a prompt
prompt = PromptRequestPiece(
    role="user",
    original_prompt_text="Your prompt here"
)

# Use memory to store results
memory = DuckDBMemory()

# Score a response
scorer = SelfAskGeneralScorer()
```

**Run with**: `python your_script.py`

---

### 2. Jupyter Notebooks (Recommended for Development)
PyRIT works excellently in Jupyter notebooks for interactive red teaming.

**Advantages:**
- Interactive exploration
- Step-by-step execution
- Easy visualization
- Great for experimentation

**Usage**: 
- Create a `.ipynb` notebook
- Import PyRIT components
- Run cells interactively

---

### 3. Gradio GUI (For Manual Scoring)
PyRIT includes a **web-based Gradio interface** for human-in-the-loop scoring.

#### Launching the GUI:

```python
from pyrit.score import HumanInTheLoopScorerGradio

# Create and launch the GUI
scorer = HumanInTheLoopScorerGradio(open_browser=True)
scorer.launch()  # Opens web browser at http://localhost:7860
```

**Features:**
- Web-based interface
- Manual review and scoring of LLM responses
- Useful for live attack scenarios
- Real-time human assessment

**When to use:**
- When you need human judgment on responses
- For quality assurance of automated scorers
- During live red team exercises

---

### 4. Docker + JupyterLab (Easiest Setup)
Microsoft provides a Docker image with everything pre-configured.

```bash
# Pull and run the Docker image
docker pull ghcr.io/microsoft/pyrit:latest
docker run -p 8888:8888 ghcr.io/microsoft/pyrit:latest

# Access JupyterLab at:
# http://localhost:8888
```

**Includes:**
- JupyterLab pre-configured
- All PyRIT dependencies
- Example notebooks
- Tutorials and documentation

**Best for:**
- First-time users
- Quick setup
- Learning PyRIT
- Consistent environment

---

## Available Tools & Components

### Models & Data Structures
Located in `pyrit.models`:
- `PromptRequestPiece` - Individual prompt requests
- `PromptRequestResponse` - Request/response structures
- `ChatMessage` - Chat message data structures
- `SeedPrompt` - Seed prompts for attacks
- `ChatMessagesDataset` - Chat datasets
- `StorageIO` - Storage operations
- And 20+ more...

### Scoring Systems
Located in `pyrit.score`:
- `SelfAskGeneralScorer` - General self-ask scoring
- `SelfAskCategoryScorer` - Category-based scoring
- `SelfAskTrueFalseScorer` - True/false scoring
- `SelfAskLikertScorer` - Likert scale scoring
- `SelfAskRefusalScorer` - Refusal detection
- `CompositeScorer` - Combine multiple scorers
- `InsecureCodeScorer` - Insecure code detection
- `PromptShieldScorer` - Prompt shield evaluation
- `AzureContentFilterScorer` - Azure content filtering
- `GandalfScorer` - Gandalf game scoring
- `HumanInTheLoopScorer` - Manual scoring interface
- `HumanInTheLoopScorerGradio` - **GUI for manual scoring**
- And 11+ more scorers...

### Memory Management
Located in `pyrit.memory`:
- `DuckDBMemory` - Local DuckDB database
- `AzureSQLMemory` - Azure SQL database
- `CentralMemory` - Centralized memory
- `MemoryInterface` - Base interface
- And more...

### Datasets
Located in `pyrit.datasets`:
- `fetch_adv_bench_dataset()` - AdvBench
- `fetch_harmbench_dataset()` - HarmBench
- `fetch_darkbench_dataset()` - DarkBench
- `fetch_red_team_social_bias_dataset()` - Social bias
- `fetch_tdc23_redteaming_dataset()` - TDC23
- `fetch_wmdp_dataset()` - WMDP
- `fetch_xstest_dataset()` - XSTest
- And 12+ more datasets...

### Common Utilities
Located in `pyrit.common`:
- `download_file()` - File downloads
- `display_image_response()` - Image display
- `is_in_ipython_session()` - Jupyter detection
- And more utilities...

---

## Quick Start Examples

### Example 1: Launch the Gradio GUI

```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps"
conda activate aisecops

python -c "from pyrit.score import HumanInTheLoopScorerGradio; HumanInTheLoopScorerGradio(open_browser=True).launch()"
```

This will:
1. Start a Gradio web server
2. Open your browser automatically
3. Provide an interface for manual scoring

### Example 2: Create a Simple Python Script

```python
#!/usr/bin/env python3
"""Simple PyRIT red team script"""

from pyrit.models import PromptRequestPiece
from pyrit.memory import DuckDBMemory

# Create memory store
memory = DuckDBMemory()

# Create a prompt
prompt = PromptRequestPiece(
    role="user",
    original_prompt_text="What is the capital of France?"
)

# Store in memory
memory.insert_prompt_entries([prompt])

print("Prompt stored successfully!")
```

Save as `simple_redteam.py` and run: `python simple_redteam.py`

### Example 3: Use Datasets

```python
from pyrit.datasets import fetch_harmbench_dataset

# Load a dataset
dataset = fetch_harmbench_dataset()

# Use the dataset for red teaming
for entry in dataset:
    # Process each prompt
    pass
```

---

## Command Summary

**PyRIT does NOT have a CLI command** like `pyrit --help`. Instead:

| Task | Command |
|------|---------|
| Run Python script | `python script.py` |
| Launch Gradio GUI | `python -c "from pyrit.score import HumanInTheLoopScorerGradio; HumanInTheLoopScorerGradio().launch()"` |
| Run Jupyter notebook | `jupyter notebook` or use Docker |
| Use Docker | `docker run -p 8888:8888 ghcr.io/microsoft/pyrit:latest` |

---

## Recommended Workflow

1. **Start with Docker** (if available):
   - Get familiar with PyRIT in JupyterLab
   - Follow included examples

2. **Move to Python scripts**:
   - Create your own red team scripts
   - Integrate with your workflow

3. **Use Gradio GUI** when needed:
   - For manual scoring
   - Quality assurance
   - Human-in-the-loop scenarios

---

## Resources

- **Official Documentation**: https://azure.github.io/PyRIT/
- **GitHub Repository**: https://github.com/Azure/PyRIT
- **Docker Image**: `ghcr.io/microsoft/pyrit:latest`

---

## Troubleshooting

### No CLI Command Found
**Expected behavior**: PyRIT is not a CLI tool. Use Python scripts instead.

### Import Errors
```bash
# Make sure you're in the right environment
conda activate aisecops

# Verify installation
python -c "import pyrit; print(pyrit.__version__)"
```

### Gradio GUI Not Launching
```bash
# Check if Gradio is installed
pip install gradio

# Try launching again
python -c "from pyrit.score import HumanInTheLoopScorerGradio; HumanInTheLoopScorerGradio().launch()"
```

---

*This guide is based on actual exploration of PyRIT v0.9.0. For the most up-to-date information, refer to the official documentation.*

