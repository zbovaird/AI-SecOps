# PyRIT Programmatic Usage Guide (macOS)

**For macOS users - no GUI required!**

PyRIT is primarily a Python library for programmatic red teaming. This guide shows how to use it without the Gradio GUI.

## Quick Start

### Basic Example

```python
from pyrit.models import PromptRequestPiece
from pyrit.memory import DuckDBMemory

# Create a prompt
prompt = PromptRequestPiece(
    role="user",
    original_value="Your test prompt here",
    conversation_id="test_001"
)

# Initialize memory
memory = DuckDBMemory()

# Use memory to store/retrieve data
# (Check actual API methods with: dir(memory))
```

### Run the Examples

```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps"
conda activate aisecops

# Basic example
python pyrit_basic_example.py

# Complete workflow
python pyrit_test_workflow.py

# Endpoint integration template
python pyrit_endpoint_example.py
```

## Key Components

### 1. PromptRequestPiece
Create prompts for testing:

```python
from pyrit.models import PromptRequestPiece

prompt = PromptRequestPiece(
    role="user",  # or "assistant" or "system"
    original_value="Your prompt text",
    conversation_id="unique_id"  # Optional, groups related prompts
)
```

**Important**: Use `original_value` (not `original_prompt_text`)

### 2. Memory Management
Store and retrieve conversation history:

```python
from pyrit.memory import DuckDBMemory

memory = DuckDBMemory()
# Check available methods: dir(memory)
```

### 3. Datasets
Load pre-built attack datasets:

```python
from pyrit.datasets import (
    fetch_harmbench_dataset,
    fetch_adv_bench_dataset,
    fetch_darkbench_dataset,
    # ... 19+ more datasets
)

dataset = fetch_harmbench_dataset()
```

### 4. Scoring Systems
Evaluate model responses:

```python
from pyrit.score import (
    SelfAskGeneralScorer,
    InsecureCodeScorer,
    PromptShieldScorer,
    # ... 19+ more scorers
)

scorer = SelfAskGeneralScorer()
# scorer.score_text(response)  # Requires model access
```

## Working Examples

All examples are available in this directory:

1. **`pyrit_basic_example.py`** - Basic PyRIT usage
2. **`pyrit_test_workflow.py`** - Complete testing workflow
3. **`pyrit_endpoint_example.py`** - Vertex AI integration template

## API Notes

### PromptRequestPiece
- Use `role="user"` (string, not enum)
- Use `original_value` (not `original_prompt_text`)
- `conversation_id` groups related messages

### Memory API
Check available methods with:
```python
memory = DuckDBMemory()
print([m for m in dir(memory) if not m.startswith('_')])
```

### Datasets
Most datasets return `SeedPromptDataset` objects that can be iterated.

## Integration with Vertex AI

To integrate with Google Vertex AI:

1. **Create a custom ChatTarget** (or use existing if available)
2. **Wrap Vertex AI API calls** in PyRIT structures
3. **Use PyRIT memory and scoring** with your endpoint

Template is in `pyrit_endpoint_example.py`.

## Resources

- **Official Docs**: https://azure.github.io/PyRIT/
- **GitHub**: https://github.com/Azure/PyRIT
- **Capabilities**: See `tools_capabilities.md`

## Common Issues

**"No module named 'rpyc'"**
- Only needed for Gradio GUI (not required for programmatic use)

**"AttributeError: USER"**
- Use `role="user"` (string), not `ChatMessageRole.USER`

**"original_prompt_text" not found**
- Use `original_value` instead

---

*All examples tested on macOS with PyRIT v0.9.0*

