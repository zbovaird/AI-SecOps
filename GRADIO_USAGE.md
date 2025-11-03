# PyRIT Gradio Interface Usage

## Overview

The Gradio web interface provides an interactive way to test AI models using PyRIT without writing Python scripts.

## Starting the Interface

### Option 1: Run in Background (Current)
```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps"
conda activate aisecops
python pyrit_gradio_app.py &
```

### Option 2: Run in Foreground
```bash
cd "/Users/zachbovaird/Documents/GitHub/AI SecOps"
conda activate aisecops
python pyrit_gradio_app.py
```

The interface will be available at: **http://localhost:7860**

## Interface Features

### 🔍 Single Prompt Test Tab
- Test individual prompts against Gemini models
- Enter your prompt in the text box
- Configure GCP Project ID and Region
- View model responses in real-time
- Responses are automatically saved to PyRIT memory

### ⚠️ Prompt Injection Tests Tab
- Run automated injection tests using HarmBench dataset
- Select number of prompts to test (1-10)
- Tests multiple injection patterns automatically
- Shows success/failure for each injection attempt
- Results include full model responses

### 📊 Dataset Explorer Tab
- Explore available security datasets
- View HarmBench dataset information
- See sample prompts and harm categories
- Useful for understanding test data

### 🔌 Connection Test Tab
- Verify Gemini API connection
- Test your GCP credentials
- Check if project ID and region are correct
- Helps troubleshoot connection issues

## Configuration

### Required Settings
1. **GCP Project ID**: Your Google Cloud project ID
   - Default: Reads from `GOOGLE_CLOUD_PROJECT` environment variable
   - Or enter manually in each tab

2. **GCP Region**: Region for Vertex AI
   - Default: `us-central1` (recommended)
   - Alternatives: `us-west2`, `us-east1`

### Environment Setup
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCP_REGION="us-central1"
```

## Usage Examples

### Example 1: Test a Simple Prompt
1. Go to "🔍 Single Prompt Test" tab
2. Enter your prompt: "What is the capital of France?"
3. Enter your Project ID
4. Click "Test Prompt"
5. View the response

### Example 2: Run Injection Tests
1. Go to "⚠️ Prompt Injection Tests" tab
2. Set number of prompts (e.g., 3)
3. Enter Project ID
4. Click "Run Injection Tests"
5. Review results showing which injections succeeded

### Example 3: Explore Datasets
1. Go to "📊 Dataset Explorer" tab
2. Click "Load Dataset Info"
3. See available prompts and their categories

## Troubleshooting

### Server Not Starting
- Check if port 7860 is already in use
- Try a different port: Edit `pyrit_gradio_app.py` and change `server_port=7860`

### Connection Errors
- Verify GCP credentials: `gcloud auth application-default login`
- Check Project ID is correct
- Ensure Vertex AI API is enabled

### Import Errors
- Ensure conda environment is activated
- Install dependencies: `pip install -r requirements.txt`

## Stopping the Server

### If Running in Background
```bash
pkill -f pyrit_gradio_app.py
```

### If Running in Foreground
Press `Ctrl+C` in the terminal

## Integration with PyRIT

All prompts tested through the interface are automatically:
- Stored in PyRIT memory (DuckDBMemory)
- Tagged with conversation IDs
- Available for later analysis

You can access the memory programmatically:
```python
from pyrit.memory import DuckDBMemory
memory = DuckDBMemory()
# Query stored prompts
```

