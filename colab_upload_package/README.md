# Colab Upload Package

## Quick Setup

1. **Upload this entire folder to Colab:**
   - Open Google Colab: https://colab.research.google.com/
   - Click folder icon (📁) in left sidebar
   - Click "Upload" → Select this entire folder

2. **Or upload as zip:**
   - Zip this folder: `zip -r colab_upload_package.zip colab_upload_package/`
   - Upload zip to Colab
   - Extract: `!unzip colab_upload_package.zip`

3. **Open notebook:**
   - File → Open → Upload
   - Select `latent_space_redteaming_complete.ipynb`

4. **Set runtime:**
   - Runtime → Change runtime type → GPU (T4 or better)

5. **Run all cells:**
   - Runtime → Run all

## Contents

- `redteam_kit/` - Main package with all modules
- `test_prompts.py` - Test prompts (optional, notebook has inline fallback)
- `latent_space_redteaming_complete.ipynb` - Complete notebook

## Verification

After uploading, run this in Colab to verify:

```python
import sys
sys.path.insert(0, '/content/colab_upload_package')
import redteam_kit
print("✓ redteam_kit imported")

try:
    from test_prompts import test_prompts
    print(f"✓ test_prompts loaded ({len(test_prompts)} prompts)")
except ImportError:
    print("ℹ️ test_prompts.py not found - will use inline prompts")
```
