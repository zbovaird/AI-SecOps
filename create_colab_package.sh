#!/bin/bash
# Create a single package for easy Colab upload

cd "$(dirname "$0")"

PACKAGE_NAME="colab_upload_package"
OUTPUT_DIR="${PACKAGE_NAME}"

echo "Creating Colab upload package..."
echo "================================"

# Create output directory
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Copy redteam_kit folder
echo "Copying redteam_kit folder..."
cp -r redteam_kit "${OUTPUT_DIR}/"

# Copy test_prompts.py
echo "Copying test_prompts.py..."
cp test_prompts.py "${OUTPUT_DIR}/"

# Copy notebook
echo "Copying notebook..."
cp latent_space_redteaming_complete.ipynb "${OUTPUT_DIR}/"

# Create README for Colab
cat > "${OUTPUT_DIR}/README.md" << 'EOF'
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
EOF

# Create zip file
echo ""
echo "Creating zip file..."
cd "${OUTPUT_DIR}"
zip -r "../${PACKAGE_NAME}.zip" . -q
cd ..

echo ""
echo "✓ Package created!"
echo ""
echo "Upload to Colab:"
echo "  1. Upload folder: ${OUTPUT_DIR}/"
echo "  2. Or upload zip: ${PACKAGE_NAME}.zip"
echo ""
echo "Package contents:"
ls -lh "${OUTPUT_DIR}"/
echo ""
echo "Zip file: ${PACKAGE_NAME}.zip ($(du -h ${PACKAGE_NAME}.zip | cut -f1))"
