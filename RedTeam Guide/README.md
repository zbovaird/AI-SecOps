# AI Red Teaming Guide

## Files in this folder

- **AI_Red_Teaming_Guide.md** - Source markdown file
- **AI_Red_Teaming_Guide.html** - HTML version (recommended for PDF conversion)

## To Create PDF

### Option 1: From HTML (Recommended)
1. Open `AI_Red_Teaming_Guide.html` in your browser (Chrome/Safari/Firefox)
2. Press `Cmd+P` (Mac) or `Ctrl+P` (Windows)
3. Select "Save as PDF" as the destination
4. Click Save

### Option 2: Using pandoc with LaTeX
```bash
# Install BasicTeX (requires password)
brew install --cask basictex

# Restart terminal, then:
pandoc AI_Red_Teaming_Guide.md -o AI_Red_Teaming_Guide.pdf
```

### Option 3: Online converter
1. Go to https://md2pdf.netlify.app/
2. Paste the contents of `AI_Red_Teaming_Guide.md`
3. Download the PDF

## Contents

1. Overview of Framework v1 (Latent Space) and v2 (Behavioral)
2. Detailed explanation of each experiment
3. Glossary of technical terms
4. Score interpretation guide
5. Combined testing pipeline
6. Real examples from Gemma-2-2b-it testing






