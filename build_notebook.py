#!/usr/bin/env python3
import json
import re

def extract_cells(filename):
    """Extract code blocks from markdown file"""
    cells = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find all Python code blocks
    pattern = r'```python\n(.*?)\n```'
    blocks = re.findall(pattern, content, re.DOTALL)
    for block in blocks:
        lines = block.split('\n')
        # Remove cell number comment if present
        if lines and lines[0].strip().startswith('# Cell'):
            block = '\n'.join(lines[1:])
        if block.strip():
            cells.append(block)
    return cells

# Build notebook structure
notebook = {
    'cells': [],
    'metadata': {
        'colab': {'provenance': [], 'toc_visible': True},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Add markdown header
header = """# Latent Space Red Teaming - Complete Analysis

This notebook performs comprehensive latent space red teaming analysis including:

- **Phase 1-7**: Core latent space analysis (vulnerability basins, attention patterns, perturbations)
- **Gradient-based attacks**: FGSM, PGD, BIM, MIM attacks on embeddings
- **Semantic perturbation**: 15 comprehensive strategies maintaining semantic similarity
- **Adaptive perturbation**: Auto-test on refusal with semantic-preserving bypasses
- **Additional red teaming attacks**: Jailbreak, prompt injection, context poisoning, token manipulation

## Instructions

1. **Upload `redteam_kit` folder** to Colab (folder icon → upload)
2. **Set runtime to GPU** (Runtime → Change runtime type → GPU - T4 or better)
3. **Run all cells** (Runtime → Run all)
4. **Authenticate Hugging Face** when prompted in Cell 5
5. **All results will be saved automatically** as JSON files
6. **Backup to Google Drive** using Cell 26

## What's Included

- **29 cells** covering all phases and attack types
- **50+ diverse prompts** for robust analysis
- **All new attack capabilities** integrated
- **Automatic result saving** and Google Drive backup"""

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': header.splitlines(keepends=True)
})

# Process cells 1-10
print('Processing UPDATED_CELLS_1-10.md...')
try:
    cells_1_10 = extract_cells('UPDATED_CELLS_1-10.md')
    print(f'  Found {len(cells_1_10)} cells')
    for i, cell in enumerate(cells_1_10):
        notebook['cells'].append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in cell.split('\n')]
        })
except Exception as e:
    print(f'  Error: {e}')

# Process cells 11+
print('Processing COMPLETE_NOTEBOOK_CELLS.md...')
try:
    cells_11_plus = extract_cells('COMPLETE_NOTEBOOK_CELLS.md')
    print(f'  Found {len(cells_11_plus)} cells')
    for cell in cells_11_plus:
        lines = cell.split('\n')
        # Skip cells 1-10 if they appear in this file
        if lines:
            first_line = lines[0].strip()
            match = re.search(r'# Cell (\d+)', first_line)
            if match and int(match.group(1)) <= 10:
                continue
        notebook['cells'].append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in cell.split('\n')]
        })
except Exception as e:
    print(f'  Error: {e}')

# Write notebook
output_file = 'latent_space_redteaming_complete.ipynb'
print(f'\nWriting {output_file} with {len(notebook["cells"])} cells...')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'✓ Created {output_file}')
print(f'  Total cells: {len(notebook["cells"])}')
