#!/usr/bin/env python3
"""Build notebook line-by-line to avoid memory issues"""
import json

def extract_cells_streaming(filename):
    """Extract code cells by processing line-by-line"""
    cells = []
    current_cell = []
    in_code = False
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '```python':
                in_code = True
                current_cell = []
            elif line.strip() == '```' and in_code:
                if current_cell:
                    # Remove cell number comment
                    if current_cell and current_cell[0].strip().startswith('# Cell'):
                        current_cell = current_cell[1:]
                    if any(l.strip() for l in current_cell):
                        cells.append('\n'.join(current_cell))
                current_cell = []
                in_code = False
            elif in_code:
                current_cell.append(line.rstrip('\n'))
    
    return cells

# Build notebook
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

# Header
header_lines = [
    '# Latent Space Red Teaming - Complete Analysis\n',
    '\n',
    'This notebook performs comprehensive latent space red teaming analysis including:\n',
    '\n',
    '- **Phase 1-7**: Core latent space analysis (vulnerability basins, attention patterns, perturbations)\n',
    '- **Gradient-based attacks**: FGSM, PGD, BIM, MIM attacks on embeddings\n',
    '- **Semantic perturbation**: 15 comprehensive strategies maintaining semantic similarity\n',
    '- **Adaptive perturbation**: Auto-test on refusal with semantic-preserving bypasses\n',
    '- **Additional red teaming attacks**: Jailbreak, prompt injection, context poisoning, token manipulation\n',
    '\n',
    '## Instructions\n',
    '\n',
    '1. **Upload `redteam_kit` folder** to Colab (folder icon → upload)\n',
    '2. **Set runtime to GPU** (Runtime → Change runtime type → GPU - T4 or better)\n',
    '3. **Run all cells** (Runtime → Run all)\n',
    '4. **Authenticate Hugging Face** when prompted in Cell 5\n',
    '5. **All results will be saved automatically** as JSON files\n',
    '6. **Backup to Google Drive** using Cell 26\n',
    '\n',
    '## What\'s Included\n',
    '\n',
    '- **29 cells** covering all phases and attack types\n',
    '- **50+ diverse prompts** for robust analysis\n',
    '- **All new attack capabilities** integrated\n',
    '- **Automatic result saving** and Google Drive backup\n'
]

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': header_lines
})

# Process cells 1-10
print('Extracting cells from UPDATED_CELLS_1-10.md...')
cells_1_10 = extract_cells_streaming('UPDATED_CELLS_1-10.md')
print(f'Found {len(cells_1_10)} cells')

for cell_text in cells_1_10:
    source_lines = [line + '\n' for line in cell_text.split('\n')]
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    })

# Process cells 11+
print('Extracting cells from COMPLETE_NOTEBOOK_CELLS.md...')
cells_11_plus = extract_cells_streaming('COMPLETE_NOTEBOOK_CELLS.md')
print(f'Found {len(cells_11_plus)} cells')

import re
for cell_text in cells_11_plus:
    lines = cell_text.split('\n')
    # Skip cells 1-10
    if lines:
        first = lines[0].strip()
        m = re.search(r'# Cell (\d+)', first)
        if m and int(m.group(1)) <= 10:
            continue
    
    source_lines = [line + '\n' for line in cell_text.split('\n')]
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    })

# Write
print(f'\nWriting notebook with {len(notebook["cells"])} cells...')
with open('latent_space_redteaming_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('Done!')
