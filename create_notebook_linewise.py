#!/usr/bin/env python3
"""
Line-by-line notebook creator to avoid memory issues
"""

import json

def extract_cells_linewise(filename):
    """Extract code cells line by line"""
    cells = []
    current_cell = []
    in_code_block = False
    cell_num = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            
            if stripped == '```python':
                in_code_block = True
                current_cell = []
            elif stripped == '```' and in_code_block:
                if current_cell:
                    # Remove cell number comment if present
                    if current_cell and current_cell[0].startswith('# Cell'):
                        current_cell = current_cell[1:]
                    # Only add non-empty cells
                    cell_text = '\n'.join(current_cell)
                    if cell_text.strip():
                        cells.append(cell_text)
                current_cell = []
                in_code_block = False
            elif in_code_block:
                current_cell.append(line.rstrip('\n'))
    
    return cells

def main():
    cells_list = []
    
    # Header
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
    
    cells_list.append(('markdown', header))
    
    # Process UPDATED_CELLS_1-10.md
    print("Reading UPDATED_CELLS_1-10.md...")
    cells_1_10 = extract_cells_linewise('UPDATED_CELLS_1-10.md')
    print(f"Found {len(cells_1_10)} cells")
    for cell in cells_1_10:
        cells_list.append(('code', cell))
    
    # Process COMPLETE_NOTEBOOK_CELLS.md (skip cells 1-10)
    print("Reading COMPLETE_NOTEBOOK_CELLS.md...")
    cells_11_plus = extract_cells_linewise('COMPLETE_NOTEBOOK_CELLS.md')
    print(f"Found {len(cells_11_plus)} cells")
    
    import re
    for cell in cells_11_plus:
        # Check if this is cell 1-10
        lines = cell.split('\n')
        if lines:
            first_line = lines[0].strip()
            match = re.search(r'# Cell (\d+)', first_line)
            if match and int(match.group(1)) <= 10:
                continue
        cells_list.append(('code', cell))
    
    # Build notebook JSON
    print(f"\nBuilding notebook with {len(cells_list)} cells...")
    notebook_cells = []
    
    for cell_type, source in cells_list:
        if cell_type == 'markdown':
            notebook_cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': source.splitlines(keepends=True)
            })
        else:
            notebook_cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': source.splitlines(keepends=True)
            })
    
    notebook = {
        'cells': notebook_cells,
        'metadata': {
            'colab': {
                'provenance': [],
                'toc_visible': True
            },
            'kernelspec': {
                'display_name': 'Python 3',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Write notebook
    output_file = 'latent_space_redteaming_complete.ipynb'
    print(f"Writing {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Created {output_file}")
    print(f"  Total cells: {len(notebook_cells)}")

if __name__ == '__main__':
    main()
