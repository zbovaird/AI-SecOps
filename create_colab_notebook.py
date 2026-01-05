#!/usr/bin/env python3
"""
Create a Colab-ready Jupyter notebook (.ipynb) from markdown cell files
"""

import json
import re
from pathlib import Path

def extract_code_from_markdown(md_file):
    """Extract Python code blocks from markdown file"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        return code_blocks
    except Exception as e:
        print(f"Warning: Could not read {md_file}: {e}")
        return []

def create_notebook_cell(source, cell_type='code'):
    """Create a notebook cell dictionary"""
    if cell_type == 'code':
        return {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': source.splitlines(keepends=True)
        }
    else:  # markdown
        return {
            'cell_type': 'markdown',
            'metadata': {},
            'source': source.splitlines(keepends=True)
        }

def create_colab_notebook():
    """Create complete Colab notebook from all cells"""
    cells = []
    
    # Add markdown header cell
    header_cell = create_notebook_cell("""# Latent Space Red Teaming - Complete Analysis

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
- **Automatic result saving** and Google Drive backup""", cell_type='markdown')
    cells.append(header_cell)
    
    # Read cells from UPDATED_CELLS_1-10.md (includes Cell 1)
    updated_cells_file = Path('UPDATED_CELLS_1-10.md')
    if updated_cells_file.exists():
        print("Reading UPDATED_CELLS_1-10.md...")
        code_blocks = extract_code_from_markdown(updated_cells_file)
        print(f"  Found {len(code_blocks)} code blocks")
        for i, code in enumerate(code_blocks):
            # Skip the first line if it's just a comment with cell number
            lines = code.split('\n')
            if lines and lines[0].startswith('# Cell'):
                code = '\n'.join(lines[1:])
            if code.strip():  # Only add non-empty cells
                cells.append(create_notebook_cell(code))
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(code_blocks)} cells...")
    
    # Read cells from COMPLETE_NOTEBOOK_CELLS.md (cells 11+)
    complete_cells_file = Path('COMPLETE_NOTEBOOK_CELLS.md')
    if complete_cells_file.exists():
        print("Reading COMPLETE_NOTEBOOK_CELLS.md...")
        code_blocks = extract_code_from_markdown(complete_cells_file)
        print(f"  Found {len(code_blocks)} code blocks")
        for i, code in enumerate(code_blocks):
            # Skip the first line if it's just a comment with cell number
            lines = code.split('\n')
            if lines and lines[0].startswith('# Cell'):
                code = '\n'.join(lines[1:])
            if code.strip():  # Only add non-empty cells
                cells.append(create_notebook_cell(code))
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(code_blocks)} cells...")
    
    # Create notebook structure
    notebook = {
        'cells': cells,
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
    
    return notebook

if __name__ == '__main__':
    print("Creating Colab notebook...")
    notebook = create_colab_notebook()
    
    output_file = 'latent_space_redteaming_complete.ipynb'
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Created {output_file}")
    print(f"  Total cells: {len(notebook['cells'])}")
    print(f"\nTo use:")
    print(f"  1. Upload {output_file} to Google Colab")
    print(f"  2. Upload redteam_kit folder to Colab")
    print(f"  3. Run all cells")
