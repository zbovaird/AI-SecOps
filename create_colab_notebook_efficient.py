#!/usr/bin/env python3
"""
Create a Colab-ready Jupyter notebook (.ipynb) from markdown cell files
Memory-efficient version that processes files line by line
"""

import json
import re
from pathlib import Path

def extract_code_blocks_streaming(md_file):
    """Extract Python code blocks from markdown file using streaming"""
    code_blocks = []
    current_block = []
    in_code_block = False
    
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '```python':
                    in_code_block = True
                    current_block = []
                elif line.strip() == '```' and in_code_block:
                    if current_block:
                        code = '\n'.join(current_block)
                        if code.strip():
                            code_blocks.append(code)
                    current_block = []
                    in_code_block = False
                elif in_code_block:
                    current_block.append(line.rstrip('\n'))
        
        # Handle unclosed block
        if in_code_block and current_block:
            code = '\n'.join(current_block)
            if code.strip():
                code_blocks.append(code)
                
    except Exception as e:
        print(f"Warning: Could not read {md_file}: {e}")
        return []
    
    return code_blocks

def create_notebook_cell(source, cell_type='code'):
    """Create a notebook cell dictionary"""
    if cell_type == 'code':
        return {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': source.splitlines(keepends=True) if isinstance(source, str) else source
        }
    else:  # markdown
        return {
            'cell_type': 'markdown',
            'metadata': {},
            'source': source.splitlines(keepends=True) if isinstance(source, str) else source
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
        code_blocks = extract_code_blocks_streaming(updated_cells_file)
        print(f"  Found {len(code_blocks)} code blocks")
        for i, code in enumerate(code_blocks):
            # Skip the first line if it's just a comment with cell number
            lines = code.split('\n')
            if lines and lines[0].strip().startswith('# Cell'):
                code = '\n'.join(lines[1:])
            if code.strip():  # Only add non-empty cells
                cells.append(create_notebook_cell(code))
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(code_blocks)} cells...")
    
    # Read cells from COMPLETE_NOTEBOOK_CELLS.md (cells 11+)
    complete_cells_file = Path('COMPLETE_NOTEBOOK_CELLS.md')
    if complete_cells_file.exists():
        print("Reading COMPLETE_NOTEBOOK_CELLS.md...")
        code_blocks = extract_code_blocks_streaming(complete_cells_file)
        print(f"  Found {len(code_blocks)} code blocks")
        
        # Track which cells we've already added from UPDATED_CELLS_1-10.md
        # We want to skip cells 1-10 from COMPLETE_NOTEBOOK_CELLS.md
        cell_num = 0
        for i, code in enumerate(code_blocks):
            # Check if this is a cell number comment
            lines = code.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            # Extract cell number if present
            cell_match = re.search(r'# Cell (\d+)', first_line)
            if cell_match:
                cell_num = int(cell_match.group(1))
                # Skip cells 1-10 since they're already in UPDATED_CELLS_1-10.md
                if cell_num <= 10:
                    continue
            
            # Skip the first line if it's just a comment with cell number
            if first_line.startswith('# Cell'):
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
    print("Creating Colab notebook (memory-efficient version)...")
    try:
        notebook = create_colab_notebook()
        
        output_file = 'latent_space_redteaming_complete.ipynb'
        print(f"\nWriting notebook to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"\n✓ Created {output_file}")
        print(f"  Total cells: {len(notebook['cells'])}")
        print(f"\nTo use:")
        print(f"  1. Upload {output_file} to Google Colab")
        print(f"  2. Upload redteam_kit folder to Colab")
        print(f"  3. Run all cells")
    except MemoryError:
        print("\n✗ Memory error. Try running with more memory or process files separately.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
