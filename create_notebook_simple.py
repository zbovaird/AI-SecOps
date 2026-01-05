#!/usr/bin/env python3
"""
Simple notebook creator - processes files in small chunks
"""

import json
import sys

def main():
    # Start building notebook
    cells = []
    
    # Header cell
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
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
    })
    
    # Process UPDATED_CELLS_1-10.md
    print("Processing UPDATED_CELLS_1-10.md...")
    try:
        with open('UPDATED_CELLS_1-10.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        print(f"Found {len(code_blocks)} code blocks")
        
        for i, code in enumerate(code_blocks):
            lines = code.split('\n')
            if lines and lines[0].strip().startswith('# Cell'):
                code = '\n'.join(lines[1:])
            if code.strip():
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [line + '\n' for line in code.split('\n')]
                })
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(code_blocks)} cells")
    except Exception as e:
        print(f"Error reading UPDATED_CELLS_1-10.md: {e}")
        return
    
    # Process COMPLETE_NOTEBOOK_CELLS.md (skip cells 1-10)
    print("\nProcessing COMPLETE_NOTEBOOK_CELLS.md...")
    try:
        with open('COMPLETE_NOTEBOOK_CELLS.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        print(f"Found {len(code_blocks)} code blocks")
        
        for i, code in enumerate(code_blocks):
            lines = code.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            # Skip cells 1-10
            cell_match = re.search(r'# Cell (\d+)', first_line)
            if cell_match and int(cell_match.group(1)) <= 10:
                continue
            
            if first_line.startswith('# Cell'):
                code = '\n'.join(lines[1:])
            
            if code.strip():
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [line + '\n' for line in code.split('\n')]
                })
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(code_blocks)} cells")
    except Exception as e:
        print(f"Error reading COMPLETE_NOTEBOOK_CELLS.md: {e}")
        return
    
    # Create notebook
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
    
    # Write notebook
    output_file = 'latent_space_redteaming_complete.ipynb'
    print(f"\nWriting {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Created {output_file}")
    print(f"  Total cells: {len(cells)}")

if __name__ == '__main__':
    main()
