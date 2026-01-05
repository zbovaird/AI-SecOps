#!/usr/bin/env python3
"""Build notebook incrementally - write as we go to avoid memory issues"""
import json
import re

output_file = 'latent_space_redteaming_complete.ipynb'

# Start notebook structure
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

# Add header
print("Adding header...")
notebook['cells'].append({
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
        '- **29+ cells** covering all phases and attack types\n',
        '- **50+ diverse prompts** for robust analysis\n',
        '- **All new attack capabilities** integrated\n',
        '- **Automatic result saving** and Google Drive backup\n',
        '\n',
        '## Memory Optimizations\n',
        '\n',
        '- **Phase 3**: Uses batched autograd, only stores statistics (not full matrices)\n',
        '- **Only vulnerability basins**: Processes only 23 identified basins, not all layers\n',
        '- **Cache clearing**: Automatic CUDA cache management between computations\n'
    ]
})

def extract_cell_from_file(filename, start_marker='```python'):
    """Generator that yields cells one at a time"""
    current_cell = []
    in_code = False
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            
            if stripped == start_marker:
                in_code = True
                current_cell = []
            elif stripped == '```' and in_code:
                if current_cell:
                    # Remove cell number comment
                    if current_cell[0].strip().startswith('# Cell'):
                        current_cell = current_cell[1:]
                    cell_text = '\n'.join(current_cell)
                    if cell_text.strip():
                        yield cell_text
                current_cell = []
                in_code = False
            elif in_code:
                current_cell.append(line.rstrip('\n'))

# Process UPDATED_CELLS_1-10.md
print("Processing UPDATED_CELLS_1-10.md...")
count = 0
for cell_text in extract_cell_from_file('UPDATED_CELLS_1-10.md'):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in cell_text.split('\n')]
    })
    count += 1
    if count % 5 == 0:
        print(f"  Added {count} cells...", end='\r')

print(f"\n  Added {count} cells from UPDATED_CELLS_1-10.md")

# Process COMPLETE_NOTEBOOK_CELLS.md (skip cells 1-10)
print("Processing COMPLETE_NOTEBOOK_CELLS.md...")
count = 0
for cell_text in extract_cell_from_file('COMPLETE_NOTEBOOK_CELLS.md'):
    lines = cell_text.split('\n')
    if lines:
        first_line = lines[0].strip()
        match = re.search(r'# Cell (\d+)', first_line)
        if match:
            cell_num = int(match.group(1))
            if cell_num <= 10:
                continue  # Skip cells 1-10
    
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in cell_text.split('\n')]
    })
    count += 1
    if count % 5 == 0:
        print(f"  Added {count} cells...", end='\r')

print(f"\n  Added {count} cells from COMPLETE_NOTEBOOK_CELLS.md")

# Write notebook
print(f"\nWriting notebook to {output_file}...")
print(f"Total cells: {len(notebook['cells'])} (1 markdown + {len(notebook['cells'])-1} code)")

try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✓ Successfully created {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"  Ready to upload to Google Colab!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

