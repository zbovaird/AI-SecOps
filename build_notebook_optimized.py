#!/usr/bin/env python3
"""Build notebook with memory optimizations - processes files in chunks"""
import json
import re
import sys

def extract_cells_chunked(filename):
    """Extract code cells by processing in chunks to avoid memory issues"""
    cells = []
    current_cell = []
    in_code = False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                stripped = line.strip()
                
                if stripped == '```python':
                    in_code = True
                    current_cell = []
                elif stripped == '```' and in_code:
                    if current_cell:
                        # Remove cell number comment from first line
                        if current_cell and current_cell[0].strip().startswith('# Cell'):
                            current_cell = current_cell[1:]
                        # Only add non-empty cells
                        cell_text = '\n'.join(current_cell)
                        if cell_text.strip():
                            cells.append(cell_text)
                    current_cell = []
                    in_code = False
                elif in_code:
                    current_cell.append(line.rstrip('\n'))
                
                # Progress indicator every 1000 lines
                if line_num % 1000 == 0 and line_num > 0:
                    print(f"  Processed {line_num} lines...", end='\r')
        
        print(f"  Processed file: {len(cells)} cells extracted")
        return cells
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

# Build notebook structure
print("Building notebook structure...")
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
print("Adding header...")
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

notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': header_lines
})

# Process cells 1-10 from UPDATED_CELLS_1-10.md
print("\nExtracting cells from UPDATED_CELLS_1-10.md...")
cells_1_10 = extract_cells_chunked('UPDATED_CELLS_1-10.md')
print(f"Found {len(cells_1_10)} cells from UPDATED_CELLS_1-10.md")

for idx, cell_text in enumerate(cells_1_10, 1):
    source_lines = [line + '\n' for line in cell_text.split('\n')]
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    })
    if idx % 5 == 0:
        print(f"  Added {idx} cells...", end='\r')

print(f"\n  Added all {len(cells_1_10)} cells from UPDATED_CELLS_1-10.md")

# Process cells 11+ from COMPLETE_NOTEBOOK_CELLS.md
print("\nExtracting cells from COMPLETE_NOTEBOOK_CELLS.md...")
cells_11_plus = extract_cells_chunked('COMPLETE_NOTEBOOK_CELLS.md')
print(f"Found {len(cells_11_plus)} cells from COMPLETE_NOTEBOOK_CELLS.md")

# Filter out cells 1-10 and add the rest
added_count = 0
for cell_text in cells_11_plus:
    lines = cell_text.split('\n')
    if lines:
        first_line = lines[0].strip()
        # Check if this is a numbered cell <= 10
        match = re.search(r'# Cell (\d+)', first_line)
        if match:
            cell_num = int(match.group(1))
            if cell_num <= 10:
                continue  # Skip cells 1-10 (already added from UPDATED_CELLS_1-10.md)
    
    source_lines = [line + '\n' for line in cell_text.split('\n')]
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    })
    added_count += 1
    if added_count % 5 == 0:
        print(f"  Added {added_count} cells...", end='\r')

print(f"\n  Added {added_count} cells from COMPLETE_NOTEBOOK_CELLS.md")

# Write notebook
output_file = 'latent_space_redteaming_complete.ipynb'
print(f"\nWriting notebook to {output_file}...")
print(f"Total cells: {len(notebook['cells'])} (1 markdown header + {len(notebook['cells'])-1} code cells)")

try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✓ Successfully created {output_file}")
    print(f"  Ready to upload to Google Colab!")
except Exception as e:
    print(f"✗ Error writing notebook: {e}")
    sys.exit(1)

