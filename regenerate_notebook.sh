#!/bin/bash
# Regenerate notebook using Python with minimal memory footprint

cd "$(dirname "$0")"

echo "Regenerating notebook from templates..."
echo "This may take a moment..."

# Use system Python if available, otherwise try python3
PYTHON_CMD=""
for cmd in /usr/bin/python3 python3 /usr/local/bin/python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python not found"
    exit 1
fi

echo "Using: $PYTHON_CMD"

# Create minimal Python script inline
cat > /tmp/build_nb_minimal.py << 'PYEOF'
import json
import re
import sys

def extract_cells(filename):
    cells = []
    current = []
    in_code = False
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if s == '```python':
                in_code = True
                current = []
            elif s == '```' and in_code:
                if current:
                    if current[0].strip().startswith('# Cell'):
                        current = current[1:]
                    if any(l.strip() for l in current):
                        cells.append('\n'.join(current))
                current = []
                in_code = False
            elif in_code:
                current.append(line.rstrip('\n'))
    return cells

nb = {
    'cells': [{
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['# Latent Space Red Teaming - Complete Analysis\n', '\n', 'This notebook includes memory-optimized Phase 3 with batched Jacobian computation.\n']
    }],
    'metadata': {
        'colab': {'provenance': [], 'toc_visible': True},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

print("Extracting cells 1-10...")
for cell in extract_cells('UPDATED_CELLS_1-10.md'):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [l + '\n' for l in cell.split('\n')]
    })

print("Extracting cells 11+...")
for cell in extract_cells('COMPLETE_NOTEBOOK_CELLS.md'):
    lines = cell.split('\n')
    if lines:
        m = re.search(r'# Cell (\d+)', lines[0].strip())
        if m and int(m.group(1)) <= 10:
            continue
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [l + '\n' for l in cell.split('\n')]
    })

print(f"Writing notebook with {len(nb['cells'])} cells...")
with open('latent_space_redteaming_complete.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")
PYEOF

$PYTHON_CMD /tmp/build_nb_minimal.py

rm -f /tmp/build_nb_minimal.py

