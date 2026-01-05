#!/usr/bin/env node
const fs = require('fs');

function extractCells(filename) {
    const content = fs.readFileSync(filename, 'utf-8');
    const cells = [];
    const lines = content.split('\n');
    let currentCell = [];
    let inCode = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim() === '```python') {
            inCode = true;
            currentCell = [];
        } else if (line.trim() === '```' && inCode) {
            if (currentCell.length > 0) {
                // Remove cell number comment
                if (currentCell[0] && currentCell[0].trim().startsWith('# Cell')) {
                    currentCell = currentCell.slice(1);
                }
                if (currentCell.some(l => l.trim())) {
                    cells.push(currentCell.join('\n'));
                }
            }
            currentCell = [];
            inCode = false;
        } else if (inCode) {
            currentCell.push(line);
        }
    }
    
    return cells;
}

// Build notebook
const notebook = {
    cells: [],
    metadata: {
        colab: { provenance: [], toc_visible: true },
        kernelspec: { display_name: 'Python 3', name: 'python3' },
        language_info: { name: 'python' }
    },
    nbformat: 4,
    nbformat_minor: 4
};

// Add header
const header = `# Latent Space Red Teaming - Complete Analysis

This notebook performs comprehensive latent space red teaming analysis including:

- **Phase 1-7**: Core latent space analysis (vulnerability basins, attention patterns, perturbations)
- **Gradient-based attacks**: FGSM, PGD, BIM, MIM attacks on embeddings
- **Semantic perturbation**: 15 comprehensive strategies maintaining semantic similarity
- **Adaptive perturbation**: Auto-test on refusal with semantic-preserving bypasses
- **Additional red teaming attacks**: Jailbreak, prompt injection, context poisoning, token manipulation

## Instructions

1. **Upload \`redteam_kit\` folder** to Colab (folder icon → upload)
2. **Set runtime to GPU** (Runtime → Change runtime type → GPU - T4 or better)
3. **Run all cells** (Runtime → Run all)
4. **Authenticate Hugging Face** in Cell 6 (run: !huggingface-cli login)
5. **All results will be saved automatically** as JSON files
6. **Backup to Google Drive** using the final cell

## What's Included

- **33 cells** covering all phases and attack types
- **Jacobian tracking** for collapse detection
- **50+ diverse prompts** for robust analysis
- **All new attack capabilities** integrated
- **Automatic result saving** and Google Drive backup`;

notebook.cells.push({
    cell_type: 'markdown',
    metadata: {},
    source: header.split('\n').map(l => l + '\n')
});

// Process cells 1-10
console.log('Extracting cells from UPDATED_CELLS_1-10.md...');
const cells1_10 = extractCells('UPDATED_CELLS_1-10.md');
console.log(`Found ${cells1_10.length} cells`);

cells1_10.forEach(cellText => {
    const sourceLines = cellText.split('\n').map(l => l + '\n');
    notebook.cells.push({
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines
    });
});

// Process cells 11+
console.log('Extracting cells from COMPLETE_NOTEBOOK_CELLS.md...');
const cells11Plus = extractCells('COMPLETE_NOTEBOOK_CELLS.md');
console.log(`Found ${cells11Plus.length} cells`);

cells11Plus.forEach(cellText => {
    const lines = cellText.split('\n');
    // Skip cells 1-10
    if (lines.length > 0) {
        const first = lines[0].trim();
        const match = first.match(/# Cell (\d+)/);
        if (match && parseInt(match[1]) <= 10) {
            return; // Skip this cell
        }
    }
    
    const sourceLines = cellText.split('\n').map(l => l + '\n');
    notebook.cells.push({
        cell_type: 'code',
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines
    });
});

// Write notebook
console.log(`\nWriting notebook with ${notebook.cells.length} cells...`);
fs.writeFileSync('latent_space_redteaming_complete.ipynb', JSON.stringify(notebook, null, 1), 'utf-8');

console.log('Done!');
console.log(`Created latent_space_redteaming_complete.ipynb with ${notebook.cells.length} cells`);
