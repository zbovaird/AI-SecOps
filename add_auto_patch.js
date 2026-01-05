const fs = require('fs');

const nb = JSON.parse(fs.readFileSync('latent_space_redteaming_complete.ipynb', 'utf-8'));

// Find adaptive perturbation cell and add auto-patch code
nb.cells.forEach((cell, idx) => {
    if (cell.cell_type === 'code' && cell.source) {
        const src = cell.source.join('');
        if (src.includes('Adaptive Perturbation Testing') && src.includes('from redteam_kit.core.modules.adaptive_perturbation import')) {
            // Create patch code lines
            const patchLines = [
                '# Auto-patch adaptive_perturbation.py to make pyrit_integration optional\n',
                'import os\n',
                "file_path = '/content/redteam_kit/core/modules/adaptive_perturbation.py'\n",
                'if os.path.exists(file_path):\n',
                "    with open(file_path, 'r') as f:\n",
                '        content = f.read()\n',
                "    # Check if already patched\n",
                "    if 'try:\\n    from .pyrit_integration import PyRITOrchestrator' not in content and 'from .pyrit_integration import PyRITOrchestrator' in content:\n",
                "        old_import = 'from .pyrit_integration import PyRITOrchestrator'\n",
                "        new_import = 'try:\\n    from .pyrit_integration import PyRITOrchestrator\\nexcept ImportError:\\n    PyRITOrchestrator = None  # Optional dependency'\n",
                '        content = content.replace(old_import, new_import)\n',
                "        with open(file_path, 'w') as f:\n",
                '            f.write(content)\n',
                "        print('✓ Patched adaptive_perturbation.py to make pyrit_integration optional')\n",
                "    elif 'PyRITOrchestrator = None' in content:\n",
                "        print('✓ adaptive_perturbation.py already patched')\n",
                '\n'
            ];
            
            // Find where to insert (before the try block)
            const lines = cell.source;
            let insertIdx = 0;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].includes('from redteam_kit.core.modules.adaptive_perturbation import')) {
                    insertIdx = i;
                    break;
                }
            }
            
            // Insert patch code
            cell.source.splice(insertIdx, 0, ...patchLines);
            console.log(`✓ Added auto-patch to adaptive perturbation cell at index ${idx}`);
        }
    }
});

fs.writeFileSync('latent_space_redteaming_complete.ipynb', JSON.stringify(nb, null, 1), 'utf-8');
console.log('Notebook updated with auto-patch');
