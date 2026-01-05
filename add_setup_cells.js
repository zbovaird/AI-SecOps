const fs = require('fs');

const nb = JSON.parse(fs.readFileSync('latent_space_redteaming_complete.ipynb', 'utf-8'));

// Setup cell
const setupCell = {
    cell_type: 'code',
    execution_count: null,
    metadata: {},
    outputs: [],
    source: `# SETUP: Required Additional Files
# Run this cell FIRST to check which files need to be uploaded

import os

print("=" * 60)
print("REQUIRED ADDITIONAL FILES CHECK")
print("=" * 60)

# Required additional module files (not included in redteam_kit by default)
required_modules = {
    "gradient_attacks.py": {
        "location": "/content/redteam_kit/core/modules/gradient_attacks.py",
        "description": "Gradient-based attacks (FGSM, PGD, BIM, MIM)",
        "required_for": "Cell 27: Gradient-Based Adversarial Attacks"
    },
    "semantic_perturbation.py": {
        "location": "/content/redteam_kit/core/modules/semantic_perturbation.py",
        "description": "Semantic-preserving perturbation strategies",
        "required_for": "Cell 28: Semantic Perturbation Testing"
    },
    "adaptive_perturbation.py": {
        "location": "/content/redteam_kit/core/modules/adaptive_perturbation.py",
        "description": "Adaptive perturbation with auto-refusal detection",
        "required_for": "Cell 29: Adaptive Perturbation Testing"
    },
    "pyrit_integration.py": {
        "location": "/content/redteam_kit/core/modules/pyrit_integration.py",
        "description": "PyRIT integration (optional, enhances adaptive_perturbation)",
        "required_for": "Cell 29 (optional)",
        "optional": True
    }
}

print("\\n📋 Required Additional Files:\\n")
print("These files are NOT included in the redteam_kit folder by default.")
print("You need to upload them separately:\\n")

missing_files = []
for file_name, info in required_modules.items():
    exists = os.path.exists(info["location"])
    status = "✓" if exists else "❌"
    optional = "(OPTIONAL)" if info.get("optional") else ""
    print(f"{status} {file_name} {optional}")
    print(f"   Description: {info['description']}")
    print(f"   Required for: {info['required_for']}")
    if not exists:
        missing_files.append(file_name)
    print()

if missing_files:
    print("=" * 60)
    print("⚠️  MISSING FILES DETECTED")
    print("=" * 60)
    print("\\nTo upload missing files:\\n")
    print("1. Download these files from your repository:\\n")
    for file_name in missing_files:
        print(f"   - redteam_kit/core/modules/{file_name}")
    print("\\n2. In Colab:")
    print("   a. Click the folder icon (📁) in the left sidebar")
    print("   b. Click upload (⬆️)")
    print("   c. Upload the .py files to /content/")
    print("\\n3. Run the next cell (Cell 3) to automatically move them to the correct location")
    print("\\n4. Re-run this cell to verify all files are in place")
else:
    print("=" * 60)
    print("✅ All required files are present!")
    print("=" * 60)
    print("\\nYou can proceed with running all cells.")

print("\\n" + "=" * 60)
`.split('\n').map(l => l + '\n')
};

// Mover cell
const moverCell = {
    cell_type: 'code',
    execution_count: null,
    metadata: {},
    outputs: [],
    source: `# AUTO-MOVE: Uploaded Files to Correct Location
# Run this cell AFTER uploading files to automatically place them correctly

import os
import shutil

target_dir = "/content/redteam_kit/core/modules"
os.makedirs(target_dir, exist_ok=True)

required_files = ["gradient_attacks.py", "semantic_perturbation.py", "adaptive_perturbation.py", "pyrit_integration.py"]

print("Searching for uploaded files and moving to correct location...")
print("=" * 60)

moved_count = 0
for file_name in required_files:
    target_path = os.path.join(target_dir, file_name)
    
    # Check if already in place
    if os.path.exists(target_path):
        print(f"✓ {file_name} already in place")
        continue
    
    # Search for file
    found = False
    for root, dirs, files in os.walk("/content"):
        if file_name in files:
            source_path = os.path.join(root, file_name)
            try:
                shutil.copy2(source_path, target_path)
                print(f"✓ Found and moved {file_name}")
                print(f"  From: {source_path}")
                print(f"  To:   {target_path}")
                moved_count += 1
                found = True
                break
            except Exception as e:
                print(f"❌ Error moving {file_name}: {e}")
    
    if not found:
        print(f"⚠️  {file_name} not found - please upload it")

print(f"\\n✓ Moved {moved_count} file(s)")
print("\\nRe-run Cell 2 to verify all files are in place.")
`.split('\n').map(l => l + '\n')
};

// Insert after header (index 1)
nb.cells.splice(1, 0, setupCell);
nb.cells.splice(2, 0, moverCell);

// Update gradient attacks cell
nb.cells.forEach((cell, idx) => {
    if (cell.cell_type === 'code' && cell.source) {
        const src = cell.source.join('');
        if (src.includes('Gradient-Based Adversarial Attacks') && src.includes('from redteam_kit.core.modules.gradient_attacks import')) {
            const newSrc = src.replace(
                /try:\s+from redteam_kit\.core\.modules\.gradient_attacks import GradientAttackEngine\s+except ImportError:\s+print\("Installing required packages\.\.\."\)\s+!pip install adversarial-robustness-toolbox -q\s+from redteam_kit\.core\.modules\.gradient_attacks import GradientAttackEngine/g,
                `# Try to import gradient attacks module
try:
    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine
    GRADIENT_ATTACKS_AVAILABLE = True
except ImportError as e:
    GRADIENT_ATTACKS_AVAILABLE = False
    print("⚠️  Gradient attacks module not found")
    print("   The gradient_attacks.py file is missing from your redteam_kit folder")
    print("   \\n   To fix this:")
    print("   1. Upload gradient_attacks.py to Colab (/content/)")
    print("   2. Run Cell 3 to automatically move it to the correct location")
    print("   3. Re-run this cell")
    print(f"   \\n   Error: {e}")
    raise ImportError("gradient_attacks module not available. Please add gradient_attacks.py to redteam_kit/core/modules/")`
            );
            cell.source = newSrc.split('\n').map(l => l + '\n');
            console.log(`✓ Updated gradient attacks cell at index ${idx}`);
        }
        
        // Update adaptive perturbation
        if (src.includes('Adaptive Perturbation Testing') && src.includes('from redteam_kit.core.modules.adaptive_perturbation import')) {
            const newSrc = src.replace(
                /try:\s+from redteam_kit\.core\.modules\.adaptive_perturbation import AdaptivePerturbationEngine, create_simple_model_wrapper\s+except ImportError:\s+print\("Installing required packages\.\.\."\)\s+!pip install sentence-transformers scikit-learn -q\s+from redteam_kit\.core\.modules\.adaptive_perturbation import AdaptivePerturbationEngine, create_simple_model_wrapper/g,
                `# Try to import adaptive perturbation module
try:
    from redteam_kit.core.modules.adaptive_perturbation import AdaptivePerturbationEngine, create_simple_model_wrapper
    ADAPTIVE_PERTURBATION_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_PERTURBATION_AVAILABLE = False
    print("⚠️  Adaptive perturbation module not found")
    print("   The adaptive_perturbation.py file is missing from your redteam_kit folder")
    print("   \\n   To fix this:")
    print("   1. Upload adaptive_perturbation.py to Colab (/content/)")
    print("   2. Run Cell 3 to automatically move it to the correct location")
    print("   3. If pyrit_integration.py is also missing, adaptive_perturbation will work without it")
    print("   4. Re-run this cell")
    print(f"   \\n   Error: {e}")
    raise ImportError("adaptive_perturbation module not available. Please add adaptive_perturbation.py to redteam_kit/core/modules/")`
            );
            cell.source = newSrc.split('\n').map(l => l + '\n');
            console.log(`✓ Updated adaptive perturbation cell at index ${idx}`);
        }
    }
});

fs.writeFileSync('latent_space_redteaming_complete.ipynb', JSON.stringify(nb, null, 1), 'utf-8');
console.log(`\n✅ Notebook updated!`);
console.log(`   Total cells: ${nb.cells.length}`);
console.log(`   Added Cell 2: Required Files Check`);
console.log(`   Added Cell 3: Auto-Move Files`);
console.log(`   Updated error handling in attack cells`);
