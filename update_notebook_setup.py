#!/usr/bin/env python3
"""Update notebook with setup cells and better error handling"""
import json

# Read notebook
with open('latent_space_redteaming_complete.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Setup cell (Cell 2)
setup_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# SETUP: Required Additional Files\n',
        '# Run this cell FIRST to check which files need to be uploaded\n',
        '\n',
        'import os\n',
        '\n',
        'print("=" * 60)\n',
        'print("REQUIRED ADDITIONAL FILES CHECK")\n',
        'print("=" * 60)\n',
        '\n',
        '# Required additional module files (not included in redteam_kit by default)\n',
        'required_modules = {\n',
        '    "gradient_attacks.py": {\n',
        '        "location": "/content/redteam_kit/core/modules/gradient_attacks.py",\n',
        '        "description": "Gradient-based attacks (FGSM, PGD, BIM, MIM)",\n',
        '        "required_for": "Cell 27: Gradient-Based Adversarial Attacks"\n',
        '    },\n',
        '    "semantic_perturbation.py": {\n',
        '        "location": "/content/redteam_kit/core/modules/semantic_perturbation.py",\n',
        '        "description": "Semantic-preserving perturbation strategies",\n',
        '        "required_for": "Cell 28: Semantic Perturbation Testing"\n',
        '    },\n',
        '    "adaptive_perturbation.py": {\n',
        '        "location": "/content/redteam_kit/core/modules/adaptive_perturbation.py",\n',
        '        "description": "Adaptive perturbation with auto-refusal detection",\n',
        '        "required_for": "Cell 29: Adaptive Perturbation Testing"\n',
        '    },\n',
        '    "pyrit_integration.py": {\n',
        '        "location": "/content/redteam_kit/core/modules/pyrit_integration.py",\n',
        '        "description": "PyRIT integration (optional, enhances adaptive_perturbation)",\n',
        '        "required_for": "Cell 29 (optional)",\n',
        '        "optional": True\n',
        '    }\n',
        '}\n',
        '\n',
        'print("\\n📋 Required Additional Files:\\n")\n',
        'print("These files are NOT included in the redteam_kit folder by default.")\n',
        'print("You need to upload them separately:\\n")\n',
        '\n',
        'missing_files = []\n',
        'for file_name, info in required_modules.items():\n',
        '    exists = os.path.exists(info["location"])\n',
        '    status = "✓" if exists else "❌"\n',
        '    optional = "(OPTIONAL)" if info.get("optional") else ""\n',
        '    print(f"{status} {file_name} {optional}")\n',
        '    print(f"   Description: {info[\"description\"]}")\n',
        '    print(f"   Required for: {info[\"required_for\"]}")\n',
        '    if not exists:\n',
        '        missing_files.append(file_name)\n',
        '    print()\n',
        '\n',
        'if missing_files:\n',
        '    print("=" * 60)\n',
        '    print("⚠️  MISSING FILES DETECTED")\n',
        '    print("=" * 60)\n',
        '    print("\\nTo upload missing files:\\n")\n',
        '    print("1. Download these files from your repository:\\n")\n',
        '    for file_name in missing_files:\n',
        '        print(f"   - redteam_kit/core/modules/{file_name}")\n',
        '    print("\\n2. In Colab:")\n',
        '    print("   a. Click the folder icon (📁) in the left sidebar")\n',
        '    print("   b. Click upload (⬆️)")\n',
        '    print("   c. Upload the .py files to /content/")\n',
        '    print("\\n3. Run the next cell (Cell 3) to automatically move them to the correct location")\n',
        '    print("\\n4. Re-run this cell to verify all files are in place")\n',
        'else:\n',
        '    print("=" * 60)\n',
        '    print("✅ All required files are present!")\n',
        '    print("=" * 60)\n',
        '    print("\\nYou can proceed with running all cells.")\n',
        '\n',
        'print("\\n" + "=" * 60)\n'
    ]
}

# File mover cell (Cell 3)
mover_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# AUTO-MOVE: Uploaded Files to Correct Location\n',
        '# Run this cell AFTER uploading files to automatically place them correctly\n',
        '\n',
        'import os\n',
        'import shutil\n',
        '\n',
        'target_dir = "/content/redteam_kit/core/modules"\n',
        'os.makedirs(target_dir, exist_ok=True)\n',
        '\n',
        'required_files = ["gradient_attacks.py", "semantic_perturbation.py", "adaptive_perturbation.py", "pyrit_integration.py"]\n',
        '\n',
        'print("Searching for uploaded files and moving to correct location...")\n',
        'print("=" * 60)\n',
        '\n',
        'moved_count = 0\n',
        'for file_name in required_files:\n',
        '    target_path = os.path.join(target_dir, file_name)\n',
        '    \n',
        '    # Check if already in place\n',
        '    if os.path.exists(target_path):\n',
        '        print(f"✓ {file_name} already in place")\n',
        '        continue\n',
        '    \n',
        '    # Search for file\n',
        '    found = False\n',
        '    for root, dirs, files in os.walk("/content"):\n',
        '        if file_name in files:\n',
        '            source_path = os.path.join(root, file_name)\n',
        '            try:\n',
        '                shutil.copy2(source_path, target_path)\n',
        '                print(f"✓ Found and moved {file_name}")\n',
        '                print(f"  From: {source_path}")\n',
        '                print(f"  To:   {target_path}")\n',
        '                moved_count += 1\n',
        '                found = True\n',
        '                break\n',
        '            except Exception as e:\n',
        '                print(f"❌ Error moving {file_name}: {e}")\n',
        '    \n',
        '    if not found:\n',
        '        print(f"⚠️  {file_name} not found - please upload it")\n',
        '\n',
        'print(f"\\n✓ Moved {moved_count} file(s)")\n',
        'print("\\nRe-run Cell 2 to verify all files are in place.")\n'
    ]
}

# Insert cells after header (index 1)
nb['cells'].insert(1, setup_cell)
nb['cells'].insert(2, mover_cell)

# Now update cells with better error handling
# Find gradient attacks cell and update it
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell['source']:
        source_text = ''.join(cell['source'])
        
        # Update gradient attacks cell
        if 'Gradient-Based Adversarial Attacks' in source_text and 'from redteam_kit.core.modules.gradient_attacks import' in source_text:
            old_import = """try:
    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine
except ImportError:
    print("Installing required packages...")
    !pip install adversarial-robustness-toolbox -q
    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine"""
            
            new_import = """# Try to import gradient attacks module
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
    raise ImportError("gradient_attacks module not available. Please add gradient_attacks.py to redteam_kit/core/modules/")"""
            
            cell['source'] = ''.join(cell['source']).replace(old_import, new_import).splitlines(keepends=True)
            print(f"✓ Updated gradient attacks cell at index {i}")
        
        # Update adaptive perturbation cell
        if 'Adaptive Perturbation Testing' in source_text and 'from redteam_kit.core.modules.adaptive_perturbation import' in source_text:
            old_import = """try:
    from redteam_kit.core.modules.adaptive_perturbation import AdaptivePerturbationEngine, create_simple_model_wrapper
except ImportError:
    print("Installing required packages...")
    !pip install sentence-transformers scikit-learn -q
    from redteam_kit.core.modules.adaptive_perturbation import AdaptivePerturbationEngine, create_simple_model_wrapper"""
            
            new_import = """# Try to import adaptive perturbation module
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
    raise ImportError("adaptive_perturbation module not available. Please add adaptive_perturbation.py to redteam_kit/core/modules/")"""
            
            cell['source'] = ''.join(cell['source']).replace(old_import, new_import).splitlines(keepends=True)
            print(f"✓ Updated adaptive perturbation cell at index {i}")

# Write updated notebook
with open('latent_space_redteaming_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook updated!")
print(f"   Total cells: {len(nb['cells'])}")
print(f"   Added Cell 2: Required Files Check")
print(f"   Added Cell 3: Auto-Move Files")
print(f"   Updated error handling in attack cells")
