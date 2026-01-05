
import json
import os

filepath = 'latent_space_redteaming_complete3.ipynb'

if not os.path.exists(filepath):
    print(f"Error: {filepath} not found.")
    exit(1)

with open(filepath, 'r') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

changes_made = 0

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    source_text = "".join(cell['source'])
    
    # Target 1: Encoding Attacks
    if "ExternalTool Encoding Attack Strategies" in source_text and "try:" in source_text:
        print("Found Encoding Attack cell. Cleaning...")
        cell['source'] = [
            "# Logic removed to prevent syntax errors and import issues\n",
            "print('ExternalTool Encoding Attack Strategies - MOVED TO EXTERNAL NOTEBOOK')\n",
            "print('Skipping this section as functionality is now in pyrit_redteaming.ipynb')\n"
        ]
        changes_made += 1
        
    # Target 2: Prompt Injection
    if "Phase 9: Prompt Injection Testing" in source_text and "PromptInjectionModule" in source_text:
        print("Found Prompt Injection cell. Cleaning...")
        cell['source'] = [
            "# Logic removed to prevent syntax errors and import issues\n",
            "print('Phase 9: Prompt Injection Testing - MOVED TO EXTERNAL NOTEBOOK')\n",
            "print('Skipping this section as functionality is now in pyrit_redteaming.ipynb')\n"
        ]
        changes_made += 1

if changes_made > 0:
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully cleaned {changes_made} cells.")
else:
    print("No matching cells found to clean. They might have already been modified.")
