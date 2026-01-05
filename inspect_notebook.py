
import json

filepath = 'latent_space_redteaming_complete3.ipynb'

with open(filepath, 'r') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell.get('source', []))
    
    # Check for Duplicate Cell 1 candidates (look for pip installs that indicate setup)
    if "pip install adversarial-robustness-toolbox" in source_str:
        print(f"\n--- Cell {i} (Potential Duplicate Setup) ---")
        print(source_str[:200] + "...")

    # Check for Indentation Error Cell 1
    if "ExternalTool Encoding Attack Strategies" in source_str:
        print(f"\n--- Cell {i} (Encoding Attack Strategies) ---")
        print(source_str)
        
    # Check for Indentation Error Cell 2
    if "Phase 9: Prompt Injection Testing" in source_str:
        print(f"\n--- Cell {i} (Prompt Injection Testing) ---")
        print(source_str)
