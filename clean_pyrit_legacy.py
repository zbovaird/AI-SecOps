
import json
import re

target_file = 'latent_space_redteaming_complete3.ipynb'

def clean_notebook(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, 'r') as f:
        notebook = json.load(f)
    
    new_cells = []
    removed_count = 0
    
    for cell in notebook['cells']:
        source = "".join(cell.get('source', []))
        
        # Criteria 1: Remove "PyRIT Encoding Attack Strategies" cell entirely
        if "PyRIT Encoding Attack Strategies" in source:
            print("Removing Cell: PyRIT Encoding Attack Strategies")
            removed_count += 1
            continue
            
        # Criteria 2: Modify "Prompt Injection Testing" cell to remove multi-turn logic
        if "Phase 9: Prompt Injection Testing" in source:
            print("Cleaning Cell: Prompt Injection Testing")
            # We filter out the specific lines about multi-turn PyRit
            lines = cell['source']
            new_source = []
            skip_block = False
            for line in lines:
                if "# Try to use PyRIT MultiTurnOrchestrator" in line:
                    skip_block = True
                    new_source.append("# [PyRIT Integration Removed] See pyrit_redteaming.ipynb for multi-turn attacks.\n")
                    new_source.append("MULTI_TURN_AVAILABLE = False\n")
                    continue
                
                if skip_block:
                    # simplistic check to end the block? 
                    # The block is a try/except or if statement. 
                    # Let's just strip lines that contain "PyRIT" specifically inside this cell
                    pass
                
                if "from redteam_kit.core.modules.pyrit_integration" in line:
                    continue # Skip import
                    
                if "# Test multi-turn attacks if PyRIT is available" in line:
                     # Skip the later block too
                     continue
                     
                new_source.append(line)
            
            cell['source'] = new_source
            new_cells.append(cell)
            continue

        # Criteria 3: Modify "Adaptive Perturbation Testing" 
        if "Adaptive Perturbation Testing" in source:
            print("Cleaning Cell: Adaptive Perturbation Testing")
            lines = cell['source']
            new_source = []
            for line in lines:
                if "from redteam_kit.core.modules.pyrit_integration" in line:
                    new_source.append("    # PyRIT integration removed in this notebook\n")
                    continue
                if "pyrit_enabled=False" in line:
                    new_source.append(line) # Keep this as is
                    continue
                if "MULTI_TURN_AVAILABLE =" in line:
                    new_source.append("    MULTI_TURN_AVAILABLE = False\n")
                    continue
                
                new_source.append(line)
            cell['source'] = new_source
            new_cells.append(cell)
            continue
            
        new_cells.append(cell)
    
    notebook['cells'] = new_cells
    print(f"Removed {removed_count} cells and cleaned others.")
    
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Done.")

if __name__ == "__main__":
    clean_notebook(target_file)
