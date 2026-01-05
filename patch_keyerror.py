
import json

target_file = 'latent_space_redteaming_complete3.ipynb'

def patch_notebook(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, 'r') as f:
        notebook = json.load(f)
    
    modified = False
    for cell in notebook['cells']:
        source = cell.get('source', [])
        new_source = []
        skip_next = False
        
        for i, line in enumerate(source):
            # Locate the key error section around line 2011 (in previous view)
            # The pattern is: for target_layer in target_layers:
            #                     if target_layer in pert['propagation_metrics']:
            
            if "for target_layer in target_layers:" in line:
                # Add check before this loop
                new_source.append("    if 'propagation_metrics' not in pert: continue # Safety check added by Agent\n")
                new_source.append(line)
            else:
                new_source.append(line)
                
        if len(new_source) != len(source):
            cell['source'] = new_source
            modified = True
            
    if modified:
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=1)
        print("Successfully patched KeyError vulnerability.")
    else:
        print("No changes likely made (pattern not found exactly?).")

if __name__ == "__main__":
    patch_notebook(target_file)
