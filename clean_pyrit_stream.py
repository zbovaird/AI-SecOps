
import json
import sys

input_file = 'latent_space_redteaming_complete3.ipynb'
output_file = 'latent_space_redteaming_complete3_cleaned.ipynb'

def clean_stream():
    try:
        with open(input_file, 'r') as f:
            notebook = json.load(f)
        
        cleaned_cells = []
        
        for cell in notebook['cells']:
            # Convert source to string for easy checking
            source_text = "".join(cell.get('source', [])).lower()
            
            # 1. Skip if it's the "PyRIT Encoding Attack Strategies" cell
            if "pyrit encoding attack strategies" in source_text:
                continue
                
            # 2. Skip if it's the Prompt Injection cell (which has heavy PyRIT logic)
            # We will replace it with a simplified version if needed, or just skip the PyRit parts?
            # The user wants "no Pyrit references".
            # If the cell is heavily PyRit, we drop it.
            if "pyrit_integration" in source_text or "pyritorchestrator" in source_text:
                # This is a code cell using PyRit. 
                # We want to keep the non-PyRit parts if it's mixed (like Prompt Injection Phase 9).
                
                # Let's filter the lines inside the cell
                new_source = []
                lines = cell.get('source', [])
                for line in lines:
                    line_lower = line.lower()
                    if "pyrit" in line_lower:
                        # Replace with comment
                        new_source.append("# [PyRIT Logic Removed]\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                cleaned_cells.append(cell)
            else:
                # 3. Also check for pip install pyrit
                new_source = []
                lines = cell.get('source', [])
                for line in lines:
                    if "pip install" in line and "pyrit" in line:
                         # Remove pyrit from pip install line
                         new_line = line.replace(" pyrit", "").replace("pyrit ", "")
                         new_source.append(new_line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                cleaned_cells.append(cell)

        notebook['cells'] = cleaned_cells
        
        with open(output_file, 'w') as f:
            json.dump(notebook, f, indent=1)
            
        print("Success")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_stream()
