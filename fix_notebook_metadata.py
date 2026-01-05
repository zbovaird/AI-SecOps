
import json
import os

target_file = 'latent_space_redteaming_complete3.ipynb'

def fix_metadata(filepath):
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    # Clean metadata
    print("Sanitizing metadata...")
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    # Force a generic Colab-friendly kernelspec
    notebook['metadata']['kernelspec'] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    
    # Ensure language info is correct
    notebook['metadata']['language_info'] = {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.5"
    }

    # Add specific Colab metadata that might help the extension identify it
    if 'colab' not in notebook['metadata']:
        notebook['metadata']['colab'] = {
            "provenance": []
        }
        
    # Remove potentially conflicting 'accelerator' keys if they exist at top level
    # (Sometimes these get stuck from previous runs)
    if 'accelerator' in notebook['metadata']:
        print("Removing specific accelerator metadata to force re-selection...")
        del notebook['metadata']['accelerator']

    output_file = filepath 
    print(f"Writing fixed metadata to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    if os.path.exists(target_file):
        fix_metadata(target_file)
    else:
        print(f"File not found: {target_file}")
