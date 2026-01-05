#!/usr/bin/env python3
"""
Helper script to prepare notebook for Colab connection
Note: Actual connection requires UI interaction in Cursor, but this prepares everything
"""

import json
import sys
from pathlib import Path

def prepare_notebook_for_colab(notebook_path):
    """Prepare notebook metadata for Colab connection"""
    
    print("Preparing notebook for Colab connection...")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Check current kernel spec
    current_kernel = notebook.get('metadata', {}).get('kernelspec', {})
    print(f"Current kernel: {current_kernel.get('name', 'Not set')}")
    
    # Update metadata to be Colab-compatible
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    # Set Colab-compatible kernel (Colab extension will override this)
    notebook['metadata']['kernelspec'] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    
    # Add Colab metadata
    if 'colab' not in notebook['metadata']:
        notebook['metadata']['colab'] = {
            "provenance": [],
            "toc_visible": True
        }
    
    # Save updated notebook
    output_path = notebook_path.replace('.ipynb', '_colab_ready.ipynb')
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Notebook prepared: {output_path}")
    print(f"  Total cells: {len(notebook['cells'])}")
    print("\nNext steps:")
    print("1. Open this notebook in Cursor")
    print("2. Click 'Select Kernel' (top right)")
    print("3. Choose 'Colab' → 'New Colab Server'")
    print("4. Sign in with Google when prompted")
    print("5. Start running cells!")
    
    return output_path

if __name__ == "__main__":
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "latent_space_redteaming_complete.ipynb"
    
    if not Path(notebook_path).exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    prepare_notebook_for_colab(notebook_path)
