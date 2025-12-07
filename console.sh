#!/bin/bash
# Launch the AI SecOps Red Team Console

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python redteam_kit/console.py
else
    echo "Virtual environment not found. Please run ./install_tools.sh first."
    # Fallback to trying python3 directly if venv is missing
    python3 redteam_kit/console.py
fi
