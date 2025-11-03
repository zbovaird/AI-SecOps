#!/bin/bash
# Installation script for PyRIT and IBM ART

set -e

echo "=========================================="
echo "Installing PyRIT and IBM ART"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyRIT
echo ""
echo "Installing Microsoft PyRIT..."
pip install pyrit

# Install IBM ART
echo ""
echo "Installing IBM Adversarial Robustness Toolbox..."
pip install adversarial-robustness-toolbox

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 -c "
try:
    import pyrit
    print('✓ PyRIT installed successfully')
    print(f'  Location: {pyrit.__file__}')
except ImportError as e:
    print('✗ PyRIT installation failed:', e)

try:
    import art
    print('✓ IBM ART installed successfully')
    print(f'  Version: {art.__version__}')
    print(f'  Location: {art.__file__}')
except ImportError as e:
    print('✗ IBM ART installation failed:', e)
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To explore capabilities, run:"
echo "  python explore_tools.py"

