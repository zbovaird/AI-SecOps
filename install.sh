#!/bin/bash
# Cross-platform installation script for AI SecOps

set -e

echo "======================================================================"
echo "AI SecOps Red Team Kit - Installation"
echo "======================================================================"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Windows;;
    MINGW*)     MACHINE=Windows;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "[+] Detected OS: $MACHINE"
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 is required but not installed."
    echo "    Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[+] Python version: $(python3 --version)"
echo ""

# Create virtual environment
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[+] Virtual environment created"
else
    echo "[+] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
if [ "$MACHINE" = "Windows" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "[+] Virtual environment activated"
echo ""

# Upgrade pip
echo "[4/4] Installing dependencies..."
pip install --upgrade pip setuptools wheel
echo ""

# Install package
echo "[+] Installing AI SecOps Red Team Kit..."
pip install -e .

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "To activate the virtual environment:"
if [ "$MACHINE" = "Windows" ]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "To run tests:"
echo "  cd redteam_kit && bash examples/safe_test_with_creds.sh"
echo ""
echo "For more information, see README.md"
echo "======================================================================"

