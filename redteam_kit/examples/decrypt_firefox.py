#!/usr/bin/env python3
"""
Firefox Password Decryptor
Uses firefox_decrypt library or direct NSS decryption
FOR AUTHORIZED SECURITY TESTING ONLY
"""

import sys
import os
import json

def decrypt_firefox_passwords(profile_path):
    """Decrypt Firefox passwords from profile"""
    
    logins_file = os.path.join(profile_path, "logins.json")
    key4_db = os.path.join(profile_path, "key4.db")
    
    if not os.path.exists(logins_file):
        print(f"ERROR: logins.json not found at {logins_file}", file=sys.stderr)
        return {}
    
    if not os.path.exists(key4_db):
        print(f"ERROR: key4.db not found at {key4_db}", file=sys.stderr)
        return {}
    
    decrypted = {}
    
    # Method 1: Try firefox_decrypt library
    try:
        from firefox_decrypt import decrypt_passwords
        passwords = decrypt_passwords(profile_path)
        for entry in passwords:
            url = entry.get("url", "")
            username = entry.get("username", "")
            password = entry.get("password", "")
            if url:
                decrypted[url] = {"username": username, "password": password}
        return decrypted
    except ImportError:
        pass
    
    # Method 2: Try using dbus (Linux) or security command (macOS)
    try:
        import subprocess
        
        # Try firefox_decrypt command-line tool
        result = subprocess.run(
            ["firefox_decrypt", profile_path],
            capture_output=True,
            text=True,
            timeout=30,
            stderr=subprocess.DEVNULL
        )
        
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        url = parts[0].strip()
                        username = parts[1].strip()
                        password = parts[2].strip()
                        decrypted[url] = {"username": username, "password": password}
            return decrypted
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Method 3: Try manual decryption using NSS/Python
    try:
        # Read logins.json
        with open(logins_file, 'r') as f:
            data = json.load(f)
        
        if 'logins' in data:
            # Try to use PyNSS or similar
            try:
                import nss
                # This requires NSS library bindings
                # Placeholder for actual implementation
                pass
            except ImportError:
                # Fallback: Extract encrypted strings (can't decrypt without NSS)
                print("WARNING: Passwords are encrypted and require NSS library for decryption", file=sys.stderr)
                print("Install firefox_decrypt: pip install firefox-decrypt", file=sys.stderr)
                
                # Return encrypted passwords with note
                for login in data['logins']:
                    url = login.get('hostname', '')
                    if url:
                        decrypted[url] = {
                            "username": login.get('username', ''),
                            "password": "ENCRYPTED - requires decryption",
                            "encrypted_password": login.get('password', '')
                        }
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
    
    return decrypted


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 decrypt_firefox.py <firefox_profile_path>", file=sys.stderr)
        sys.exit(1)
    
    profile_path = sys.argv[1]
    passwords = decrypt_firefox_passwords(profile_path)
    
    # Output as JSON
    print(json.dumps(passwords, indent=2))

