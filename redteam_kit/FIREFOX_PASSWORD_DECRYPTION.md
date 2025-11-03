# Firefox Password Decryption Guide

## Where Are the Passwords?

Firefox stores passwords in:
- **File**: `~/Library/Application Support/Firefox/Profiles/[profile]/logins.json`
- **Encryption Key**: `~/Library/Application Support/Firefox/Profiles/[profile]/key4.db`

## Current Status

✅ **238 Firefox passwords found** in `logins.json`
🔒 **Passwords are ENCRYPTED** - require decryption
✅ **key4.db file exists** - decryption key available

## How to Decrypt Firefox Passwords

### Option 1: Using firefox-decrypt (Recommended)

```bash
# Install firefox-decrypt
pip3 install firefox-decrypt

# Decrypt passwords
firefox-decrypt ~/Library/Application\ Support/Firefox/Profiles/yxo4pq2t.default-release

# Or use Python directly
python3 -m firefox_decrypt ~/Library/Application\ Support/Firefox/Profiles/yxo4pq2t.default-release
```

### Option 2: Using Firefox's Built-in Password Manager

1. Open Firefox
2. Click menu → Passwords (or Settings → Privacy & Security → Logins and Passwords)
3. Click eye icon to reveal passwords
4. Copy passwords manually

### Option 3: Using Python Script (if firefox-decrypt installed)

```python
from firefox_decrypt import decrypt_passwords

profile_path = "~/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release"
passwords = decrypt_passwords(profile_path)

for entry in passwords:
    print(f"URL: {entry['url']}")
    print(f"Username: {entry['username']}")
    print(f"Password: {entry['password']}")
    print()
```

## Why Are They Encrypted?

Firefox encrypts passwords using NSS (Network Security Services) for security:
- Passwords are encrypted with AES-256
- Encryption key is stored in `key4.db`
- On macOS, the master key may be stored in Keychain
- Requires `logins.json` + `key4.db` + (optional) master password

## What We Can Extract Without Decryption

Currently, our script can extract:
- ✅ URLs/domains (238 found)
- ✅ Usernames (if available)
- ❌ Actual passwords (encrypted - require decryption)

## Next Steps

To get actual passwords, you need to:
1. Install `firefox-decrypt`: `pip3 install firefox-decrypt`
2. Run decryption against the profile
3. The decrypted passwords will then be available

Note: The passwords ARE there, they're just encrypted for security!

