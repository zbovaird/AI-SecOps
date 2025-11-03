# Firefox Password Decryption Instructions

## Installation

Due to resource limits in this environment, firefox-decrypt must be installed manually:

```bash
pip3 install firefox-decrypt
```

Or if using a virtual environment:

```bash
source venv/bin/activate  # or: venv\Scripts\activate.bat (Windows)
pip install firefox-decrypt
```

## Decryption

Once installed, passwords will be automatically decrypted when running credential harvesting.

To decrypt manually:

```bash
python3 -m firefox_decrypt ~/Library/Application\ Support/Firefox/Profiles/yxo4pq2t.default-release
```

## Integration

The `post_exploit.py` module will automatically:
1. Check if firefox-decrypt is installed
2. Decrypt passwords if available
3. Fall back to showing encrypted fields if not installed

## Current Status

✅ **238 Firefox passwords found** (encrypted)
✅ **key4.db file exists** (decryption key available)
❌ **firefox-decrypt not installed** (needs manual installation)

Once firefox-decrypt is installed, passwords will be automatically decrypted during credential harvesting.

