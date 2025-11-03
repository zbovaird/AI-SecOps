#!/usr/bin/env python3
"""
Test credential harvesting on local machine
FOR AUTHORIZED SECURITY TESTING ONLY
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redteam_kit.utils.logger import FrameworkLogger
from redteam_kit.core.modules.post_exploit import PostExploitation


def main():
    print("=" * 70)
    print("Credential Harvesting Test - Local Machine")
    print("=" * 70)
    print()
    
    # Initialize logger
    logger = FrameworkLogger("credential_test")
    
    # Initialize post-exploitation module
    print("[*] Initializing Post-Exploitation module...")
    post_exploit = PostExploitation(logger)
    print("[+] Module initialized\n")
    
    # Harvest credentials
    print("[*] Starting credential harvesting...")
    print("[*] This may take a moment (stealthy delays included)...")
    print()
    
    results = post_exploit.harvest_credentials()
    
    print("=" * 70)
    print("CREDENTIAL HARVESTING RESULTS")
    print("=" * 70)
    print()
    
    print(f"Status: {results.get('status', 'unknown')}")
    print(f"Total Credentials Found: {results.get('count', 0)}")
    print(f"Sources: {', '.join(results.get('sources', []))}")
    print()
    
    if results.get('count', 0) > 0:
        print("=" * 70)
        print("DETAILED CREDENTIALS")
        print("=" * 70)
        print()
        
        creds = results.get('credentials_found', [])
        
        # Group by type
        by_type = {}
        for cred in creds:
            cred_type = cred.get('type', 'unknown')
            if cred_type not in by_type:
                by_type[cred_type] = []
            by_type[cred_type].append(cred)
        
        # Display by type
        for cred_type, cred_list in sorted(by_type.items()):
            print(f"\n[{cred_type.upper()}] ({len(cred_list)} found)")
            print("-" * 70)
            
            for i, cred in enumerate(cred_list, 1):
                print(f"\n  Credential {i}:")
                if cred.get('username'):
                    print(f"    Username: {cred['username']}")
                if cred.get('url'):
                    print(f"    URL: {cred['url']}")
                if cred.get('service'):
                    print(f"    Service: {cred['service']}")
                if cred.get('name'):
                    print(f"    Variable Name: {cred['name']}")
                if cred.get('value'):
                    print(f"    Value: {cred['value']}")
                if cred.get('source'):
                    print(f"    Source: {cred['source']}")
                if cred.get('location'):
                    print(f"    Location: {cred['location']}")
        
        print()
        print("=" * 70)
        print("SUMMARY BY TYPE")
        print("=" * 70)
        for cred_type, cred_list in sorted(by_type.items()):
            print(f"  {cred_type}: {len(cred_list)}")
    else:
        print("[!] No credentials found")
        print()
        print("Note: This could mean:")
        print("  - No saved browser passwords")
        print("  - No credentials in config files")
        print("  - Credential managers are empty")
        print("  - Protected paths require permissions")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"redteam_kit/reports/credential_test_{timestamp}.json"
    
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"[+] Results saved to: {results_file}")
    except Exception as e:
        print(f"[!] Could not save results: {e}")
    
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[!] Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

