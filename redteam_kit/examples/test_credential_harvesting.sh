#!/bin/bash
# Test credential harvesting on local machine
# FOR AUTHORIZED SECURITY TESTING ONLY

set -e

echo "======================================================================"
echo "Credential Harvesting Test - Local Machine"
echo "======================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")/.." || exit 1

echo "[*] Starting credential harvesting test..."
echo "[*] Using stealthy shell-based methods..."
echo ""

# Run Python test in a way that handles kills gracefully
python3 << 'PYTHON_SCRIPT'
import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redteam_kit.utils.logger import FrameworkLogger
from redteam_kit.core.modules.post_exploit import PostExploitation

try:
    print("[*] Initializing Post-Exploitation module...")
    logger = FrameworkLogger("credential_test")
    post_exploit = PostExploitation(logger)
    print("[+] Module initialized\n")
    
    print("[*] Harvesting credentials...")
    print("[*] This may take a moment (stealthy delays included)...\n")
    
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
            
            for i, cred in enumerate(cred_list[:10], 1):  # Limit to first 10 per type
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
            
            if len(cred_list) > 10:
                print(f"\n    ... and {len(cred_list) - 10} more")
        
        print()
        print("=" * 70)
        print("SUMMARY BY TYPE")
        print("=" * 70)
        for cred_type, cred_list in sorted(by_type.items()):
            print(f"  {cred_type}: {len(cred_list)}")
        
        # Save results
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
    else:
        print("[!] No credentials found")
        print()
        print("Note: This could mean:")
        print("  - No saved browser passwords")
        print("  - No credentials in config files")
        print("  - Credential managers are empty")
        print("  - Protected paths require permissions")
    
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n\n[!] Test interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n[!] Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

exit_code=$?

if [ $exit_code -eq 137 ] || [ $exit_code -eq 9 ]; then
    echo ""
    echo "[!] Process was killed (Killed: 9) - resource limits encountered"
    echo "[*] This is expected in some environments due to resource constraints"
    echo ""
    echo "[*] Trying alternative lightweight approach..."
    echo ""
    
    # Fallback: Just test individual components
    echo "[*] Testing individual credential sources..."
    echo ""
    
    # Test environment variables
    echo "[+] Checking environment variables..."
    env | grep -iE "(API|SECRET|PASSWORD|TOKEN|KEY)" | head -5 || echo "  No credentials in environment"
    echo ""
    
    # Test config files
    echo "[+] Checking config files..."
    for file in ~/.aws/credentials ~/.aws/config ~/.env 2>/dev/null; do
        if [ -f "$file" ] && [ -r "$file" ]; then
            echo "  Found: $file"
        fi
    done
    echo ""
    
    echo "[*] Partial test complete (resource limits prevented full test)"
fi

exit $exit_code

