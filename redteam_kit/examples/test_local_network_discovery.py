#!/usr/bin/env python3
"""
Minimal Local Network Discovery Test
Tests basic network discovery functionality with minimal resource usage
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery

def main():
    print("="*70)
    print("LOCAL NETWORK DISCOVERY TEST")
    print("="*70)
    
    # Initialize logger
    logger = FrameworkLogger("network_discovery_test")
    
    # Create network discovery instance
    discovery = NetworkDiscovery(logger)
    
    # Test 1: Auto-detect network range
    print("\n[1/3] Testing network range auto-detection...")
    network_range = discovery._get_local_network_range()
    
    if not network_range:
        print("[!] Failed to detect network range")
        return
    
    print(f"[+] Detected network: {network_range}")
    
    # Test 2: Test ping on localhost
    print("\n[2/3] Testing ping functionality...")
    localhost_alive = discovery._ping_host("127.0.0.1")
    print(f"[+] Localhost (127.0.0.1) is {'alive' if localhost_alive else 'not responding'}")
    
    # Test 3: Scan only 5 IPs as a test
    print("\n[3/3] Testing host discovery on first 5 IPs...")
    print("      This will scan .1 through .5\n")
    
    # Extract network prefix
    network_prefix = '.'.join(network_range.split('.')[:-1])
    
    # Test just a few IPs
    test_ips = [f"{network_prefix}.{i}" for i in range(1, 6)]  # .1 to .5
    
    discovered = []
    for ip in test_ips:
        print(f"  Scanning {ip}...", end=" ", flush=True)
        if discovery._ping_host(ip):
            hostname = discovery._get_hostname(ip)
            discovered.append({
                "ip": ip,
                "hostname": hostname,
                "status": "alive"
            })
            print(f"✓ Found ({hostname})")
        else:
            print("✗ No response")
        
        # Small delay between tests
        import time
        time.sleep(0.5)
    
    print(f"\n[+] Discovery test complete!")
    print(f"[+] Found {len(discovered)} active host(s) in test range")
    
    if discovered:
        print("\n" + "="*70)
        print("DISCOVERED HOSTS")
        print("="*70)
        for host in discovered:
            print(f"  {host['ip']:<18} {host['hostname']}")
        print("="*70)
        print("\n[*] To scan full network, use:")
        print(f"    discovery.discover_network('{network_range}', stealth_mode=True)")
    else:
        print("\n[!] No hosts discovered in test range")
        print("[!] This could mean:")
        print("    - Firewall blocking ICMP")
        print("    - No hosts at those IPs")
        print("    - Network configuration issue")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
