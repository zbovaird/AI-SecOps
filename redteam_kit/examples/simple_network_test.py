#!/usr/bin/env python3
"""
Simple Network Discovery Test
Minimal resource usage test for network discovery
"""

import socket
import subprocess
import platform
import time

def ping_host(ip):
    """Ping a host"""
    try:
        if platform.system() == "Windows":
            cmd = f'ping -n 1 -w 1000 {ip}'
        else:
            cmd = f'ping -c 1 -W 1 {ip}'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2)
        return result.returncode == 0
    except:
        return False

def get_hostname(ip):
    """Get hostname from IP"""
    try:
        return socket.gethostbyaddr(ip)[0]
    except:
        return "unknown"

def main():
    print("="*70)
    print("LOCAL NETWORK DISCOVERY TEST")
    print("="*70)
    
    # Get local network info
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    network_prefix = '.'.join(local_ip.split('.')[:-1])
    
    print(f"\nHostname: {hostname}")
    print(f"Local IP: {local_ip}")
    print(f"Network range: {network_prefix}.0/24")
    
    # Test localhost
    print("\n[1/3] Testing ping functionality...")
    localhost_alive = ping_host("127.0.0.1")
    print(f"[+] Localhost is {'alive' if localhost_alive else 'not responding'}")
    
    # Scan first 10 IPs
    print(f"\n[2/3] Scanning first 10 IPs ({network_prefix}.1-10)...")
    discovered = []
    
    for i in range(1, 11):
        ip = f"{network_prefix}.{i}"
        print(f"  {ip}...", end=" ", flush=True)
        
        if ping_host(ip):
            hostname = get_hostname(ip)
            discovered.append({"ip": ip, "hostname": hostname})
            print(f"✓ Found ({hostname})")
        else:
            print("✗")
        
        time.sleep(0.3)  # Small delay
    
    # Results
    print(f"\n[3/3] Results:")
    print(f"[+] Found {len(discovered)} active host(s)")
    
    if discovered:
        print("\n" + "="*70)
        print("DISCOVERED HOSTS")
        print("="*70)
        print(f"{'IP Address':<18} {'Hostname':<30}")
        print("-"*70)
        for host in discovered:
            print(f"{host['ip']:<18} {host['hostname']:<30}")
        print("="*70)
    else:
        print("\n[!] No hosts discovered")
        print("[!] Possible reasons:")
        print("    - Firewall blocking ICMP")
        print("    - No hosts at those IPs")
        print("    - Network configuration")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

