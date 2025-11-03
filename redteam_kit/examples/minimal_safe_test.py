#!/usr/bin/env python3
"""
Minimal Safe Network Test
Simplified version with minimal resource usage
"""

import socket
import subprocess
import platform
import time
import json
from pathlib import Path
from datetime import datetime

def ping_host(ip):
    """Ping a host safely"""
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

def scan_port(ip, port, timeout=0.5):
    """Scan a single port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    print("="*70)
    print("SAFE LOCAL NETWORK TEST (Minimal)")
    print("="*70)
    print("\n[*] Safe operations only - no disruption\n")
    
    # Get network info
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    network_prefix = '.'.join(local_ip.split('.')[:-1])
    
    print(f"[+] Hostname: {hostname}")
    print(f"[+] Local IP: {local_ip}")
    print(f"[+] Network: {network_prefix}.0/24")
    print(f"[+] Scanning first 10 IPs only\n")
    
    # Test localhost
    print("[1/4] Testing ping...")
    localhost_alive = ping_host("127.0.0.1")
    print(f"[+] Localhost: {'alive' if localhost_alive else 'not responding'}\n")
    
    # Discover hosts
    print("[2/4] Discovering hosts...")
    discovered = []
    
    for i in range(1, 11):  # First 10 IPs
        ip = f"{network_prefix}.{i}"
        print(f"  {ip}...", end=" ", flush=True)
        
        if ping_host(ip):
            hostname = get_hostname(ip)
            discovered.append({"ip": ip, "hostname": hostname})
            print(f"✓ Found ({hostname})")
        else:
            print("✗")
        
        time.sleep(0.5)  # Safe delay
    
    print(f"\n[+] Found {len(discovered)} active host(s)\n")
    
    # Scan ports (limited)
    print("[3/4] Scanning common ports (read-only)...")
    safe_ports = [22, 80, 443]
    
    for host in discovered:
        ip = host["ip"]
        print(f"  {ip}...", end=" ", flush=True)
        open_ports = []
        
        for port in safe_ports:
            if scan_port(ip, port, timeout=0.3):
                open_ports.append(port)
            time.sleep(0.2)  # Delay between ports
        
        host["open_ports"] = open_ports
        print(f"✓ {len(open_ports)} port(s) open")
    
    print()
    
    # Generate simple report
    print("[4/4] Generating report...")
    
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Markdown report
    md_content = []
    md_content.append("# Safe Local Network Test Report")
    md_content.append("")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    md_content.append("## Summary")
    md_content.append("")
    md_content.append(f"- **Network:** {network_prefix}.0/24")
    md_content.append(f"- **Hosts Discovered:** {len(discovered)}")
    md_content.append(f"- **Test Type:** Safe, non-disruptive")
    md_content.append("")
    md_content.append("## Discovered Hosts")
    md_content.append("")
    md_content.append("| IP Address | Hostname | Open Ports |")
    md_content.append("|------------|----------|------------|")
    
    for host in discovered:
        ports_str = ", ".join(map(str, host["open_ports"])) if host["open_ports"] else "none"
        md_content.append(f"| {host['ip']} | {host['hostname']} | {ports_str} |")
    
    md_content.append("")
    md_content.append("## Findings")
    md_content.append("")
    
    if discovered:
        md_content.append(f"**Info:** Discovered {len(discovered)} active host(s) on local network")
        md_content.append("")
    
    all_ports = []
    for host in discovered:
        all_ports.extend(host["open_ports"])
    
    if all_ports:
        from collections import Counter
        port_counts = Counter(all_ports)
        md_content.append("**Medium:** Exposed network services detected")
        md_content.append("")
        md_content.append("Open ports:")
        for port, count in sorted(port_counts.items()):
            md_content.append(f"- Port {port}: {count} host(s)")
        md_content.append("")
    
    md_content.append("---")
    md_content.append("*Safe test completed - no services disrupted*")
    
    md_path = report_dir / f"safe_test_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_content))
    
    # JSON report
    json_data = {
        "metadata": {
            "network": f"{network_prefix}.0/24",
            "local_ip": local_ip,
            "timestamp": timestamp,
            "test_type": "safe_non_disruptive"
        },
        "hosts": discovered,
        "summary": {
            "hosts_discovered": len(discovered),
            "total_open_ports": len(all_ports)
        }
    }
    
    json_path = report_dir / f"safe_test_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Display results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Hosts Discovered: {len(discovered)}")
    
    if discovered:
        print("\nDiscovered Hosts:")
        for host in discovered:
            ports_str = ", ".join(map(str, host["open_ports"])) if host["open_ports"] else "none"
            print(f"  {host['ip']:<18} {host['hostname']:<25} Ports: {ports_str}")
    
    print("\n" + "="*70)
    print(f"[+] Report saved: {md_path}")
    print(f"[+] JSON saved: {json_path}")
    print("[+] Test completed safely - no disruption")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

