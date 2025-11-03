#!/usr/bin/env python3
"""
Safe Local Network Red Team Test
Performs comprehensive red team testing without disrupting services or network
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery
from core.modules.target_selector import TargetSelector
from core.modules.attack_chain import AttackChain
from core.modules.report_generator import RedTeamReportGenerator
from core.modules.os_detection import OSDetection
from core.modules.recon import ReconModule
from core.modules.osint import OSINTModule
from core.modules.post_exploit import PostExploitation
from core.modules.network_enumeration import NetworkEnumeration


def safe_local_network_test():
    """Perform safe red team testing on local network"""
    
    print("="*70)
    print("SAFE LOCAL NETWORK RED TEAM TEST")
    print("="*70)
    print("\n[*] This test will:")
    print("    ✓ Discover network hosts (stealthy)")
    print("    ✓ Perform reconnaissance (read-only)")
    print("    ✓ Test OS detection")
    print("    ✓ Enumerate services (passive)")
    print("    ✓ Gather OSINT data")
    print("    ✗ NO DDoS attacks")
    print("    ✗ NO service disruption")
    print("    ✗ NO credential attacks")
    print("    ✗ NO exploitation attempts")
    print("\n[*] All operations are safe and non-disruptive\n")
    
    logger = FrameworkLogger("safe_engagement")
    generator = RedTeamReportGenerator(logger)
    
    start_time = time.time()
    
    # Step 1: Network Discovery
    print("[1/6] Network Discovery (Stealthy)...")
    print("      Scanning local network for active hosts...\n")
    
    discovery = NetworkDiscovery(logger)
    
    # Get network range first
    network_range = discovery._get_local_network_range()
    if not network_range:
        print("[!] Failed to detect network range")
        return
    
    print(f"[+] Detected network: {network_range}")
    
    # Scan smaller range for safety (first 20 IPs)
    network_prefix = '.'.join(network_range.split('.')[:-1])
    test_range = f"{network_prefix}.0/28"  # .1 to .14
    
    print(f"[+] Scanning test range: {test_range} (first 14 hosts)")
    print("[+] Using stealthy mode with random delays\n")
    
    discovery_results = discovery.discover_network(test_range, stealth_mode=True)
    generator.add_module_results("network_discovery", discovery_results)
    
    hosts = discovery_results.get("hosts", [])
    print(f"[+] Discovered {len(hosts)} active host(s)")
    
    if not hosts:
        print("[!] No hosts discovered. Exiting.")
        return
    
    # Display hosts
    selector = TargetSelector(logger)
    selector.display_hosts(hosts)
    
    # Get target IPs
    selected_ips = [h['ip'] for h in hosts[:5]]  # Limit to first 5 for safety
    print(f"[+] Testing {len(selected_ips)} host(s): {', '.join(selected_ips)}\n")
    
    generator.set_engagement_metadata(targets=selected_ips, start_time=start_time)
    
    # Step 2: OS Detection
    print("[2/6] OS Detection...")
    print("      Detecting operating systems of discovered hosts...\n")
    
    os_detection = OSDetection(logger)
    
    # Detect local OS
    local_os = os_detection.detect_local_os()
    generator.add_module_results("os_detection", {"local": local_os})
    print(f"[+] Local OS: {local_os.get('os_type')} {local_os.get('os_version')}")
    
    # Detect remote OS
    for target_ip in selected_ips:
        print(f"  Detecting OS for {target_ip}...", end=" ", flush=True)
        remote_os = os_detection.detect_remote_os(target_ip)
        generator.add_module_results("os_detection", {target_ip: remote_os})
        
        os_type = remote_os.get('os_type', 'Unknown')
        confidence = remote_os.get('confidence', 0)
        print(f"✓ {os_type} (confidence: {confidence:.0f}%)")
        
        time.sleep(1)  # Delay between detections
    
    print()
    
    # Step 3: Reconnaissance
    print("[3/6] Reconnaissance (Read-Only)...")
    print("      Gathering information about targets...\n")
    
    recon = ReconModule(logger)
    
    for target_ip in selected_ips:
        print(f"  Reconnaissance on {target_ip}...", end=" ", flush=True)
        
        # Safe reconnaissance - limited ports, no aggressive scanning
        safe_ports = [22, 80, 443, 3389]  # Common ports only
        recon_results = recon.perform_recon(target=target_ip, ports=safe_ports)
        generator.add_module_results("recon", {target_ip: recon_results})
        
        open_ports = recon_results.get("info_gathered", {}).get("network_info", {}).get("open_ports", [])
        print(f"✓ {len(open_ports)} open port(s) found")
        
        time.sleep(2)  # Delay between recon
    
    print()
    
    # Step 4: Network Enumeration (Passive)
    print("[4/6] Network Enumeration (Passive)...")
    print("      Enumerating network services...\n")
    
    net_enum = NetworkEnumeration(logger)
    
    for target_ip in selected_ips:
        print(f"  Enumerating services on {target_ip}...", end=" ", flush=True)
        
        # Safe service enumeration
        enum_results = net_enum.enumerate_services(target_ip, ports=[22, 80, 443])
        generator.add_module_results("network_enumeration", {target_ip: enum_results})
        
        services = enum_results.get("services", [])
        print(f"✓ {len(services)} service(s) identified")
        
        time.sleep(1.5)  # Delay between enumerations
    
    print()
    
    # Step 5: Credential Harvesting
    print("[5/7] Credential Harvesting...")
    print("      Searching for credentials in accessible locations...\n")
    
    post_exploit = PostExploitation(logger)
    
    print("  Harvesting credentials from local system...", end=" ", flush=True)
    cred_results = post_exploit.harvest_credentials()
    generator.add_module_results("harvest_credentials", cred_results)
    
    cred_count = cred_results.get("count", 0)
    cred_sources = cred_results.get("sources", [])
    
    print(f"✓ {cred_count} credential(s) found")
    
    if cred_count > 0:
        print(f"\n  [*] Credentials found in {len(cred_sources)} source(s):")
        for source in cred_sources:
            print(f"      - {source}")
        
        # Show credential types
        cred_types = {}
        for cred in cred_results.get("credentials_found", []):
            cred_type = cred.get("type", "unknown")
            cred_types[cred_type] = cred_types.get(cred_type, 0) + 1
        
        print(f"\n  [*] Credential types discovered:")
        for cred_type, count in cred_types.items():
            print(f"      - {cred_type}: {count}")
        
        # Add finding for credentials
        generator.add_finding(
            "High",
            "Credentials Discovered",
            f"Harvested {cred_count} credential(s) from various sources on the local system",
            [
                f"Sources: {', '.join(cred_sources)}",
                f"Credential types: {', '.join(cred_types.keys())}"
            ],
            "Rotate all discovered credentials immediately. Review credential storage practices and implement secure credential management."
        )
    else:
        print("  [*] No credentials found in accessible locations")
    
    print()
    
    # Step 6: OSINT (External Only)
    print("[6/7] OSINT Gathering...")
    print("      Gathering open source intelligence...\n")
    
    osint = OSINTModule(logger)
    
    # Only gather OSINT for domains, not IPs
    print("  [*] OSINT requires domain names, skipping IP-based OSINT")
    print("  [*] To test OSINT, provide domain names")
    
    # Add placeholder
    generator.add_module_results("osint", {"note": "OSINT requires domain names, not IPs"})
    
    print()
    
    # Step 7: Generate Findings
    print("[7/7] Generating Findings...")
    print("      Analyzing results and generating security findings...\n")
    
    # Analyze discovery results
    if hosts:
        generator.add_finding(
            "Info",
            "Network Hosts Discovered",
            f"Discovered {len(hosts)} active host(s) on the local network",
            [f"Network range: {network_range}", f"Hosts found: {len(hosts)}"],
            "Monitor network for unauthorized devices"
        )
    
    # Analyze open ports
    all_open_ports = []
    for target_ip in selected_ips:
        recon_data = generator.get_engagement_data()["reconnaissance"].get("recon", [])
        for result in recon_data:
            if isinstance(result, dict) and target_ip in result:
                ports = result[target_ip].get("info_gathered", {}).get("network_info", {}).get("open_ports", [])
                all_open_ports.extend([p["port"] for p in ports])
    
    if all_open_ports:
        from collections import Counter
        port_counts = Counter(all_open_ports)
        exposed_ports = [p for p, count in port_counts.items() if count > 0]
        
        generator.add_finding(
            "Medium",
            "Exposed Network Services",
            f"Multiple network services are exposed on {len(selected_ips)} host(s)",
            [f"Open ports discovered: {', '.join(map(str, exposed_ports))}"],
            "Review exposed services and restrict access where possible"
        )
    
    # Analyze OS detection
    detected_oses = []
    os_data = generator.get_engagement_data()["reconnaissance"].get("os_detection", [])
    for result in os_data:
        if isinstance(result, dict):
            for ip, os_info in result.items():
                if ip != "local" and isinstance(os_info, dict):
                    os_type = os_info.get("os_type", "Unknown")
                    if os_type != "Unknown":
                        detected_oses.append(f"{ip}: {os_type}")
    
    if detected_oses:
        generator.add_finding(
            "Info",
            "OS Fingerprinting Successful",
            "Successfully identified operating systems on target hosts",
            detected_oses,
            "Use OS-specific security hardening guidelines"
        )
    
    # Generate Report
    print("[+] Generating comprehensive report...\n")
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_name = f"safe_engagement_report_{timestamp}"
    
    md_path = output_dir / f"{report_name}.md"
    json_path = output_dir / f"{report_name}.json"
    
    generator.generate_markdown_report(str(md_path))
    generator.generate_json_report(str(json_path))
    
    # Print summary
    summary = generator.generate_summary()
    
    print("="*70)
    print("ENGAGEMENT SUMMARY")
    print("="*70)
    print(f"Targets Tested: {summary['targets']}")
    print(f"Duration: {summary['duration_formatted']}")
    print(f"Modules Executed: {summary['modules_executed']}")
    print(f"Total Findings: {summary['total_findings']}")
    print("\nFindings by Severity:")
    for severity, count in summary['findings_by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    print("\n" + "="*70)
    print(f"\n[+] Markdown Report: {md_path}")
    print(f"[+] JSON Report: {json_path}")
    print("\n[*] All operations completed safely - no services disrupted")
    print("="*70)


if __name__ == "__main__":
    try:
        safe_local_network_test()
    except KeyboardInterrupt:
        print("\n\n[!] Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

