#!/usr/bin/env python3
"""
Safe Local Network Test with Credential Harvesting
Performs comprehensive red team testing including credential harvesting
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery
from core.modules.target_selector import TargetSelector
from core.modules.report_generator import RedTeamReportGenerator
from core.modules.post_exploit import PostExploitation

def main():
    print("="*70)
    print("SAFE LOCAL NETWORK TEST WITH CREDENTIAL HARVESTING")
    print("="*70)
    print("\n[*] Safe operations only - no disruption\n")
    
    logger = FrameworkLogger("safe_engagement")
    generator = RedTeamReportGenerator(logger)
    
    start_time = time.time()
    
    # Step 1: Network Discovery
    print("[1/3] Network Discovery...")
    discovery = NetworkDiscovery(logger)
    network_range = discovery._get_local_network_range()
    
    if not network_range:
        print("[!] Failed to detect network range")
        return
    
    print(f"[+] Network: {network_range}")
    network_prefix = '.'.join(network_range.split('.')[:-1])
    test_range = f"{network_prefix}.0/28"
    
    print(f"[+] Scanning {test_range}...")
    discovery_results = discovery.discover_network(test_range, stealth_mode=True)
    generator.add_module_results("network_discovery", discovery_results)
    
    hosts = discovery_results.get("hosts", [])
    print(f"[+] Found {len(hosts)} host(s)\n")
    
    if not hosts:
        print("[!] No hosts discovered")
        return
    
    generator.set_engagement_metadata(targets=[h['ip'] for h in hosts], start_time=start_time)
    
    # Step 2: Credential Harvesting
    print("[2/3] Credential Harvesting...")
    print("      Searching for credentials in accessible locations...\n")
    
    post_exploit = PostExploitation(logger)
    cred_results = post_exploit.harvest_credentials()
    generator.add_module_results("harvest_credentials", cred_results)
    
    cred_count = cred_results.get("count", 0)
    cred_sources = cred_results.get("sources", [])
    
    print(f"[+] Credentials found: {cred_count}")
    
    if cred_count > 0:
        print(f"[+] Sources: {', '.join(cred_sources)}")
        
        cred_types = {}
        for cred in cred_results.get("credentials_found", []):
            cred_type = cred.get("type", "unknown")
            cred_types[cred_type] = cred_types.get(cred_type, 0) + 1
        
        print(f"[+] Types: {', '.join(cred_types.keys())}")
        
        # Add finding
        generator.add_finding(
            "High",
            "Credentials Discovered",
            f"Harvested {cred_count} credential(s) from {len(cred_sources)} source(s)",
            [
                f"Sources: {', '.join(cred_sources)}",
                f"Credential types: {', '.join(cred_types.keys())}"
            ],
            "Rotate all discovered credentials immediately"
        )
    
    print()
    
    # Step 3: Generate Report
    print("[3/3] Generating Report...\n")
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    md_path = output_dir / f"safe_engagement_report_{timestamp}.md"
    json_path = output_dir / f"safe_engagement_report_{timestamp}.json"
    
    generator.generate_markdown_report(str(md_path))
    generator.generate_json_report(str(json_path))
    
    summary = generator.generate_summary()
    
    print("="*70)
    print("ENGAGEMENT SUMMARY")
    print("="*70)
    print(f"Targets: {summary['targets']}")
    print(f"Duration: {summary['duration_formatted']}")
    print(f"Modules: {summary['modules_executed']}")
    print(f"Findings: {summary['total_findings']}")
    print("\nFindings by Severity:")
    for severity, count in summary['findings_by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    
    credentials = generator.extract_credentials_from_results()
    if credentials:
        print(f"\n🔐 CREDENTIALS DISCOVERED: {len(credentials)}")
    
    print("\n" + "="*70)
    print(f"[+] Report: {md_path}")
    print(f"[+] JSON: {json_path}")
    print("[+] Test completed safely")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

