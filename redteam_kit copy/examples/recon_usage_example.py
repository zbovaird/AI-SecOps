#!/usr/bin/env python3
"""
Recon Module - Practical Usage Example
Demonstrates exactly how to use the Recon module in a red team engagement
"""

from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

def main():
    print("=" * 70)
    print("REAL-WORLD RECON MODULE USAGE EXAMPLE")
    print("=" * 70)
    print()
    
    # Step 1: Initialize
    print("[1] Initializing Reconnaissance Module...")
    logger = FrameworkLogger("recon_demo")
    recon = ReconModule(logger)
    print("    ✓ Logger created")
    print("    ✓ Recon module initialized")
    print()
    
    # Step 2: Perform Reconnaissance
    print("[2] Performing Reconnaissance...")
    recon_results = recon.perform_recon()
    print(f"    ✓ Status: {recon_results['status']}")
    print()
    
    # Step 3: Analyze Results
    print("[3] Analyzing Collected Intelligence...")
    print()
    
    system_info = recon_results["info_gathered"]["system_info"]
    print(f"    System Information:")
    print(f"      Platform: {system_info['platform']}")
    print(f"      Architecture: {system_info['architecture']}")
    print()
    
    services = recon_results["info_gathered"]["service_detection"]
    print(f"    Services Detected:")
    for service in services:
        print(f"      • {service['service'].upper()} on port {service['port']}")
    print()
    
    vulnerabilities = recon_results["info_gathered"]["vulnerability_scan"]
    print(f"    Vulnerabilities Found:")
    for vuln in vulnerabilities:
        print(f"      • [{vuln['severity'].upper()}] {vuln['description']}")
    print()
    
    # Step 4: Retrieve Stored Data
    print("[4] Retrieving Stored Reconnaissance Data...")
    stored_data = recon.get_recon_data()
    if stored_data:
        print("    ✓ Data successfully retrieved")
    print()
    
    # Step 5: Use Findings
    print("[5] Using Findings for Attack Planning...")
    if services:
        print("    [+] Services detected - potential attack vectors identified")
    if vulnerabilities:
        print("    [+] Vulnerabilities identified - prioritize exploitation")
    print()
    
    print("=" * 70)
    print("RECONNAISSANCE COMPLETE - Ready for next phase")
    print("=" * 70)

if __name__ == "__main__":
    main()

