#!/usr/bin/env python3
"""
Recon Module - Standalone Usage
How to use recon module from scratch for a new engagement
"""

from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

def standalone_recon_example():
    """
    Example: Using recon module standalone (without attack chain)
    Use this when you want to gather intelligence first, then decide what to do
    """
    print("=" * 70)
    print("STANDALONE RECON MODULE USAGE")
    print("=" * 70)
    print()
    
    # Step 1: Setup
    print("[*] Initializing reconnaissance module...")
    logger = FrameworkLogger("standalone_recon")
    recon = ReconModule(logger)
    
    # Step 2: Perform reconnaissance
    print("[*] Performing reconnaissance on target...")
    results = recon.perform_recon()
    
    # Step 3: Analyze what you found
    print("\n[*] Intelligence Analysis:")
    print("-" * 70)
    
    # Check system info
    system = results["info_gathered"]["system_info"]
    print(f"\n[+] System Information:")
    print(f"    Platform: {system['platform']}")
    print(f"    Architecture: {system['architecture']}")
    print(f"    Version: {system['version']}")
    
    # Check services
    services = results["info_gathered"]["service_detection"]
    print(f"\n[+] Services Detected ({len(services)}):")
    for service in services:
        print(f"    • {service['service'].upper()} on port {service['port']}")
        print(f"      Status: {service['status']}, Version: {service['version']}")
    
    # Check vulnerabilities
    vulns = results["info_gathered"]["vulnerability_scan"]
    print(f"\n[+] Vulnerabilities Found ({len(vulns)}):")
    for vuln in vulns:
        print(f"    • [{vuln['severity'].upper()}] {vuln['type']}")
        print(f"      {vuln['description']}")
    
    # Check users
    users = results["info_gathered"]["user_enumeration"]
    print(f"\n[+] Users Enumerated: {len(users)}")
    
    # Step 4: Make decisions based on findings
    print("\n[*] Attack Planning:")
    print("-" * 70)
    
    if services:
        print("\n[!] Recommended next steps based on services:")
        for service in services:
            if service['service'] == 'http':
                print("    → HTTP service: Test for web application vulnerabilities")
                print("      - SQL injection")
                print("      - XSS (Cross-Site Scripting)")
                print("      - Command injection")
            elif service['service'] == 'ssh':
                print("    → SSH service: Test for authentication weaknesses")
                print("      - Brute force attacks")
                print("      - Weak keys")
                print("      - Known vulnerabilities")
    
    if vulns:
        print("\n[!] Prioritize exploitation:")
        critical = [v for v in vulns if v['severity'] == 'critical']
        if critical:
            print(f"    → {len(critical)} CRITICAL vulnerabilities - exploit first!")
        high = [v for v in vulns if v['severity'] == 'high']
        if high:
            print(f"    → {len(high)} HIGH vulnerabilities - exploit next")
    
    # Step 5: Store results for later use
    print("\n[*] Storing reconnaissance data...")
    stored_data = recon.get_recon_data()
    print("    ✓ Data stored and ready for attack planning")
    
    return recon, results


def integrated_workflow_example():
    """
    Example: Using recon as part of attack chain workflow
    Use this when you want automated multi-stage engagement
    """
    print("\n\n" + "=" * 70)
    print("INTEGRATED ATTACK CHAIN WORKFLOW")
    print("=" * 70)
    print()
    
    from core.modules.attack_chain import AttackChain, AttackStage
    
    print("[*] Initializing attack chain (includes recon module)...")
    logger = FrameworkLogger("attack_chain")
    chain = AttackChain(logger)
    
    print("\n[*] Executing reconnaissance stage...")
    recon_result = chain.execute_stage(AttackStage.INITIAL_RECON)
    print(f"    Status: {recon_result['status']}")
    
    # Access recon data from chain
    recon_data = chain.recon.get_recon_data()
    services = recon_data["info_gathered"]["service_detection"]
    
    print(f"\n[+] Found {len(services)} service(s)")
    
    # Based on recon, execute next stages
    print("\n[*] Proceeding to credential harvesting...")
    chain.execute_stage(AttackStage.CREDENTIAL_HARVEST)
    
    print("\n[*] Proceeding to privilege escalation...")
    chain.execute_stage(AttackStage.PRIVILEGE_ESCALATION)
    
    print("\n[*] Attack chain execution complete")
    
    return chain


if __name__ == "__main__":
    # Run standalone example
    recon, results = standalone_recon_example()
    
    # Optionally run integrated example
    # chain = integrated_workflow_example()

