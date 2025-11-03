"""
Practical Guide: Using the Recon Module for Red Teaming

This guide shows exactly how to use the Reconnaissance module in real red team scenarios.

STEP 1: Setup and Initialization
---------------------------------
First, you need to initialize the logger and the recon module:

    from utils.logger import FrameworkLogger
    from core.modules.recon import ReconModule
    
    # Create a logger instance
    logger = FrameworkLogger("recon_engagement")
    
    # Initialize the recon module
    recon = ReconModule(logger)

STEP 2: Perform Initial Reconnaissance
---------------------------------------
Execute reconnaissance to gather information about your target:

    # Run reconnaissance
    recon_results = recon.perform_recon()
    
    # Check the status
    print(f"Status: {recon_results['status']}")
    print(f"Timestamp: {recon_results['timestamp']}")

STEP 3: Analyze Collected Data
------------------------------
Extract and analyze the different types of information gathered:

    # System Information
    system_info = recon_results["info_gathered"]["system_info"]
    print(f"Platform: {system_info['platform']}")
    print(f"Architecture: {system_info['architecture']}")
    print(f"Version: {system_info['version']}")
    
    # Network Information
    network_info = recon_results["info_gathered"]["network_info"]
    print(f"Open ports: {network_info['open_ports']}")
    print(f"Active connections: {network_info['active_connections']}")
    
    # Service Detection
    services = recon_results["info_gathered"]["service_detection"]
    for service in services:
        print(f"Service: {service['service']} on port {service['port']}")
        print(f"  Status: {service['status']}")
        print(f"  Version: {service['version']}")
    
    # Vulnerabilities
    vulnerabilities = recon_results["info_gathered"]["vulnerability_scan"]
    for vuln in vulnerabilities:
        print(f"Vulnerability: {vuln['type']}")
        print(f"  Severity: {vuln['severity']}")
        print(f"  Description: {vuln['description']}")
    
    # User Enumeration
    users = recon_results["info_gathered"]["user_enumeration"]
    print(f"Users found: {len(users)}")

STEP 4: Store and Retrieve Data Later
--------------------------------------
The module stores reconnaissance data for later retrieval:

    # Get stored recon data
    stored_data = recon.get_recon_data()
    
    # Use it for planning next steps
    if stored_data:
        services = stored_data["info_gathered"]["service_detection"]
        vulnerabilities = stored_data["info_gathered"]["vulnerability_scan"]
        
        # Plan exploits based on discovered services
        if any(s['service'] == 'http' for s in services):
            print("HTTP service detected - potential web app target")
        
        # Prioritize vulnerabilities
        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
        print(f"Critical vulnerabilities: {len(critical_vulns)}")

STEP 5: Real-World Example Workflow
------------------------------------
Here's a complete example of how you'd use it in an engagement:

    # 1. Initialize
    logger = FrameworkLogger("engagement_2024")
    recon = ReconModule(logger)
    
    # 2. Perform recon
    print("[*] Starting reconnaissance phase...")
    results = recon.perform_recon()
    
    # 3. Analyze findings
    print("\n[*] Analyzing reconnaissance results...")
    
    # Check for web services
    web_services = [s for s in results["info_gathered"]["service_detection"] 
                   if s['service'] in ['http', 'https']]
    if web_services:
        print(f"[+] Found {len(web_services)} web service(s)")
        for ws in web_services:
            print(f"    - {ws['service']} on port {ws['port']}")
    
    # Check for vulnerabilities
    vulns = results["info_gathered"]["vulnerability_scan"]
    if vulns:
        print(f"[+] Found {len(vulns)} potential vulnerability/vulnerabilities")
        for vuln in vulns:
            print(f"    - {vuln['severity'].upper()}: {vuln['description']}")
    
    # 4. Store for attack chain
    recon_data = recon.get_recon_data()
    
    # 5. Use findings to inform next steps
    print("\n[*] Reconnaissance complete. Findings stored for attack planning.")

INTEGRATION WITH ATTACK CHAIN
------------------------------
The recon module is typically used as the first stage in an attack chain:

    from core.modules.attack_chain import AttackChain, AttackStage
    
    chain = AttackChain(logger)
    
    # Recon runs automatically as first stage
    chain.execute_stage(AttackStage.INITIAL_RECON)
    
    # Then use the recon data for planning exploits
    recon_data = chain.recon.get_recon_data()
    
    # Based on recon findings, execute appropriate exploits
    if recon_data["info_gathered"]["service_detection"]:
        # Execute exploit against discovered services
        chain.execute_stage(AttackStage.CREDENTIAL_HARVEST)

KEY POINTS:
----------
1. Recon is ALWAYS the first step - gather intelligence before attacking
2. Store results for later use in attack planning
3. Analyze findings to prioritize targets
4. Use discovered services/vulnerabilities to guide exploit selection
5. Integrate with other modules based on recon findings
"""

