#!/usr/bin/env python3
"""
Network Discovery and Target Selection Example
Demonstrates how to discover network hosts and select targets for attack chain
"""

from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery
from core.modules.target_selector import TargetSelector
from core.modules.attack_chain import AttackChain

def main():
    logger = FrameworkLogger("network_discovery_demo")
    
    print("="*70)
    print("Network Discovery and Target Selection Demo")
    print("="*70)
    
    # Step 1: Auto-detect and discover network hosts (stealthy)
    print("\n[1/3] Discovering network hosts (stealthy mode)...")
    discovery = NetworkDiscovery(logger)
    network_results = discovery.discover_local_network(stealth_mode=True)
    
    if network_results.get("error"):
        print(f"[!] Error: {network_results['error']}")
        return
    
    if not network_results.get("hosts"):
        print("[!] No hosts discovered on the network")
        return
    
    print(f"[+] Discovered {len(network_results['hosts'])} hosts")
    
    # Step 2: Display and select targets
    print("\n[2/3] Selecting targets...")
    selector = TargetSelector(logger)
    selected_ips = selector.interactive_select(network_results['hosts'])
    
    if not selected_ips:
        print("[!] No targets selected")
        return
    
    # Step 3: Execute attack chain sequentially (stealthy)
    print(f"\n[3/3] Executing attack chain on {len(selected_ips)} target(s)...")
    print("Note: Using sequential execution (stealthy mode)")
    
    chain = AttackChain(logger)
    results = chain.execute_on_targets(
        targets=selected_ips,
        profile="recon_only",  # Use recon_only profile for demo
        sequential=True  # Sequential is more stealthy
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"Total targets: {results['total_targets']}")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print("\nResults per target:")
    for target_ip, target_results in results['results'].items():
        status = target_results.get('overall_status', 'unknown')
        print(f"  {target_ip}: {status}")
    print("="*70)

if __name__ == "__main__":
    main()

