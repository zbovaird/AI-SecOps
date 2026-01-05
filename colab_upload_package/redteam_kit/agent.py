#!/usr/bin/env python3
"""
Red Team Agent
--------------
An interactive agent to orchestrate attack chains and red team engagements.
Wraps the AttackChain functionality into a user-friendly CLI agent.
"""

import sys
import os
import argparse
from typing import List, Optional

# Ensure we can import from the current directory and parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

class RedTeamAgent:
    def __init__(self, target: Optional[str] = None, model_path: Optional[str] = None, verbose: bool = True):
        self.logger = FrameworkLogger("redteam_agent")
        self.target = target
        self.chain = AttackChain(self.logger, target=target, model_path=model_path)
        self.verbose = verbose

    def list_profiles(self):
        """List available attack profiles"""
        print("\nAvailable Attack Profiles:")
        profiles = self.chain.get_available_profiles()
        for name, stages in profiles.items():
            print(f"  - {name.ljust(25)} ({len(stages)} stages)")
            if self.verbose:
                print(f"      Stages: {', '.join(stages[:3])}...")
        print("")

    def run_profile(self, profile: str, exclude_cleanup: bool = False):
        """Run a specific attack profile"""
        print(f"\n[+] Starting Agent with profile: {profile}")
        print(f"[+] Target: {self.target if self.target else 'Localhost/Discovery'}")
        
        exclude_stages = [AttackStage.CLEANUP] if exclude_cleanup else None
        
        try:
            results = self.chain.execute_full_chain(
                profile=profile,
                exclude_stages=exclude_stages
            )
            self._report_results(results)
        except KeyboardInterrupt:
            print("\n[!] Agent execution interrupted by user.")
        except Exception as e:
            print(f"\n[!] Error during execution: {e}")
            self.logger.error(f"Agent execution error: {e}")

    def run_custom(self, stages: List[str]):
        """Run a custom set of stages"""
        print("\n[+] Starting Agent with custom stages")
        
        # Convert string stages to AttackStage enum
        enum_stages = []
        for s in stages:
            try:
                enum_stages.append(AttackStage(s))
            except ValueError:
                print(f"[!] Warning: Invalid stage '{s}', skipping.")
        
        if not enum_stages:
            print("[!] No valid stages selected.")
            return

        try:
            results = self.chain.execute_full_chain(stages=enum_stages)
            self._report_results(results)
        except Exception as e:
            print(f"\n[!] Error during execution: {e}")

    def run_autonomous(self):
        """Run in autonomous mode with dynamic decision making"""
        print(f"\n[+] Starting Agent in AUTONOMOUS MODE against {self.target or 'Local Network'}")
        print("[*] Phase 1: Initial Reconnaissance")
        
        # Step 1: Run Recon
        recon_results = self.chain.execute_stage(AttackStage.INITIAL_RECON)
        
        # Analyze results
        open_ports = []
        if 'results' in recon_results and 'open_ports' in recon_results['results']:
            open_ports = recon_results['results']['open_ports']
        
        print(f"[*] Analysis: Found open ports: {open_ports}")
        
        # Step 2: Decide next steps
        next_stages = []
        
        # Decision Logic
        if 80 in open_ports or 443 in open_ports or 8080 in open_ports:
            print("[!] Decision: Web ports detected. Scheduling Web App Vulnerability Scan.")
            next_stages.append(AttackStage.WEB_APP_TESTING)
            # Also schedule transfer attacks as web ports might host LLM interfaces
            print("[!] Decision: Potential LLM interface. Scheduling Transfer Attacks.")
            next_stages.append(AttackStage.TRANSFER_ATTACK)
            
        if 445 in open_ports or 139 in open_ports:
            print("[!] Decision: SMB ports detected. Scheduling Credential Harvesting.")
            next_stages.append(AttackStage.CREDENTIAL_HARVEST)
            
        if 22 in open_ports:
            print("[!] Decision: SSH detected. Scheduling Brute Force attempts.")
            if AttackStage.CREDENTIAL_HARVEST not in next_stages:
                next_stages.append(AttackStage.CREDENTIAL_HARVEST)

        if not next_stages:
            print("[-] Decision: No specific attack vectors identified. Ending engagement.")
            return

        # Step 3: Execute decided stages
        print(f"[*] Phase 2: Executing {len(next_stages)} dynamic stages...")
        for stage in next_stages:
            self.chain.execute_stage(stage, delay=1.0)
            
        print("[+] Autonomous engagement complete.")

    def _report_results(self, results: dict):
        """Print a summary of the results"""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {results.get('overall_status', 'Unknown')}")
        print(f"Duration: {results.get('duration', 0):.2f}s")
        print(f"Detected OS: {results.get('detected_os', 'Unknown')}")
        
        print("\nStage Results:")
        for stage_res in results.get('stages', []):
            status = stage_res.get('status', 'unknown')
            print(f"  - {stage_res['stage'].ljust(20)}: {status.upper()}")
            
            # Print specific findings if any
            if 'results' in stage_res and stage_res['results']:
                # Just print a summary or count of results to avoid clutter
                res_data = stage_res['results']
                if isinstance(res_data, dict):
                    keys = list(res_data.keys())
                    if keys:
                        print(f"      Findings: {', '.join(keys[:3])}...")
        
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Red Team Agent - Automated Attack Chain Orchestrator")
    parser.add_argument("-t", "--target", help="Target IP address, domain, or URL")
    parser.add_argument("-m", "--model-path", help="Path to model or HuggingFace ID for whitebox attacks")
    parser.add_argument("-p", "--profile", help="Attack profile to run (e.g., recon_only, full_engagement)")
    parser.add_argument("-l", "--list-profiles", action="store_true", help="List available attack profiles")
    parser.add_argument("--stages", nargs="+", help="Run specific stages (space separated)")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup stage")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--autonomous", "--auto", action="store_true", help="Run in autonomous mode (decides next steps based on recon)")
    
    args = parser.parse_args()
    
    agent = RedTeamAgent(target=args.target, model_path=args.model_path)
    
    if args.list_profiles:
        agent.list_profiles()
        return

    if args.autonomous:
        agent.run_autonomous()
        return

    if args.interactive:
        print("Red Team Agent - Interactive Mode")
        print("---------------------------------")
        if not args.target:
            target = input("Enter target IP/Domain (leave empty for local/discovery): ").strip()
            if target:
                agent.target = target
                agent.chain.target = target
        
        agent.list_profiles()
        profile = input("Select profile to run (default: recon_only): ").strip() or "recon_only"
        
        confirm = input(f"Ready to run '{profile}' against '{agent.target or 'Local/Discovery'}'? (y/n): ")
        if confirm.lower() == 'y':
            agent.run_profile(profile, exclude_cleanup=args.no_cleanup)
        else:
            print("Execution cancelled.")
        return

    if args.stages:
        agent.run_custom(args.stages)
    elif args.profile:
        agent.run_profile(args.profile, exclude_cleanup=args.no_cleanup)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
