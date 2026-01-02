#!/usr/bin/env python3
"""
Red Team Report Generator Script
Generates comprehensive reports from red team engagement results
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import FrameworkLogger
from core.modules.report_generator import RedTeamReportGenerator
from core.modules.attack_chain import AttackChain, AttackStage


def main():
    parser = argparse.ArgumentParser(description="Generate Red Team Engagement Report")
    parser.add_argument("--targets", nargs="+", help="Target IPs/hostnames")
    parser.add_argument("--results-json", help="Path to JSON file with existing results")
    parser.add_argument("--output", "-o", default="redteam_report.md", help="Output file path")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both", help="Output format")
    parser.add_argument("--profile", help="Attack profile to use")
    parser.add_argument("--run-tests", action="store_true", help="Run tests before generating report")
    
    args = parser.parse_args()
    
    print("="*70)
    print("RED TEAM REPORT GENERATOR")
    print("="*70)
    
    logger = FrameworkLogger("report_generator")
    generator = RedTeamReportGenerator(logger)
    
    # Set metadata
    targets = args.targets or ["TARGET_IP_HERE"]  # Replace with actual target IPs
    generator.set_engagement_metadata(targets)
    
    # Load existing results or run tests
    if args.results_json:
        print(f"\n[+] Loading results from {args.results_json}")
        import json
        with open(args.results_json, 'r') as f:
            results = json.load(f)
        
        # Add results to generator
        for section, modules in results.items():
            if section != "metadata":
                for module_name, module_results in modules.items():
                    if isinstance(module_results, list):
                        for result in module_results:
                            generator.add_module_results(module_name, result)
                    else:
                        generator.add_module_results(module_name, module_results)
    
    elif args.run_tests:
        print("\n[+] Running red team tests...")
        print(f"[+] Targets: {', '.join(targets)}")
        
        chain = AttackChain(logger)
        profile = args.profile or "recon_only"
        
        print(f"[+] Using profile: {profile}")
        print("[+] Executing attack chain...\n")
        
        # Execute on targets
        results = chain.execute_on_targets(targets, profile=profile, sequential=True)
        
        # Add results to generator
        for target_ip, target_results in results.get("results", {}).items():
            for stage_name, stage_data in target_results.get("stage_results", {}).items():
                module_name = stage_name.lower().replace("_", "")
                generator.add_module_results(module_name, stage_data)
        
        # Generate findings from results
        _generate_findings_from_results(generator, results)
    
    else:
        print("\n[!] No results provided. Use --results-json or --run-tests")
        print("[*] Generating empty report template...")
        
        # Add sample findings for demonstration
        generator.add_finding(
            "High",
            "Sample Finding",
            "This is a sample finding. Replace with actual results.",
            ["Evidence 1", "Evidence 2"],
            "Implement proper security controls"
        )
    
    # Generate reports
    print(f"\n[+] Generating report(s)...")
    
    output_dir = Path(args.output).parent
    output_name = Path(args.output).stem
    
    if args.format in ["markdown", "both"]:
        md_path = output_dir / f"{output_name}.md"
        generator.generate_markdown_report(str(md_path))
        print(f"[+] Markdown report: {md_path}")
    
    if args.format in ["json", "both"]:
        json_path = output_dir / f"{output_name}.json"
        generator.generate_json_report(str(json_path))
        print(f"[+] JSON report: {json_path}")
    
    # Print summary
    summary = generator.generate_summary()
    print("\n" + "="*70)
    print("REPORT SUMMARY")
    print("="*70)
    print(f"Targets: {summary['targets']}")
    print(f"Duration: {summary['duration_formatted']}")
    print(f"Modules Executed: {summary['modules_executed']}")
    print(f"Total Findings: {summary['total_findings']}")
    print("\nFindings by Severity:")
    for severity, count in summary['findings_by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    print("="*70)


def _generate_findings_from_results(generator: RedTeamReportGenerator, results: Dict):
    """Generate findings from attack chain results"""
    
    # Check for discovered hosts
    if "reconnaissance" in str(results).lower():
        generator.add_finding(
            "Info",
            "Network Discovery",
            "Network reconnaissance was performed to identify active hosts",
            ["Network scanning completed"],
            "Monitor network traffic for reconnaissance activities"
        )
    
    # Check for open ports
    if "open_ports" in str(results):
        generator.add_finding(
            "Medium",
            "Open Ports Discovered",
            "Multiple open ports were discovered on target systems",
            ["Port scanning revealed exposed services"],
            "Close unnecessary ports and restrict access to required services"
        )
    
    # Check for vulnerabilities
    if "vulnerable" in str(results).lower() or "vulnerabilities" in str(results).lower():
        generator.add_finding(
            "High",
            "Vulnerabilities Detected",
            "Security vulnerabilities were identified during testing",
            ["Vulnerability scanning completed"],
            "Address identified vulnerabilities according to risk assessment"
        )
    
    # Check for successful connections
    if "successful_connections" in str(results):
        generator.add_finding(
            "Critical",
            "Unauthorized Access Achieved",
            "Successful connections were established to target systems",
            ["Lateral movement successful"],
            "Implement network segmentation and access controls"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

