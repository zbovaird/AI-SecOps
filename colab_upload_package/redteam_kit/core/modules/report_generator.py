"""
Red Team Report Generator
Generates comprehensive reports from red team engagement results
"""

from typing import Dict, List, Optional
import time
import json
from datetime import datetime
from pathlib import Path
from utils.logger import FrameworkLogger


class RedTeamReportGenerator:
    """Red team report generator"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize report generator
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.engagement_data = {
            "metadata": {
                "start_time": None,
                "end_time": None,
                "duration": None,
                "targets": [],
                "modules_executed": [],
                "version": "1.0"
            },
            "reconnaissance": {},
            "exploitation": {},
            "post_exploitation": {},
            "persistence": {},
            "lateral_movement": {},
            "data_collection": {},
            "findings": [],
            "recommendations": []
        }
    
    def add_module_results(self, module_name: str, results: Dict):
        """
        Add results from a module execution
        
        Args:
            module_name: Name of the module (e.g., "recon", "exploit")
            results: Results dictionary from module execution
        """
        self.logger.info(f"Adding results from {module_name} module")
        
        # Map module names to report sections
        section_map = {
            "recon": "reconnaissance",
            "os_detection": "reconnaissance",
            "network_discovery": "reconnaissance",
            "network_enumeration": "reconnaissance",
            "osint": "reconnaissance",
            "exploit": "exploitation",
            "web_app_testing": "exploitation",
            "credential_attacks": "exploitation",
            "post_exploit": "post_exploitation",
            "harvest_credentials": "post_exploitation",
            "privilege_escalation": "post_exploitation",
            "lateral_movement": "lateral_movement",
            "advanced_persistence": "persistence",
            "data_collection": "data_collection",
            "covering_tracks": "post_exploitation"
        }
        
        section = section_map.get(module_name, "reconnaissance")
        
        if section not in self.engagement_data:
            self.engagement_data[section] = {}
        
        if module_name not in self.engagement_data[section]:
            self.engagement_data[section][module_name] = []
        
        self.engagement_data[section][module_name].append(results)
        
        # Track module execution
        if module_name not in self.engagement_data["metadata"]["modules_executed"]:
            self.engagement_data["metadata"]["modules_executed"].append(module_name)
    
    def add_finding(self, severity: str, title: str, description: str, 
                   evidence: Optional[List[str]] = None, recommendation: Optional[str] = None):
        """
        Add a security finding
        
        Args:
            severity: Severity level (Critical, High, Medium, Low, Info)
            title: Finding title
            description: Detailed description
            evidence: List of evidence items
            recommendation: Recommended remediation
        """
        finding = {
            "severity": severity,
            "title": title,
            "description": description,
            "evidence": evidence or [],
            "recommendation": recommendation,
            "timestamp": time.time()
        }
        
        self.engagement_data["findings"].append(finding)
        self.logger.info(f"Added finding: {severity} - {title}")
    
    def set_engagement_metadata(self, targets: List[str], start_time: Optional[float] = None):
        """
        Set engagement metadata
        
        Args:
            targets: List of target IPs/hostnames
            start_time: Engagement start time (defaults to now)
        """
        self.engagement_data["metadata"]["targets"] = targets
        self.engagement_data["metadata"]["start_time"] = start_time or time.time()
        self.engagement_data["metadata"]["end_time"] = time.time()
        duration = self.engagement_data["metadata"]["end_time"] - self.engagement_data["metadata"]["start_time"]
        self.engagement_data["metadata"]["duration"] = duration
    
    def generate_summary(self) -> Dict:
        """
        Generate engagement summary
        
        Returns:
            Dictionary containing engagement summary
        """
        findings_by_severity = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0,
            "Info": 0
        }
        
        for finding in self.engagement_data["findings"]:
            severity = finding["severity"]
            if severity in findings_by_severity:
                findings_by_severity[severity] += 1
        
        summary = {
            "targets": len(self.engagement_data["metadata"]["targets"]),
            "modules_executed": len(self.engagement_data["metadata"]["modules_executed"]),
            "total_findings": len(self.engagement_data["findings"]),
            "findings_by_severity": findings_by_severity,
            "duration_seconds": self.engagement_data["metadata"]["duration"],
            "duration_formatted": self._format_duration(self.engagement_data["metadata"]["duration"])
        }
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Markdown report content
        """
        self.engagement_data["metadata"]["end_time"] = time.time()
        duration = self.engagement_data["metadata"]["end_time"] - self.engagement_data["metadata"]["start_time"]
        self.engagement_data["metadata"]["duration"] = duration
        
        summary = self.generate_summary()
        
        report = []
        report.append("# Red Team Engagement Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Targets:** {summary['targets']}")
        report.append(f"- **Duration:** {summary['duration_formatted']}")
        report.append(f"- **Modules Executed:** {summary['modules_executed']}")
        report.append(f"- **Total Findings:** {summary['total_findings']}")
        report.append("")
        report.append("### Findings by Severity")
        report.append("")
        for severity, count in summary['findings_by_severity'].items():
            if count > 0:
                report.append(f"- **{severity}:** {count}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Methodology
        report.append("## Methodology")
        report.append("")
        report.append("### Modules Executed")
        report.append("")
        for module in self.engagement_data["metadata"]["modules_executed"]:
            report.append(f"- {module}")
        report.append("")
        report.append("---")
        report.append("")
        
        # Findings
        if self.engagement_data["findings"]:
            report.append("## Security Findings")
            report.append("")
            
            # Group by severity
            findings_by_severity = {
                "Critical": [],
                "High": [],
                "Medium": [],
                "Low": [],
                "Info": []
            }
            
            for finding in self.engagement_data["findings"]:
                severity = finding["severity"]
                if severity in findings_by_severity:
                    findings_by_severity[severity].append(finding)
            
            for severity in ["Critical", "High", "Medium", "Low", "Info"]:
                findings = findings_by_severity[severity]
                if findings:
                    report.append(f"### {severity} Severity Findings")
                    report.append("")
                    
                    for i, finding in enumerate(findings, 1):
                        report.append(f"#### {i}. {finding['title']}")
                        report.append("")
                        report.append(f"**Description:** {finding['description']}")
                        report.append("")
                        
                        if finding['evidence']:
                            report.append("**Evidence:**")
                            for evidence in finding['evidence']:
                                report.append(f"- {evidence}")
                            report.append("")
                        
                        if finding['recommendation']:
                            report.append(f"**Recommendation:** {finding['recommendation']}")
                            report.append("")
                        
                        report.append("---")
                        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        
        # Reconnaissance
        if self.engagement_data["reconnaissance"]:
            report.append("### Reconnaissance")
            report.append("")
            for module_name, module_results in self.engagement_data["reconnaissance"].items():
                report.append(f"#### {module_name.title()}")
                report.append("")
                report.append("```json")
                report.append(json.dumps(module_results, indent=2))
                report.append("```")
                report.append("")
        
        # Exploitation
        if self.engagement_data["exploitation"]:
            report.append("### Exploitation")
            report.append("")
            for module_name, module_results in self.engagement_data["exploitation"].items():
                report.append(f"#### {module_name.title()}")
                report.append("")
                report.append("```json")
                report.append(json.dumps(module_results, indent=2))
                report.append("```")
                report.append("")
        
        # Post-Exploitation
        if self.engagement_data["post_exploitation"]:
            report.append("### Post-Exploitation")
            report.append("")
            for module_name, module_results in self.engagement_data["post_exploitation"].items():
                report.append(f"#### {module_name.title().replace('_', ' ')}")
                report.append("")
                
                # Special handling for credential harvesting
                if module_name == "harvest_credentials" or "credentials_found" in str(module_results):
                    # Extract credentials from results
                    all_creds = []
                    for result in module_results:
                        if isinstance(result, dict):
                            if "credentials_found" in result:
                                all_creds.extend(result["credentials_found"])
                            elif isinstance(result, dict) and any("credentials" in str(v).lower() for v in result.values()):
                                # Handle nested structures
                                for key, value in result.items():
                                    if isinstance(value, dict) and "credentials_found" in value:
                                        all_creds.extend(value["credentials_found"])
                                    elif isinstance(value, list):
                                        for item in value:
                                            if isinstance(item, dict) and "credentials_found" in item:
                                                all_creds.extend(item["credentials_found"])
                    
                    if all_creds:
                        report.append(f"**Credentials Discovered: {len(all_creds)}**")
                        report.append("")
                        report.append("| Type | Source | Value | Location |")
                        report.append("|------|--------|-------|----------|")
                        
                        for cred in all_creds:
                            cred_type = cred.get("type", "unknown")
                            source = cred.get("source", "unknown")
                            value = cred.get("value", "")
                            location = cred.get("location", cred.get("name", ""))
                            
                            # Truncate long values for display
                            if len(value) > 60:
                                value_display = value[:57] + "..."
                            else:
                                value_display = value
                            
                            report.append(f"| {cred_type} | {source} | `{value_display}` | {location} |")
                        report.append("")
                        
                        # Add detailed credential information
                        report.append("**Detailed Credential Information:**")
                        report.append("")
                        for i, cred in enumerate(all_creds, 1):
                            report.append(f"{i}. **{cred.get('type', 'unknown').upper()}**")
                            report.append(f"   - **Source:** `{cred.get('source', 'unknown')}`")
                            if cred.get('name'):
                                report.append(f"   - **Variable Name:** `{cred.get('name')}`")
                            report.append(f"   - **Value:** `{cred.get('value', 'N/A')}`")
                            if cred.get('location'):
                                report.append(f"   - **Location:** `{cred.get('location')}`")
                            report.append("")
                    else:
                        report.append("*No credentials found*")
                        report.append("")
                else:
                    report.append("```json")
                    report.append(json.dumps(module_results, indent=2))
                    report.append("```")
                    report.append("")
        
        # Add dedicated Credentials section if any were found
        credentials = self.extract_credentials_from_results()
        if credentials:
            report.append("## 🔐 Credentials Discovered")
            report.append("")
            report.append(f"**Total Credentials Found:** {len(credentials)}")
            report.append("")
            report.append("### Credential Summary")
            report.append("")
            
            # Group by type
            cred_by_type = {}
            cred_by_source = {}
            
            for cred in credentials:
                cred_type = cred.get("type", "unknown")
                source = cred.get("source", "unknown")
                
                if cred_type not in cred_by_type:
                    cred_by_type[cred_type] = []
                cred_by_type[cred_type].append(cred)
                
                if source not in cred_by_source:
                    cred_by_source[source] = []
                cred_by_source[source].append(cred)
            
            report.append("**By Type:**")
            for cred_type, creds in sorted(cred_by_type.items()):
                report.append(f"- **{cred_type}:** {len(creds)}")
            report.append("")
            
            report.append("**By Source:**")
            for source, creds in sorted(cred_by_source.items()):
                report.append(f"- **{source}:** {len(creds)}")
            report.append("")
            
            report.append("### Detailed Credential Information")
            report.append("")
            report.append("> **⚠️ WARNING:** The following credentials were discovered during this engagement.")
            report.append("> **IMMEDIATE ACTION REQUIRED:** Rotate all credentials immediately.")
            report.append("")
            
            for i, cred in enumerate(credentials, 1):
                report.append(f"#### Credential {i}: {cred.get('type', 'unknown').upper()}")
                report.append("")
                report.append(f"- **Type:** `{cred.get('type', 'unknown')}`")
                
                if cred.get('name'):
                    report.append(f"- **Variable Name:** `{cred.get('name')}`")
                
                report.append(f"- **Source:** `{cred.get('source', 'unknown')}`")
                
                # Show full value (not truncated) for credentials
                value = cred.get('value', 'N/A')
                report.append(f"- **Value:** `{value}`")
                
                if cred.get('location'):
                    report.append(f"- **File Location:** `{cred.get('location')}`")
                
                # Add context if available
                if cred.get('source') == 'environment':
                    report.append(f"- **Environment Variable:** `{cred.get('name', 'N/A')}`")
                
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Recommendations
        if self.engagement_data["findings"]:
            report.append("## Recommendations")
            report.append("")
            report.append("Based on the findings above, the following recommendations are provided:")
            report.append("")
            
            # Credential-specific recommendations
            credentials = self.extract_credentials_from_results()
            if credentials:
                report.append("### ⚠️ CRITICAL: Credential Security")
                report.append("")
                report.append("**IMMEDIATE ACTION REQUIRED:** Credentials were discovered during this engagement.")
                report.append("")
                report.append("1. **Rotate all discovered credentials immediately**")
                report.append("   - Change passwords, API keys, tokens, and secrets")
                report.append("   - Invalidate existing sessions")
                report.append("")
                report.append("2. **Review credential storage locations:**")
                for source in sorted(set(c.get("source", "unknown") for c in credentials)):
                    report.append(f"   - `{source}`")
                report.append("")
                report.append("3. **Implement credential management best practices:**")
                report.append("   - Use environment variables or secure vaults (AWS Secrets Manager, HashiCorp Vault)")
                report.append("   - Never commit credentials to version control")
                report.append("   - Use least-privilege access principles")
                report.append("   - Regularly audit credential storage")
                report.append("   - Implement credential rotation policies")
                report.append("   - Use strong, unique credentials")
                report.append("   - Monitor for credential exposure")
                report.append("")
            
            critical_recs = [f for f in self.engagement_data["findings"] if f["severity"] == "Critical"]
            high_recs = [f for f in self.engagement_data["findings"] if f["severity"] == "High"]
            
            if critical_recs or high_recs:
                report.append("### Priority Remediations")
                report.append("")
                
                for finding in critical_recs + high_recs:
                    if finding['recommendation']:
                        report.append(f"- **{finding['title']}:** {finding['recommendation']}")
                        report.append("")
            
            report.append("### General Security Recommendations")
            report.append("")
            report.append("- Implement network segmentation")
            report.append("- Enable logging and monitoring")
            report.append("- Regular security assessments")
            report.append("- Security awareness training")
            report.append("- Patch management process")
            report.append("")
        
        # Footer
        report.append("---")
        report.append("")
        report.append("*This report was generated by the Red Team Kit Framework*")
        report.append("")
        report.append(f"*Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        report_content = "\n".join(report)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_content
    
    def generate_json_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate JSON report
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            JSON report dictionary
        """
        self.engagement_data["metadata"]["end_time"] = time.time()
        duration = self.engagement_data["metadata"]["end_time"] - self.engagement_data["metadata"]["start_time"]
        self.engagement_data["metadata"]["duration"] = duration
        
        report = {
            "metadata": self.engagement_data["metadata"],
            "summary": self.generate_summary(),
            "reconnaissance": self.engagement_data["reconnaissance"],
            "exploitation": self.engagement_data["exploitation"],
            "post_exploitation": self.engagement_data["post_exploitation"],
            "persistence": self.engagement_data["persistence"],
            "lateral_movement": self.engagement_data["lateral_movement"],
            "data_collection": self.engagement_data["data_collection"],
            "findings": self.engagement_data["findings"],
            "credentials": self.extract_credentials_from_results()  # Dedicated credentials section
        }
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"JSON report saved to {output_path}")
        
        return report
    
    def extract_credentials_from_results(self) -> List[Dict]:
        """
        Extract all credentials from engagement results
        
        Returns:
            List of credential dictionaries
        """
        all_credentials = []
        
        # Check post_exploitation section
        if "post_exploitation" in self.engagement_data:
            for module_name, module_results in self.engagement_data["post_exploitation"].items():
                if isinstance(module_results, list):
                    for result in module_results:
                        if isinstance(result, dict):
                            if "credentials_found" in result:
                                all_credentials.extend(result["credentials_found"])
                            elif "credentials" in result:
                                if isinstance(result["credentials"], list):
                                    all_credentials.extend(result["credentials"])
                elif isinstance(module_results, dict):
                    # Handle case where module_results is a dict directly
                    if "credentials_found" in module_results:
                        all_credentials.extend(module_results["credentials_found"])
        
        return all_credentials
    
    def get_engagement_data(self) -> Dict:
        """Get all engagement data"""
        return self.engagement_data

