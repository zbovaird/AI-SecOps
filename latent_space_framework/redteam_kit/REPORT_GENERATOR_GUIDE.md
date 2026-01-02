# Red Team Report Generator

## Overview

The Red Team Report Generator creates comprehensive reports from your red team engagement activities. It aggregates results from all modules and generates professional reports in Markdown and JSON formats.

## Quick Start

### Generate Report from Test Results

```bash
cd redteam_kit
python3 examples/generate_report.py --targets 192.168.1.100 --run-tests --profile recon_only
```

### Generate Report from Existing Results

```bash
python3 examples/generate_report.py --results-json results.json --output my_report.md
```

### Generate Both Formats

```bash
python3 examples/generate_report.py --results-json results.json --format both --output report
# Generates: report.md and report.json
```

## Usage Examples

### Basic Usage

```python
from utils.logger import FrameworkLogger
from core.modules.report_generator import RedTeamReportGenerator
from core.modules.attack_chain import AttackChain

logger = FrameworkLogger("engagement")
generator = RedTeamReportGenerator(logger)

# Set metadata
generator.set_engagement_metadata(targets=["192.168.1.100"])

# Run tests
chain = AttackChain(logger)
results = chain.execute_on_targets(["192.168.1.100"], profile="recon_only")

# Add results to generator
for target_ip, target_results in results["results"].items():
    for stage_name, stage_data in target_results.get("stage_results", {}).items():
        generator.add_module_results(stage_name, stage_data)

# Generate report
generator.generate_markdown_report("redteam_report.md")
generator.generate_json_report("redteam_report.json")
```

### Adding Custom Findings

```python
# Add a finding
generator.add_finding(
    severity="High",
    title="Open SSH Port",
    description="SSH port 22 is exposed to the network",
    evidence=["Port scan revealed port 22/tcp open", "SSH banner: OpenSSH 7.4"],
    recommendation="Restrict SSH access to authorized IPs only"
)

# Generate report
generator.generate_markdown_report("report.md")
```

## Report Sections

The generated report includes:

1. **Executive Summary**
   - Targets tested
   - Engagement duration
   - Modules executed
   - Findings summary by severity

2. **Methodology**
   - List of modules executed
   - Testing approach

3. **Security Findings**
   - Findings grouped by severity (Critical, High, Medium, Low, Info)
   - Each finding includes:
     - Title and description
     - Evidence
     - Recommendations

4. **Detailed Results**
   - Complete module execution results
   - Organized by category (Reconnaissance, Exploitation, Post-Exploitation)

5. **Recommendations**
   - Priority remediations
   - General security recommendations

## Command Line Options

```bash
python3 examples/generate_report.py [OPTIONS]

Options:
  --targets IP1 IP2 ...     Target IPs/hostnames
  --results-json FILE       Path to JSON file with existing results
  --output FILE             Output file path (default: redteam_report.md)
  --format FORMAT          Output format: markdown, json, or both
  --profile PROFILE         Attack profile to use when running tests
  --run-tests              Run tests before generating report
```

## Report Formats

### Markdown (.md)
- Human-readable format
- Suitable for sharing with stakeholders
- Includes formatted sections and code blocks

### JSON (.json)
- Machine-readable format
- Suitable for automation and integration
- Complete structured data

## Integration with Attack Chain

The report generator automatically integrates with the AttackChain:

```python
from core.modules.attack_chain import AttackChain
from core.modules.report_generator import RedTeamReportGenerator

chain = AttackChain(logger)
generator = RedTeamReportGenerator(logger)

# Execute attack chain
results = chain.execute_on_targets(["192.168.1.100"], profile="recon_only")

# Generate report
generator.set_engagement_metadata(targets=["192.168.1.100"])
for target_ip, target_results in results["results"].items():
    for stage_name, stage_data in target_results.get("stage_results", {}).items():
        generator.add_module_results(stage_name, stage_data)

generator.generate_markdown_report("engagement_report.md")
```

## Finding Severity Levels

- **Critical**: Immediate action required (e.g., unauthorized access)
- **High**: Significant security issue (e.g., exposed sensitive data)
- **Medium**: Moderate security concern (e.g., open ports)
- **Low**: Minor security issue (e.g., information disclosure)
- **Info**: Informational finding (e.g., network discovery)

## Example Output

```markdown
# Red Team Engagement Report

## Executive Summary

- **Targets:** 1
- **Duration:** 5m 23s
- **Modules Executed:** 3
- **Total Findings:** 5

### Findings by Severity

- **High:** 2
- **Medium:** 2
- **Info:** 1

## Security Findings

### High Severity Findings

#### 1. Open SSH Port
**Description:** SSH port 22 is exposed to the network
**Evidence:**
- Port scan revealed port 22/tcp open
- SSH banner: OpenSSH 7.4
**Recommendation:** Restrict SSH access to authorized IPs only
```

## Best Practices

1. **Run tests first**: Use `--run-tests` to execute tests and generate report
2. **Use profiles**: Select appropriate attack profiles for your engagement
3. **Add findings**: Manually add findings for critical discoveries
4. **Review reports**: Always review generated reports before sharing
5. **Version control**: Track report versions for engagement history

## Notes

- Reports are saved to the specified output path
- JSON reports contain complete structured data
- Markdown reports are formatted for readability
- All timestamps are included in reports
- Reports include metadata about the engagement

