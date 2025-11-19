# Red Team Kit - Attack Chain Usage Instructions

The Red Team Kit provides a comprehensive attack chain framework for testing real targets. This guide covers how to use the attack chain and modules against different types of targets.

**⚠️ WARNING: FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

This toolkit is designed for legitimate security testing purposes only. Use only on systems you own or have explicit written authorization to test.

## Prerequisites

```bash
# Install required dependencies
cd redteam_kit
pip install -r requirements.txt
```

## Target Types Supported

The attack chain supports multiple target formats:

- **IP Addresses**: `192.168.1.100`, `10.0.0.1`
- **Domain Names**: `example.com`, `target.org`
- **URLs**: `http://target.com`, `https://192.168.1.100:8080`
- **Hostnames**: `server01.internal`

## Basic Attack Chain Usage

### 1. Single Target Reconnaissance

```python
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

# Initialize with target IP
logger = FrameworkLogger("recon")
recon = ReconModule(logger, target="192.168.1.100")

# Perform reconnaissance
results = recon.perform_recon()

# Analyze results
print(f"Target: {results['target']}")
print(f"IP: {results['ip_address']}")
print(f"Open ports: {len(results['info_gathered']['network_info']['open_ports'])}")
print(f"Services found: {len(results['info_gathered']['service_detection'])}")

# Access specific information
services = results["info_gathered"]["service_detection"]
for service in services:
    print(f"  - {service['service']} on port {service['port']}")
    print(f"    Version: {service['version']}")
```

### 2. Full Attack Chain Execution

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

# Initialize attack chain with target
logger = FrameworkLogger("engagement")
chain = AttackChain(logger, target="192.168.1.100")

# Execute full chain automatically
results = chain.execute_full_chain()

# Or execute stages individually
chain.execute_stage(AttackStage.INITIAL_RECON)
chain.execute_stage(AttackStage.CREDENTIAL_HARVEST)
chain.execute_stage(AttackStage.PRIVILEGE_ESCALATION)

# View results
all_results = chain.get_results()
```

### 3. Exploiting Discovered Vulnerabilities

```python
from utils.logger import FrameworkLogger
from core.modules.exploit import ExploitModule

logger = FrameworkLogger("exploit")
exploit = ExploitModule(logger)

# Test vulnerability first
test_result = exploit.test_exploit(
    target="http://192.168.1.100",
    exploit_type="sql_injection"
)

if test_result['vulnerable']:
    # Execute exploit
    result = exploit.execute_exploit(
        target="http://192.168.1.100",
        exploit_type="sql_injection"
    )
    print(f"Exploit status: {result['status']}")
    print(f"Access gained: {result['access_gained']}")
```

## Target-Specific Examples

### Example 1: Web Application Target

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

logger = FrameworkLogger("web_app_test")
chain = AttackChain(logger, target="https://target-webapp.com")

# Perform reconnaissance
recon_result = chain.execute_stage(AttackStage.INITIAL_RECON)

# Check for web services
services = recon_result["results"]["info_gathered"]["service_detection"]
web_services = [s for s in services if s["service"] in ["http", "https"]]

if web_services:
    # Test for SQL injection
    exploit = chain.exploit
    sql_test = exploit.test_exploit(
        target="https://target-webapp.com",
        exploit_type="sql_injection"
    )
    
    if sql_test['vulnerable']:
        exploit.execute_exploit(
            target="https://target-webapp.com",
            exploit_type="sql_injection"
        )
```

### Example 2: Internal Network Target

```python
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

logger = FrameworkLogger("internal_network")
recon = ReconModule(logger, target="10.0.0.50")

# Scan specific ports
results = recon.perform_recon(
    target="10.0.0.50",
    ports=[22, 80, 443, 3389, 445, 3306]
)

# Check for SSH
ssh_services = [s for s in results["info_gathered"]["service_detection"] 
                if s["service"] == "ssh"]

if ssh_services:
    print("SSH service detected - test for weak keys or brute force")
    # Use post_exploit module for credential attacks
```

### Example 3: Domain Name Target

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain

logger = FrameworkLogger("domain_test")
chain = AttackChain(logger, target="example.com")

# Domain will be resolved to IP automatically
results = chain.execute_full_chain()

# Check resolved IP
recon_data = chain.recon.get_recon_data()
print(f"Resolved IP: {recon_data['ip_address']}")
```

### Example 4: Custom Port Scanning

```python
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

logger = FrameworkLogger("custom_scan")
recon = ReconModule(logger)

# Scan custom ports
custom_ports = [8080, 8443, 9000, 27017, 5432]  # Custom web, MongoDB, PostgreSQL
results = recon.perform_recon(
    target="192.168.1.100",
    ports=custom_ports
)

print(f"Scanned {len(custom_ports)} custom ports")
print(f"Found {len(results['info_gathered']['network_info']['open_ports'])} open ports")
```

## Advanced Usage

### Post-Exploitation After Successful Exploit

```python
from utils.logger import FrameworkLogger
from core.modules.post_exploit import PostExploitation

logger = FrameworkLogger("post_exploit")
post_exploit = PostExploitation(logger, target="192.168.1.100")

# Harvest credentials
creds = post_exploit.harvest_credentials()
print(f"Credentials found: {creds['count']}")
print(f"Sources: {creds['sources']}")

# Escalate privileges
escalation = post_exploit.escalate_privileges()
print(f"Privilege: {escalation['previous_level']} -> {escalation['new_level']}")

# Move laterally
lateral = post_exploit.move_laterally()
print(f"Targets discovered: {len(lateral['targets_discovered'])}")
print(f"Successful connections: {len(lateral['successful_connections'])}")

# Collect data
data = post_exploit.collect_data()
print(f"Data collected: {data['items_collected']} items")
print(f"Total size: {data['total_size']} bytes")
```

### Establishing Persistence

```python
from utils.logger import FrameworkLogger
from core.modules.advanced_persistence import AdvancedPersistence

logger = FrameworkLogger("persistence")
persistence = AdvancedPersistence(logger)

# Establish persistence (requires payload path)
payload_path = "/path/to/backdoor.py"
result = persistence.establish_persistence(payload_path=payload_path)

print(f"Persistence methods: {result['methods_established']}")
print(f"Methods: {result['methods']}")
print(f"Survival rate: {result['survival_rate']:.2%}")

# View methods
methods = persistence.get_persistence_methods()
for method in methods:
    print(f"{method['type']}: {method['status']}")
```

### Complete Engagement Workflow

```python
#!/usr/bin/env python3
"""
Complete Red Team Engagement Workflow
"""

from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

def run_engagement(target: str):
    """Run complete engagement against target"""
    
    logger = FrameworkLogger("engagement")
    chain = AttackChain(logger, target=target)
    
    print(f"[*] Starting engagement against {target}")
    
    # Phase 1: Reconnaissance
    print("\n[*] Phase 1: Reconnaissance")
    recon_result = chain.execute_stage(AttackStage.INITIAL_RECON)
    recon_data = chain.recon.get_recon_data()
    
    print(f"    ✓ Target: {recon_data['target']}")
    print(f"    ✓ IP: {recon_data['ip_address']}")
    print(f"    ✓ Services: {len(recon_data['info_gathered']['service_detection'])}")
    
    # Phase 2: Exploitation
    print("\n[*] Phase 2: Exploitation")
    services = recon_data["info_gathered"]["service_detection"]
    
    for service in services:
        if service["service"] == "http" or service["service"] == "https":
            target_url = f"{service['service']}://{target}"
            test = chain.exploit.test_exploit(target_url, "sql_injection")
            if test['vulnerable']:
                print(f"    ✓ SQL injection vulnerability found!")
                chain.exploit.execute_exploit(target_url, "sql_injection")
    
    # Phase 3: Post-Exploitation
    print("\n[*] Phase 3: Post-Exploitation")
    chain.execute_stage(AttackStage.CREDENTIAL_HARVEST)
    chain.execute_stage(AttackStage.PRIVILEGE_ESCALATION)
    
    # Phase 4: Persistence
    print("\n[*] Phase 4: Persistence")
    chain.execute_stage(AttackStage.PERSISTENCE)
    
    # Phase 5: Data Collection
    print("\n[*] Phase 5: Data Collection")
    chain.execute_stage(AttackStage.DATA_COLLECTION)
    chain.execute_stage(AttackStage.DATA_EXFILTRATION)
    
    # Phase 6: Cleanup
    print("\n[*] Phase 6: Cleanup")
    chain.execute_stage(AttackStage.CLEANUP)
    
    print("\n[*] Engagement complete!")
    return chain.get_results()

# Run engagement
if __name__ == "__main__":
    results = run_engagement("192.168.1.100")
    print(f"\n[+] Attack chain status: {results['chain_complete']}")
```

## Module Usage

### Reconnaissance Module

The `ReconModule` performs real reconnaissance on targets:

```python
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

logger = FrameworkLogger("recon")
recon = ReconModule(logger, target="192.168.1.100")

# Perform full reconnaissance
results = recon.perform_recon()

# Or scan specific ports
results = recon.perform_recon(
    target="192.168.1.100",
    ports=[22, 80, 443, 3306]
)

# Get stored reconnaissance data
recon_data = recon.get_recon_data()
```

### Exploit Module

The `ExploitModule` tests and executes exploits:

```python
from utils.logger import FrameworkLogger
from core.modules.exploit import ExploitModule

logger = FrameworkLogger("exploit")
exploit = ExploitModule(logger)

# Available exploit types:
# - "sql_injection"
# - "command_injection"
# - "remote_code_execution"
# - "xxe_injection"
# - "buffer_overflow"
# - "deserialization"

# Test exploit first
test = exploit.test_exploit(
    target="http://target.com",
    exploit_type="sql_injection"
)

if test['vulnerable']:
    result = exploit.execute_exploit(
        target="http://target.com",
        exploit_type="sql_injection"
    )

# View exploit history
history = exploit.get_exploit_history()
```

### Post-Exploitation Module

The `PostExploitation` module handles post-exploitation activities:

```python
from utils.logger import FrameworkLogger
from core.modules.post_exploit import PostExploitation

logger = FrameworkLogger("post_exploit")
post_exploit = PostExploitation(logger, target="192.168.1.100")

# Harvest credentials from various sources
creds = post_exploit.harvest_credentials(target_path="/path/to/search")

# Attempt privilege escalation
escalation = post_exploit.escalate_privileges()

# Perform lateral movement
lateral = post_exploit.move_laterally(target_hosts=["10.0.0.1", "10.0.0.2"])

# Collect sensitive data
data = post_exploit.collect_data(search_paths=["/etc", "/var"])

# Exfiltrate collected data
exfil = post_exploit.exfiltrate_data()

# Access collected data
all_data = post_exploit.get_collected_data()
all_creds = post_exploit.get_credentials()
```

### Persistence Module

The `AdvancedPersistence` module establishes persistence mechanisms:

```python
from utils.logger import FrameworkLogger
from core.modules.advanced_persistence import AdvancedPersistence

logger = FrameworkLogger("persistence")
persistence = AdvancedPersistence(logger)

# Establish persistence
result = persistence.establish_persistence(payload_path="/path/to/payload")

# View established methods
methods = persistence.get_persistence_methods()

# Remove persistence (cleanup)
cleanup = persistence.remove_persistence()
```

### Attack Chain Module

The `AttackChain` orchestrates multi-stage attacks:

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

logger = FrameworkLogger("engagement")
chain = AttackChain(logger, target="192.168.1.100")

# Available stages:
# - AttackStage.INITIAL_RECON
# - AttackStage.CREDENTIAL_HARVEST
# - AttackStage.PRIVILEGE_ESCALATION
# - AttackStage.PERSISTENCE
# - AttackStage.LATERAL_MOVEMENT
# - AttackStage.DATA_COLLECTION
# - AttackStage.DATA_EXFILTRATION
# - AttackStage.CLEANUP

# Execute single stage
result = chain.execute_stage(AttackStage.INITIAL_RECON)

# Execute full chain
results = chain.execute_full_chain()

# Get all results
all_results = chain.get_results()

# Access individual modules
recon_data = chain.recon.get_recon_data()
exploit_history = chain.exploit.get_exploit_history()
```

## Command Line Usage

```bash
# Run reconnaissance on target
cd redteam_kit
python -c "
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

logger = FrameworkLogger('cli_recon')
recon = ReconModule(logger, target='192.168.1.100')
results = recon.perform_recon()
print(f'Open ports: {len(results[\"info_gathered\"][\"network_info\"][\"open_ports\"])}')
print(f'Services: {len(results[\"info_gathered\"][\"service_detection\"])}')
"
```

## Best Practices

### 1. Always Test Connectivity First

```python
# Verify target is reachable before full engagement
recon = ReconModule(logger, target="192.168.1.100")
results = recon.perform_recon(ports=[80, 443])  # Quick scan first
if results['status'] == 'failed':
    print("Target not reachable")
```

### 2. Use Targeted Port Scans

```python
# Don't scan all ports - use specific ports based on target type
web_ports = [80, 443, 8080, 8443]
database_ports = [3306, 5432, 27017]
```

### 3. Test Exploits Before Execution

```python
# Always test first
test = exploit.test_exploit(target, exploit_type)
if test['vulnerable']:
    exploit.execute_exploit(target, exploit_type)
```

### 4. Handle Errors Gracefully

```python
try:
    results = chain.execute_full_chain()
except Exception as e:
    logger.error(f"Attack chain failed: {e}")
    # Fallback to individual stages
```

### 5. Store Results for Analysis

```python
import json

results = chain.get_results()
with open('engagement_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Security Considerations

- **Authorization**: Only test targets you own or have explicit written authorization
- **Rate Limiting**: Don't overwhelm targets with rapid scans
- **Cleanup**: Always run cleanup phase to remove persistence mechanisms
- **Logging**: Review logs for unintended actions
- **Legal Compliance**: Ensure all activities comply with local laws and regulations

## Troubleshooting

### Target Not Resolving

```python
# Check DNS resolution
import socket
try:
    ip = socket.gethostbyname("example.com")
    print(f"Resolved to: {ip}")
except socket.gaierror:
    print("DNS resolution failed")
```

### Port Scanning Blocked

```python
# Try different timeout values
recon = ReconModule(logger, target="192.168.1.100")
results = recon.perform_recon(ports=[22, 80, 443])
# If ports appear closed, may be firewall blocking
```

### Exploit Testing Fails

```python
# Verify target is actually vulnerable
test = exploit.test_exploit(target, exploit_type)
print(f"Vulnerable: {test['vulnerable']}")
print(f"Confidence: {test['confidence']:.2%}")
# Low confidence may indicate false positive
```

### Permission Denied Errors

- Many operations require elevated privileges
- Check if you have necessary permissions on target system
- Some persistence methods require admin/root access

### Module Import Errors

```bash
# Ensure you're in the redteam_kit directory
cd redteam_kit

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Additional Resources

- See `README.md` for project overview
- See `MODULE_DOCUMENTATION.md` for detailed module information
- See `PROJECT_STRUCTURE.md` for project organization
- See `examples/` directory for more usage examples

## Getting Help

- Review module docstrings for detailed API documentation
- Check logs for error messages
- Review example scripts in `examples/` directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Extending the Kit

### Adding New Modules

When adding a new attack module to the kit, you must ensure it is accessible via the interactive console.

1.  **Create the Module**: Implement your module in `core/modules/`.
2.  **Update AttackChain**: Add the new stage to `AttackStage` enum and `AttackChain` class in `core/modules/attack_chain.py`.
3.  **Update Console**: Register the new module in `redteam_kit/console.py`.

**Example `console.py` entry:**

```python
"exploit/new_category/my_module": {
    "stage": AttackStage.MY_NEW_STAGE,
    "description": "Description of what the module does",
    "options": {
        "REQUIRED_OPTION": {"value": "default", "required": True, "description": "Help text"}
    }
}
```

