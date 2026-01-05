# Network Discovery & Target Selection Guide

## Overview
The Red Team Kit now includes stealthy network discovery and interactive target selection capabilities. This allows you to automatically discover hosts on your network and select which ones to test.

## Features

### 1. Stealthy Network Discovery
- **Auto-detects** local network range (/24 subnet)
- **Stealthy scanning** using ICMP ping with random delays
- **Minimal footprint** to avoid IDS/IPS detection
- **Hostname resolution** for discovered hosts

### 2. Interactive Target Selection
- **Simple numbered list** display of discovered hosts
- **Flexible selection**:
  - Single: `1` or `3`
  - Multiple: `1,3,5`
  - Range: `1-5`
  - Mixed: `1,3-5,7`
  - All: `all`

### 3. Sequential Execution (Stealthy)
- Executes attack chain on each selected target **one at a time**
- **Random delays** between targets (2-5 seconds)
- **Lower traffic volume** = less detectable
- **Natural timing patterns** = avoids IDS alerts

## Quick Start

### Basic Usage

```python
from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery
from core.modules.target_selector import TargetSelector
from core.modules.attack_chain import AttackChain

logger = FrameworkLogger("engagement")

# Step 1: Discover network hosts (stealthy)
discovery = NetworkDiscovery(logger)
results = discovery.discover_local_network(stealth_mode=True)

# Step 2: Select targets interactively
selector = TargetSelector(logger)
selected_ips = selector.interactive_select(results['hosts'])

# Step 3: Execute attack chain sequentially (stealthy)
chain = AttackChain(logger)
results = chain.execute_on_targets(
    targets=selected_ips,
    profile="recon_only",
    sequential=True  # Sequential is more stealthy
)
```

### Run the Example

```bash
cd redteam_kit
python3 examples/network_discovery_example.py
```

## Network Discovery Options

### Stealthy Mode (Recommended)
```python
# Slow but stealthy - uses ICMP ping with random delays (0.5-2 seconds)
results = discovery.discover_local_network(stealth_mode=True)
```

### Fast Mode (Less Stealthy)
```python
# Faster but more detectable - uses port scanning
results = discovery.discover_local_network(stealth_mode=False)
```

### Custom Network Range
```python
# Discover specific network range
results = discovery.discover_network("192.168.1.0/24", stealth_mode=True)
```

## Target Selection Examples

### Interactive Selection
```
DISCOVERED NETWORK HOSTS
======================================================================
#    IP Address        Hostname                 Open Ports          
----------------------------------------------------------------------
1    192.168.1.100    server01                22, 80, 443        
2    192.168.1.101    workstation01           22, 3389           
3    192.168.1.102    printer01               none               
======================================================================
Total: 3 hosts discovered

Select targets (numbers, ranges like '1-5', or 'all'): 1,3
```

### Programmatic Filtering
```python
# Filter by open ports
ssh_hosts = selector.filter_by_ports(results['hosts'], [22])
http_hosts = selector.filter_by_ports(results['hosts'], [80, 443])

# Filter by hostname pattern
servers = selector.filter_by_hostname(results['hosts'], "server")
```

## Attack Chain Execution

### Sequential (Stealthy - Recommended)
```python
# Executes one target at a time with random delays
results = chain.execute_on_targets(
    targets=["192.168.1.100", "192.168.1.101"],
    profile="recon_only",
    sequential=True  # Stealthy mode
)
```

### Parallel (Fast - Less Stealthy)
```python
# Executes all targets simultaneously (not recommended for stealth)
results = chain.execute_on_targets(
    targets=["192.168.1.100", "192.168.1.101"],
    profile="recon_only",
    sequential=False  # Fast but detectable
)
```

## Why Sequential is More Stealthy

1. **Lower Traffic Volume** - One target at a time = less network traffic
2. **Natural Timing** - Random delays mimic human behavior
3. **Less Detectable** - Doesn't create traffic bursts
4. **Avoids Rate Limits** - Reduces chance of triggering rate limiting
5. **Smaller Footprint** - Minimal impact on network monitoring

## Selection Format Examples

- `1` - Select host #1
- `1,3,5` - Select hosts 1, 3, and 5
- `1-5` - Select hosts 1 through 5
- `1,3-5,7` - Select hosts 1, 3-5, and 7
- `all` - Select all discovered hosts

## Output Example

```
[+] Selected 2 target(s):
    - 192.168.1.100
    - 192.168.1.101

[1/2] Executing on 192.168.1.100
[1/2] Completed 192.168.1.100: completed
Waiting 3.2s before next target (stealthy delay)...
[2/2] Executing on 192.168.1.101
[2/2] Completed 192.168.1.101: completed

EXECUTION SUMMARY
======================================================================
Total targets: 2
Completed: 2
Failed: 0
Duration: 45.67 seconds
```

## Best Practices

1. **Always use stealth mode** for network discovery
2. **Use sequential execution** for multi-target attacks
3. **Start with recon_only** profile to assess targets
4. **Select targets carefully** - don't test everything
5. **Monitor timing** - random delays help avoid detection

## Troubleshooting

### No hosts discovered
- Check network connectivity
- Verify network range detection
- Try non-stealthy mode to debug

### Selection errors
- Use valid numbers (1 to number of hosts)
- Check range format (e.g., "1-5" not "5-1")
- Use commas to separate selections

### Execution failures
- Check target accessibility
- Verify attack profile exists
- Review logs for detailed errors

