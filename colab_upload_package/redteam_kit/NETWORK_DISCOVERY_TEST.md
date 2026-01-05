# Local Network Discovery Test

## Issue
Python scripts are being killed by the system (likely resource limits or security restrictions).

## Manual Testing Instructions

You can test the network discovery functionality manually by running:

```bash
cd redteam_kit
python3 examples/simple_network_test.py
```

Or use the full module:

```python
from utils.logger import FrameworkLogger
from core.modules.network_discovery import NetworkDiscovery
from core.modules.target_selector import TargetSelector

# Initialize
logger = FrameworkLogger("test")
discovery = NetworkDiscovery(logger)

# Discover local network (stealthy)
results = discovery.discover_local_network(stealth_mode=True)
print(f"Found {len(results['hosts'])} hosts")

# Display results
selector = TargetSelector(logger)
selector.display_hosts(results['hosts'])
```

## Alternative: Direct Terminal Test

You can also test network discovery directly using ping:

```bash
# Get your network range
ip=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
network=$(echo $ip | cut -d'.' -f1-3)

# Scan first 10 IPs
for i in {1..10}; do
    ping -c 1 -W 1 ${network}.${i} > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "${network}.${i} is alive"
    fi
done
```

## What the Code Does

The network discovery module:
1. Auto-detects your local network range (/24 subnet)
2. Uses ICMP ping to discover active hosts
3. Applies stealthy random delays (0.5-2 seconds) between scans
4. Optionally performs port scanning (disabled in stealth mode)
5. Resolves hostnames for discovered hosts

## Stealth Features

- Random delays between scans (0.5-2 seconds)
- ICMP-only scanning (less detectable than port scans)
- Natural timing patterns
- Minimal network footprint

## Notes

If Python scripts are being killed, this might be due to:
- System resource limits (check with `ulimit -a`)
- Security restrictions
- Memory limits
- macOS security policies

Try running from a different directory or with different permissions if needed.

