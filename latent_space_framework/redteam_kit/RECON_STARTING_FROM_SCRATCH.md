# Using Recon Module - From Scratch Guide

## Two Approaches: Standalone vs Integrated

### Approach 1: Standalone Recon (Manual Control)

**When to use:** You want to gather intelligence first, analyze it, then manually decide what to do next.

**Step-by-step:**

```python
# 1. Start from scratch - just initialize
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

logger = FrameworkLogger("my_engagement")
recon = ReconModule(logger)

# 2. Perform reconnaissance
results = recon.perform_recon()

# 3. Analyze what you found
services = results["info_gathered"]["service_detection"]
vulnerabilities = results["info_gathered"]["vulnerability_scan"]

# 4. Make decisions based on findings
if services:
    for service in services:
        if service['service'] == 'http':
            print("Found web service - test for web vulnerabilities")
            # Now manually use exploit module or other tools
        
if vulnerabilities:
    for vuln in vulnerabilities:
        if vuln['severity'] == 'critical':
            print("Critical vulnerability - prioritize exploitation")
            # Manually execute exploit

# 5. Store for later
stored_data = recon.get_recon_data()
```

**Pros:**
- Full control over each step
- Can analyze before proceeding
- Flexible decision-making

**Cons:**
- More manual work
- Need to manually chain modules together

---

### Approach 2: Integrated Attack Chain (Automated Workflow)

**When to use:** You want an automated multi-stage engagement that flows from recon → exploit → post-exploit → persistence.

**Step-by-step:**

```python
# 1. Start with attack chain (includes recon automatically)
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

logger = FrameworkLogger("automated_engagement")
chain = AttackChain(logger)

# 2. Execute full chain (recon happens automatically first)
results = chain.execute_full_chain()

# OR execute stages individually:
chain.execute_stage(AttackStage.INITIAL_RECON)  # Recon happens here
chain.execute_stage(AttackStage.CREDENTIAL_HARVEST)
chain.execute_stage(AttackStage.PRIVILEGE_ESCALATION)
# etc.

# 3. Access recon data from chain
recon_data = chain.recon.get_recon_data()
```

**Pros:**
- Automated workflow
- Stages flow naturally
- Less manual work

**Cons:**
- Less control over individual steps
- Less flexibility for analysis

---

## Complete Example: Starting from Scratch

```python
#!/usr/bin/env python3
"""
Starting a Red Team Engagement from Scratch
"""

from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

# Initialize
logger = FrameworkLogger("engagement_2024")
recon = ReconModule(logger)

# Step 1: RECONNAISSANCE
print("[*] Phase 1: Reconnaissance")
results = recon.perform_recon()

# Step 2: ANALYZE
print("\n[*] Phase 2: Analysis")
services = results["info_gathered"]["service_detection"]
vulns = results["info_gathered"]["vulnerability_scan"]

print(f"Services found: {len(services)}")
print(f"Vulnerabilities found: {len(vulns)}")

# Step 3: DECIDE NEXT STEPS
print("\n[*] Phase 3: Planning")
if services:
    print("Planning exploits based on discovered services...")
    # Now you'd use exploit module, post_exploit module, etc.

# Step 4: EXECUTE (using other modules)
print("\n[*] Phase 4: Execution")
# Based on recon findings, use appropriate modules:
# - exploit.py for vulnerabilities
# - post_exploit.py for credential harvesting
# - advanced_persistence.py for maintaining access
# etc.
```

## Recommendation

**For beginners:** Start with **Standalone Recon** approach
- You learn how each module works individually
- You understand the data flow
- You can make informed decisions

**For advanced users:** Use **Integrated Attack Chain**
- Faster execution
- Automated workflows
- Less manual management

## Key Point

The recon module can be used **both ways**:
1. **Standalone** - Just gather intelligence, then manually use other modules
2. **Integrated** - Part of attack chain that automatically flows to next stages

Both approaches work! Choose based on your needs.

