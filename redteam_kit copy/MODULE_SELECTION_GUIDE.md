# Attack Chain Module Selection Guide

## Overview
The Attack Chain supports flexible module selection through **attack profiles** and **custom stage lists**. This allows you to run only the modules needed for specific engagement objectives.

## Attack Profiles

### Available Profiles

1. **`recon_only`** - Reconnaissance only
   - OS Detection
   - Initial Reconnaissance

2. **`initial_access`** - Gain initial access
   - OS Detection
   - Initial Reconnaissance
   - Credential Harvest
   - Privilege Escalation

3. **`post_exploit_focus`** - Post-exploitation activities
   - OS Detection
   - Credential Harvest
   - Privilege Escalation
   - Lateral Movement
   - Data Collection

4. **`persistence_establishment`** - Establish persistence
   - OS Detection
   - Persistence

5. **`data_exfiltration`** - Data exfiltration focus
   - OS Detection
   - Data Collection
   - Data Exfiltration

6. **`full_engagement`** - Complete attack chain (all stages)

## Usage Examples

### Using Predefined Profiles

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain

logger = FrameworkLogger("engagement")
chain = AttackChain(logger, target="192.168.1.100")

# Reconnaissance only
results = chain.execute_full_chain(profile="recon_only")

# Initial access focus
results = chain.execute_full_chain(profile="initial_access")

# Post-exploitation focus
results = chain.execute_full_chain(profile="post_exploit_focus")
```

### Custom Stage Selection

```python
from core.modules.attack_chain import AttackChain, AttackStage

chain = AttackChain(logger, target="192.168.1.100")

# Select specific stages
stages = [
    AttackStage.OS_DETECTION,
    AttackStage.INITIAL_RECON,
    AttackStage.CREDENTIAL_HARVEST
]
results = chain.execute_full_chain(stages=stages)
```

### Excluding Stages

```python
# Use full engagement but skip cleanup
results = chain.execute_full_chain(
    profile="full_engagement",
    exclude_stages=[AttackStage.CLEANUP]
)
```

### Creating Custom Profiles

```python
# Create a custom profile
chain.create_custom_profile("web_app_focus", [
    AttackStage.OS_DETECTION,
    AttackStage.INITIAL_RECON,
    AttackStage.CREDENTIAL_HARVEST
])

# Use custom profile
results = chain.execute_full_chain(profile="web_app_focus")
```

### View Available Profiles

```python
# List all available profiles
profiles = chain.get_available_profiles()
print("Available profiles:")
for name, stages in profiles.items():
    print(f"  {name}: {stages}")
```

## Attack Stages

All available stages:

- `AttackStage.OS_DETECTION` - Detect target OS
- `AttackStage.INITIAL_RECON` - Initial reconnaissance
- `AttackStage.CREDENTIAL_HARVEST` - Harvest credentials
- `AttackStage.PRIVILEGE_ESCALATION` - Escalate privileges
- `AttackStage.PERSISTENCE` - Establish persistence
- `AttackStage.LATERAL_MOVEMENT` - Move laterally
- `AttackStage.DATA_COLLECTION` - Collect data
- `AttackStage.DATA_EXFILTRATION` - Exfiltrate data
- `AttackStage.CLEANUP` - Cleanup and cover tracks

## Best Practices

1. **Always start with OS detection** - It's automatically included
2. **Use profiles for common scenarios** - Faster than manual selection
3. **Create custom profiles** - For repeated engagement types
4. **Exclude cleanup in testing** - Use `exclude_stages=[AttackStage.CLEANUP]`
5. **Check profile before execution** - Use `get_available_profiles()` to verify

## Engagement Scenarios

### Scenario 1: External Reconnaissance
```python
chain.execute_full_chain(profile="recon_only")
```

### Scenario 2: Initial Access Attempt
```python
chain.execute_full_chain(profile="initial_access")
```

### Scenario 3: Post-Exploitation Focus
```python
chain.execute_full_chain(profile="post_exploit_focus")
```

### Scenario 4: Complete Engagement
```python
chain.execute_full_chain(profile="full_engagement")
```

### Scenario 5: Custom Web App Testing
```python
chain.create_custom_profile("web_app_test", [
    AttackStage.OS_DETECTION,
    AttackStage.INITIAL_RECON,
    AttackStage.CREDENTIAL_HARVEST
])
chain.execute_full_chain(profile="web_app_test")
```

