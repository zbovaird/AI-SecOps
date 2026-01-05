# Red Team Testing Kit

A comprehensive framework for AI security testing and red team exercises with real target capabilities.

## Features

- **Real Target Reconnaissance**: Port scanning, service detection, banner grabbing, OS fingerprinting
- **Adversarial Prompt Generation**: Multiple obfuscation techniques for AI testing
- **Prompt Injection**: Various injection patterns for AI security testing
- **Jailbreak Techniques**: Multiple jailbreak methods for AI model testing
- **Token Manipulation**: Token-level evasion techniques
- **Attack Chain Orchestration**: Multi-stage attack workflows for real targets
- **Post-Exploitation**: Real credential harvesting, privilege escalation, lateral movement
- **Persistence**: Real persistence mechanisms (scheduled tasks, startup scripts, registry)
- **Advanced Evasion**: Evasion techniques for bypassing security controls
- **Resource Optimized**: Built-in delays and limits to prevent resource exhaustion

## Installation

```bash
cd redteam_kit
pip install -r requirements.txt
```

## Interactive Console

The Red Team Kit includes a Metasploit-style interactive console for easy operation.

```bash
# Launch the console
./console.sh
```

Once inside the console:
```bash
aisecops > help
aisecops > show modules
aisecops > use exploit/llm/prompt_injection
aisecops exploit/llm/prompt_injection > set TARGET 192.168.1.100
aisecops exploit/llm/prompt_injection > run
```

## Quick Start

### Basic Usage with Real Targets

```python
from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule

# Initialize and perform reconnaissance on a real target
logger = FrameworkLogger("engagement")
recon = ReconModule(logger, target="192.168.1.100")

# Scan target
results = recon.perform_recon()
print(f"Target: {results['target']}")
print(f"IP Address: {results['ip_address']}")
print(f"Open ports: {len(results['info_gathered']['network_info']['open_ports'])}")
print(f"Services: {len(results['info_gathered']['service_detection'])}")
```

### Full Attack Chain

```python
from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

# Initialize attack chain with target
logger = FrameworkLogger("engagement")
chain = AttackChain(logger, target="192.168.1.100")

# Execute full attack chain
results = chain.execute_full_chain()
print(f"Attack chain completed: {results['overall_status']}")
print(f"Duration: {results['duration']:.2f} seconds")
```

## Modules

### Core Exploitation Modules
- `recon.py`: Real reconnaissance with port scanning and service detection
- `exploit.py`: Real exploit testing (SQL injection, RCE, command injection, XXE)
- `post_exploit.py`: Real credential harvesting, privilege escalation, lateral movement
- `advanced_persistence.py`: Real persistence mechanisms
- `advanced_evasion.py`: Evasion techniques
- `attack_chain.py`: Multi-stage attack orchestration

### AI Security Testing Modules
- `adversarial_prompts.py`: Prompt generation with obfuscation
- `prompt_injection.py`: Injection pattern library
- `jailbreak_techniques.py`: Jailbreak method collection
- `token_manipulation.py`: Token-level techniques
- `context_poisoning.py`: Context poisoning attacks
- `advanced_payloads.py`: Payload generation

## Target Types Supported

The framework supports multiple target types:

- **IP Addresses**: `192.168.1.100`, `10.0.0.1`
- **Domain Names**: `example.com`, `target.org`
- **URLs**: `http://target.com`, `https://192.168.1.100`
- **Hostnames**: `server01.internal`

## Warning

**FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

This toolkit is designed for legitimate security testing purposes only. Use only on systems you own or have explicit written authorization to test.

## Documentation

- See `instructions.md` for detailed attack chain usage instructions
- See `MODULE_DOCUMENTATION.md` for detailed module information
- See `PROJECT_STRUCTURE.md` for project organization
- See `RESOURCE_OPTIMIZATION.md` for resource optimization details
- See `examples/` directory for usage examples

