# Red Team Testing Kit

A comprehensive framework for AI security testing and red team exercises.

## Features

- **Adversarial Prompt Generation**: Multiple obfuscation techniques
- **Prompt Injection**: Various injection patterns
- **Jailbreak Techniques**: Multiple jailbreak methods
- **Token Manipulation**: Token-level evasion techniques
- **Attack Chain Orchestration**: Multi-stage attack workflows

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from redteam_kit import RedTeamKit

kit = RedTeamKit()

# Generate adversarial prompts
variants = await kit.generate_adversarial_prompts("Your prompt here")

# Create injection variants
injections = kit.create_injection_variants("Base prompt", "Command to inject")

# Apply jailbreak techniques
jailbroken = kit.apply_jailbreak_techniques("Your query")
```

### Command Line

```bash
# Generate adversarial variants
python main.py --mode adversarial --prompt "Your prompt" --variants 5

# Create injection variants
python main.py --mode injection --prompt "Base prompt"

# Apply jailbreak techniques
python main.py --mode jailbreak --prompt "Your query"
```

## Modules

- `adversarial_prompts`: Prompt generation with obfuscation
- `prompt_injection`: Injection pattern library
- `jailbreak_techniques`: Jailbreak method collection
- `token_manipulation`: Token-level techniques
- `attack_chain`: Multi-stage attack orchestration

## Warning

**FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

This toolkit is designed for legitimate security testing purposes only.

