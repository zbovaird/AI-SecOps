# Red Team Kit - Project Structure

```
redteam_kit/
├── __init__.py                 # Package initialization
├── main.py                     # Main entry point
├── config.json                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── core/
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       ├── adversarial_prompts.py    # Adversarial prompt generation
│       ├── prompt_injection.py        # Prompt injection patterns
│       ├── jailbreak_techniques.py    # Jailbreak methods
│       ├── token_manipulation.py      # Token manipulation
│       ├── context_poisoning.py        # Context poisoning
│       ├── advanced_payloads.py        # Advanced payload generation
│       ├── attack_chain.py            # Multi-stage attack orchestration
│       ├── recon.py                   # Reconnaissance module
│       ├── post_exploit.py            # Post-exploitation
│       ├── exploit.py                 # Exploit module
│       ├── advanced_persistence.py    # Persistence techniques
│       └── advanced_evasion.py        # Evasion techniques
│
└── utils/
    ├── __init__.py
    ├── logger.py               # Logging utility
    └── config_loader.py        # Configuration loader
```

## Key Features

### Core Modules
- **Adversarial Prompts**: 8+ obfuscation techniques
- **Prompt Injection**: 8 injection patterns
- **Jailbreak Techniques**: 8 jailbreak methods
- **Token Manipulation**: 4 manipulation techniques
- **Attack Chain**: Multi-stage orchestration

### Utilities
- Structured logging
- Configuration management
- Easy extensibility

## Quick Start

```bash
cd redteam_kit
python main.py --mode adversarial --prompt "Test prompt" --variants 5
```

