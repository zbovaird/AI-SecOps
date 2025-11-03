# AI SecOps Workspace

A comprehensive workspace for AI Security Operations, focused on red teaming AI/ML models using industry-standard tools like Microsoft PyRIT and IBM Adversarial Robustness Toolbox (ART).

## Overview

This workspace enables:
- **Blackbox Testing**: Test deployed models without internal access
- **Whitebox Testing**: Deep analysis with full model access
- **Workflow Testing**: End-to-end security assessment pipelines
- **Red Team Testing**: Real target reconnaissance, exploitation, and post-exploitation capabilities

**⚠️ WARNING: FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

This toolkit is designed for legitimate security testing purposes only. Use only on systems you own or have explicit written authorization to test.

## Installation

### Quick Install (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-secops.git
   cd ai-secops
   ```

2. **Run installation script**:
   
   **Linux/macOS**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
   
   **Windows**:
   ```batch
   install.bat
   ```

3. **Activate virtual environment**:
   
   **Linux/macOS**:
   ```bash
   source venv/bin/activate
   ```
   
   **Windows**:
   ```batch
   venv\Scripts\activate.bat
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate.bat  # Windows

# Install package
pip install -e .

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check if redteam-kit command is available
redteam-kit --help

# Or run Python directly
python -c "from redteam_kit.core.modules.recon import ReconModule; print('Installation successful!')"
```

## Quick Start

1. **Install Tools**:
   ```bash
   ./install.sh  # Linux/macOS
   # OR
   install.bat    # Windows
   ```

2. **Explore Capabilities**:
   ```bash
   python explore_tools_detailed.py
   ```

3. **Review Documentation**:
   - `plan.md` - Strategic plan and architecture
   - `instructions.md` - Setup and usage instructions
   - `tools_capabilities.md` - Detailed tool capabilities by testing type

## Project Structure

```
AI SecOps/
├── plan.md                      # Strategic plan
├── instructions.md              # Setup instructions (PyRIT/ART + Red Team Kit)
├── tools_capabilities.md        # Tool capabilities reference
├── requirements.txt             # Python dependencies (PyRIT/ART)
├── setup.py                     # Package installation script
├── install.sh                   # Linux/macOS installation script
├── install.bat                  # Windows installation script
├── install_tools.sh             # Legacy installation script
├── explore_tools_detailed.py    # Tool exploration script
├── pyrit_gradio_app.py          # PyRIT Gradio GUI
├── pyrit_gemini_api.py          # PyRIT Gemini integration
├── pyrit_vertex_deepseek.py    # PyRIT DeepSeek integration
├── prompt_injection_test_gemini.py  # Prompt injection testing
├── redteam_kit/                 # Red Team Testing Kit (real target capabilities)
│   ├── core/modules/            # Core exploitation modules
│   ├── utils/                   # Utilities (logger, config, process resilience)
│   ├── examples/                # Usage examples
│   ├── instructions.md          # Red Team Kit usage instructions
│   └── README.md                # Red Team Kit documentation
└── README.md                    # This file
```

## Tools

### Microsoft PyRIT
Python Risk Identification Tool for red teaming generative AI systems.

**Key Strengths:**
- Prompt-based attack scenarios
- Endpoint testing (perfect for Vertex AI)
- Workflow orchestration
- Response scoring and evaluation

### IBM ART
Adversarial Robustness Toolbox for comprehensive model security evaluation.

**Key Strengths:**
- Extensive attack library (blackbox and whitebox)
- Defense mechanisms
- Robustness metrics
- Framework support (TensorFlow, PyTorch, Scikit-learn)

### Red Team Testing Kit
Custom framework for real target security testing and red team exercises.

**Key Features:**
- Real reconnaissance (port scanning, service detection, banner grabbing)
- Exploit testing (SQL injection, RCE, command injection, XXE)
- Post-exploitation (credential harvesting, privilege escalation, lateral movement)
- Persistence mechanisms (scheduled tasks, startup scripts, registry modification)
- Multi-stage attack chain orchestration
- Support for IP addresses, domains, URLs, and hostnames
- Process resilience handling (automatic fallback when processes are killed)
- Cross-platform support (Windows, macOS, Linux)

See `redteam_kit/README.md` for detailed documentation.

## Red Team Kit Modules

The Red Team Kit includes 28 comprehensive modules organized into several categories:

### Reconnaissance & Discovery Modules

- **`recon.py`** - Real reconnaissance with port scanning, service detection, banner grabbing, OS fingerprinting, and vulnerability scanning
- **`network_discovery.py`** - Stealthy network scanning to discover active hosts on local networks with auto-detection
- **`network_enumeration.py`** - DNS enumeration, subdomain discovery, network mapping, SMB/LDAP/SNMP enumeration
- **`os_detection.py`** - Operating system detection using TTL analysis, service banners, and platform-specific checks
- **`osint.py`** - Open source intelligence gathering (domain info, email discovery, employee info, tech stack, WHOIS)
- **`target_selector.py`** - Interactive target selection from discovered hosts with multi-target support

### Exploitation Modules

- **`exploit.py`** - Real exploit testing including SQL injection, RCE, command injection, XXE, buffer overflow, deserialization
- **`web_app_testing.py`** - Web application security testing (XSS, CSRF, SSRF, directory traversal, file upload, OWASP Top 10)
- **`credential_attacks.py`** - Password spraying, brute force attacks, hash cracking, Kerberos attacks (Windows/macOS/Linux support)

### Post-Exploitation Modules

- **`post_exploit.py`** - Credential harvesting (browsers: Chrome/Firefox/Safari/Edge/Opera, credential managers: Keychain/Credential Manager/Keyring, password files, config files), privilege escalation, lateral movement, data collection, and exfiltration with process resilience
- **`privilege_escalation.py`** - Linux (SUID, sudo misconfig, cron jobs, kernel exploits) and Windows (UAC bypass, service misconfig, DLL hijacking) escalation
- **`lateral_movement.py`** - Network pivoting techniques including SSH tunneling, RDP access, SMB share access, and credential reuse
- **`memory_operations.py`** - Process memory dumping, credential extraction from memory, LSASS dumping (Windows), memory pattern searching

### Persistence & Evasion Modules

- **`advanced_persistence.py`** - Scheduled tasks, startup scripts, registry modification, service creation, backdoor user accounts
- **`advanced_evasion.py`** - Memory-based execution, API unhooking, sleep masking, process/DLL injection, reflective loading
- **`covering_tracks.py`** - Log clearing, artifact removal, timestamp modification, command history clearing, registry cleanup

### Network & Communication Modules

- **`network_pivoting.py`** - SSH tunneling, SOCKS proxy creation, port forwarding, DNS/ICMP tunneling, proxy chains (Windows/macOS/Linux)
- **`c2_communication.py`** - Command and control channels (HTTP, DNS, ICMP beaconing) with encrypted channels (Windows/macOS/Linux)
- **`wifi_redteam.py`** - WiFi security testing including network scanning, handshake capture, password cracking, deauthentication, evil twin attacks

### Active Directory Modules

- **`active_directory.py`** - User/group/computer enumeration, Kerberoasting, AS-REP roasting, Pass-the-hash, DCSync (Windows-specific)

### AI Security Testing Modules

- **`adversarial_prompts.py`** - Generate obfuscated prompts using 8+ techniques (encoding, character substitution, Unicode, etc.)
- **`prompt_injection.py`** - 8 injection patterns including role-play, instruction injection, context manipulation, and jailbreak attempts
- **`jailbreak_techniques.py`** - 8+ jailbreak methods (DAN, AIM, character roleplay, logic bomb, etc.) to bypass AI safety restrictions
- **`token_manipulation.py`** - Token-level transformations (whitespace manipulation, case variation, Unicode addition, token splitting)
- **`context_poisoning.py`** - Context poisoning techniques (false context, history poisoning, system prompt injection, role poisoning)
- **`advanced_payloads.py`** - Payload generation with Base64, ROT13, hexadecimal encoding, multi-layer encoding, noise injection

### Orchestration & Reporting Modules

- **`attack_chain.py`** - Multi-stage attack orchestration with predefined profiles, custom stages, OS detection integration, and multi-target support
- **`report_generator.py`** - Comprehensive report generation (Markdown/JSON) with executive summaries, detailed findings, credential reporting, and recommendations

### Utility Modules

- **`process_resilience.py`** - Automatic handling of process kills with retry logic, Python fallbacks, and shell script fallbacks
- **`stealth_file_access.py`** - Stealthy file access patterns to avoid permission prompts (macOS/Windows)
- **`logger.py`** - Framework logging with multiple levels and file output
- **`config_loader.py`** - Configuration file loading and management

All modules include:
- ✅ Cross-platform support (Windows, macOS, Linux)
- ✅ Process resilience (automatic fallback on process kills)
- ✅ Stealthy timing patterns (random delays, jitter)
- ✅ Real target capabilities (not dummy data)
- ✅ Comprehensive documentation and examples

## Testing Types

### Blackbox Testing
- Test models via APIs/endpoints
- No internal model knowledge required
- Suitable for production deployments
- Tools: PyRIT prompt targets, ART query-based attacks

### Whitebox Testing
- Full access to model internals
- Gradient-based attacks
- Internal state analysis
- Tools: ART gradient attacks, PyRIT model access

### Workflow Testing
- End-to-end security assessment
- Multi-stage attack pipelines
- Comprehensive evaluation
- Tools: PyRIT orchestrators, ART evaluation frameworks

## Next Steps

1. Follow `instructions.md` for detailed setup
   - PyRIT and IBM ART installation and usage
   - Red Team Kit attack chain usage for real targets
2. Review `tools_capabilities.md` to understand available features
3. For AI Model Testing:
   - Deploy a test model to Google Vertex AI
   - Run PyRIT/ART security assessments
4. For Real Target Testing:
   - See `redteam_kit/README.md` for Red Team Kit overview
   - See `redteam_kit/instructions.md` for detailed attack chain usage
5. Run your first security assessment

## Resources

- [Microsoft PyRIT Documentation](https://azure.github.io/PyRIT/)
- [IBM ART Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## License

[Add your license here]

