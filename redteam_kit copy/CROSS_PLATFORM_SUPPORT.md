# Cross-Platform Support Status

## Summary
**Now 21 out of 25 modules (84%) fully support both macOS and Windows!**

## ✅ Fully Cross-Platform Modules (Work on Windows/macOS/Linux)

These modules use pure Python libraries or work universally:

1. **adversarial_prompts.py** - Pure text manipulation
2. **advanced_payloads.py** - Pure encoding (Base64, ROT13, hex)
3. **context_poisoning.py** - Pure text manipulation
4. **jailbreak_techniques.py** - Pure text manipulation
5. **token_manipulation.py** - Pure text manipulation
6. **prompt_injection.py** - Pure text manipulation
7. **web_app_testing.py** - Uses `requests` library (works everywhere)
8. **exploit.py** - Uses `requests`/`socket` (works everywhere)
9. **osint.py** - Uses `requests`/`subprocess` (works everywhere)
10. **network_enumeration.py** - Uses `subprocess`/`socket` (works everywhere)

## ✅ Cross-Platform with OS-Specific Implementations

These modules have explicit Windows/macOS/Linux support:

11. **os_detection.py** ✅ - Detects Windows, macOS, Linux
12. **post_exploit.py** ✅ - Windows paths, commands, and techniques
13. **advanced_persistence.py** ✅ - Windows scheduled tasks, registry, services
14. **covering_tracks.py** ✅ - Windows event logs, registry cleanup
15. **memory_operations.py** ✅ - Windows LSASS dumping, Linux memory operations
16. **privilege_escalation.py** ✅ - Windows UAC bypass, Linux sudo/SUID
17. **recon.py** ✅ - Windows port detection, Linux/macOS scanning
18. **attack_chain.py** ✅ - Orchestrates all modules
19. **credential_attacks.py** ✅ - Windows: `net use` for SMB, OpenSSH/PowerShell for SSH, RDP port check
20. **network_pivoting.py** ✅ - Windows: OpenSSH client, `netsh` port forwarding, `taskkill`
21. **c2_communication.py** ✅ - Windows ping commands, platform-aware subprocess calls

## ❌ Platform-Specific Modules

22. **active_directory.py** ❌ - Windows Active Directory only (by design)
23. **wifi_redteam.py** ❌ - macOS/Linux only (Windows requires different WiFi tools)

## 📊 Statistics

- **Total Modules:** 25
- **Fully Cross-Platform:** 21 (84%)
- **Platform-Specific:** 2 (8%)
- **AI Testing Modules (OS-agnostic):** 7 (28%)

## 🎯 Windows Support Added

### credential_attacks.py
- ✅ SSH: Uses OpenSSH client (Windows 10+) or PowerShell remoting fallback
- ✅ SMB: Uses `net use` command for Windows SMB authentication
- ✅ RDP: Checks RDP port accessibility (port 3389)

### network_pivoting.py
- ✅ SSH Tunnels: Uses OpenSSH client (Windows 10+) with `netsh` fallback
- ✅ SOCKS Proxy: Uses OpenSSH client for dynamic port forwarding
- ✅ Port Forwarding: Uses `netsh interface portproxy` as Windows alternative
- ✅ Tunnel Management: Uses `taskkill` instead of `kill` on Windows

### c2_communication.py
- ✅ ICMP Beacon: Uses Windows `ping -n` syntax
- ✅ Platform-aware subprocess calls throughout

## ✅ Current State

**Most modules DO support both macOS and Windows**, especially:
- Core exploitation modules (post_exploit, recon, persistence)
- OS detection (runs first in attack chain)
- AI testing modules (work everywhere)
- Credential attacks (Windows: `net use`, OpenSSH, PowerShell remoting)
- Network pivoting (Windows: OpenSSH, `netsh`, `taskkill`)

The only remaining gaps are:
- WiFi red teaming (Windows requires different tools)
- Active Directory (Windows-only by design)
