"""
Red Team Testing Kit - Core Modules
FOR AUTHORIZED SECURITY TESTING ONLY
"""

# Import latent space modules for direct access
from .latent_space_instrumentation import ModelInstrumentation
from .latent_space_analysis import LatentSpaceAnalyzer

__all__ = [
    # AI Security Testing Modules
    'adversarial_prompts',
    'prompt_injection',
    'jailbreak_techniques',
    'token_manipulation',
    'context_poisoning',
    'advanced_payloads',
    'semantic_perturbation',
    'adaptive_perturbation',
    'pyrit_integration',
    'gradient_attacks',
    
    # Latent Space Analysis Modules (for notebook usage)
    'ModelInstrumentation',
    'LatentSpaceAnalyzer',
    
    # Core Exploitation Modules
    'recon',
    'exploit',
    'post_exploit',
    'advanced_persistence',
    'advanced_evasion',
    'attack_chain',
    'os_detection',
    'network_discovery',
    'target_selector',
    'report_generator',
    
    # Advanced Red Team Modules
    'credential_attacks',
    'c2_communication',
    'network_enumeration',
    'web_app_testing',
    'osint',
    'active_directory',
    'privilege_escalation',
    'covering_tracks',
    'network_pivoting',
    'memory_operations',
    'wifi_redteam'
]

