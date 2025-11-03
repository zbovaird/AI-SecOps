"""
Red Team Testing Kit for AI Security Assessment
FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY
"""

# Core modules
from .modules import (
    adversarial_prompts,
    prompt_injection,
    jailbreak_techniques,
    token_manipulation,
    context_poisoning,
    advanced_payloads
)

# Utilities
from .utils import logger, config_loader

__version__ = "1.0.0"
__author__ = "AI SecOps Team"

