"""
Transferability Framework (v3) - CKA-Based Attack Transfer Analysis

Analyzes attack transferability between open-weight and closed-weight models
using Centered Kernel Alignment (CKA) to identify optimal surrogate models.

Key capabilities:
- CKA similarity computation between open models
- Attack success measurement across all models (open + closed)
- Transferability correlation analysis
- Surrogate model recommendations
"""

__version__ = "0.1.0"

from .core.cka import CKA, linear_cka, rbf_cka
from .core.model_loader import ModelLoader, OpenModelConfig
from .core.attack_suite import AttackSuite, AttackResult
from .core.api_clients import OpenAIClient, AnthropicClient

__all__ = [
    # CKA
    "CKA",
    "linear_cka",
    "rbf_cka",
    # Model Loading
    "ModelLoader",
    "OpenModelConfig",
    # Attacks
    "AttackSuite",
    "AttackResult",
    # API Clients
    "OpenAIClient",
    "AnthropicClient",
]
