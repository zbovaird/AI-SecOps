"""
Red Team Framework - Multi-Model Adversarial Testing

A model-agnostic framework for systematic red teaming of LLMs.
"""

__version__ = "0.1.0"

from .core.schema import (
    RunConfig,
    ModelConfig,
    DecodingConfig,
    SampleRecord,
    BehaviorDelta,
    StateDelta,
    StabilityDelta,
    ExperimentResult,
)
from .core.logging import get_logger, setup_logging

__all__ = [
    "RunConfig",
    "ModelConfig", 
    "DecodingConfig",
    "SampleRecord",
    "BehaviorDelta",
    "StateDelta",
    "StabilityDelta",
    "ExperimentResult",
    "get_logger",
    "setup_logging",
]
