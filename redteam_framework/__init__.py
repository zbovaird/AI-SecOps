"""
Red Team Framework - Multi-Model Adversarial Testing

A model-agnostic framework for systematic red teaming of LLMs.

Phases:
- Phase 1: Core infrastructure (schema, logging, evaluators, perturbations)
- Phase 2: Core experiments (decode fragility, logit lens)
- Phase 3: Advanced probes (multi-turn drift, attention routing, KV cache)
- Phase 4: Cross-model benchmarking harness
"""

__version__ = "0.2.0"

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
from .core.model_adapter import ModelAdapter, load_model

__all__ = [
    # Core schema
    "RunConfig",
    "ModelConfig", 
    "DecodingConfig",
    "SampleRecord",
    "BehaviorDelta",
    "StateDelta",
    "StabilityDelta",
    "ExperimentResult",
    # Logging
    "get_logger",
    "setup_logging",
    # Model adapter
    "ModelAdapter",
    "load_model",
]
