"""
Experiment implementations for the red team framework.

Phase 2 experiments for systematic vulnerability discovery.
"""

from .decode_fragility import (
    DecodeFragilitySweep,
    DecodeGridConfig,
    FragilityReport,
    PromptResult,
    run_decode_fragility_experiment,
    DEFAULT_REDTEAM_PROMPTS,
)

from .logit_lens import (
    LogitLensProbe,
    LogitLensReport,
    PromptAnalysis,
    LayerAnalysis,
    run_logit_lens_experiment,
    DEFAULT_BENIGN_PROMPTS,
    DEFAULT_ADVERSARIAL_PROMPTS,
)

__all__ = [
    # Decode Fragility
    "DecodeFragilitySweep",
    "DecodeGridConfig",
    "FragilityReport",
    "PromptResult",
    "run_decode_fragility_experiment",
    "DEFAULT_REDTEAM_PROMPTS",
    
    # Logit Lens
    "LogitLensProbe",
    "LogitLensReport",
    "PromptAnalysis",
    "LayerAnalysis",
    "run_logit_lens_experiment",
    "DEFAULT_BENIGN_PROMPTS",
    "DEFAULT_ADVERSARIAL_PROMPTS",
]

# Future experiments (Phase 3):
# - multiturn_drift: Multi-turn Residual Drift Accumulation
# - attention_routing: Attention Routing Graph Extraction
# - kv_cache: KV-Cache Persistence Probes
