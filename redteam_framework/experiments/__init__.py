"""
Experiment implementations for the red team framework.

Phase 2-3 experiments for systematic vulnerability discovery.
"""

# Phase 2: Core experiments
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

# Phase 3: Advanced probes
from .multiturn_drift import (
    MultiTurnDriftExperiment,
    DriftStrategies,
    DriftReport,
    ConversationResult,
    TurnResult,
    run_multiturn_drift_experiment,
    DEFAULT_TARGET_REQUESTS,
)

from .attention_routing import (
    AttentionRoutingAnalyzer,
    AttentionRoutingReport,
    HeadProfile,
    LayerAttentionProfile,
    run_attention_routing_experiment,
)

from .kv_cache import (
    KVCacheProbe,
    KVCacheReport,
    KVPersistenceResult,
    TurnKVProfile,
    run_kv_cache_experiment,
)

__all__ = [
    # Phase 2: Decode Fragility
    "DecodeFragilitySweep",
    "DecodeGridConfig",
    "FragilityReport",
    "PromptResult",
    "run_decode_fragility_experiment",
    "DEFAULT_REDTEAM_PROMPTS",
    
    # Phase 2: Logit Lens
    "LogitLensProbe",
    "LogitLensReport",
    "PromptAnalysis",
    "LayerAnalysis",
    "run_logit_lens_experiment",
    "DEFAULT_BENIGN_PROMPTS",
    "DEFAULT_ADVERSARIAL_PROMPTS",
    
    # Phase 3: Multi-turn Drift
    "MultiTurnDriftExperiment",
    "DriftStrategies",
    "DriftReport",
    "ConversationResult",
    "TurnResult",
    "run_multiturn_drift_experiment",
    "DEFAULT_TARGET_REQUESTS",
    
    # Phase 3: Attention Routing
    "AttentionRoutingAnalyzer",
    "AttentionRoutingReport",
    "HeadProfile",
    "LayerAttentionProfile",
    "run_attention_routing_experiment",
    
    # Phase 3: KV Cache
    "KVCacheProbe",
    "KVCacheReport",
    "KVPersistenceResult",
    "TurnKVProfile",
    "run_kv_cache_experiment",
]
