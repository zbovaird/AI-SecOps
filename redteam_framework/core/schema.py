"""
Canonical schema definitions for the red team framework.

All experiments output records conforming to these schemas, enabling
cross-model and cross-experiment comparisons.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
import json
import hashlib


class ModelFamily(str, Enum):
    """Supported model families with family-specific handling."""
    GEMMA = "gemma"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    PHI = "phi"
    GPT2 = "gpt2"  # For testing only - no safety training
    OTHER = "other"


class BlockType(str, Enum):
    """Canonical block/module types for indexing."""
    EMBED = "embed"
    ATTN = "attn"
    MLP = "mlp"
    NORM = "norm"
    LM_HEAD = "lm_head"


@dataclass
class DecodingConfig:
    """Decoding parameters for generation."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 256
    do_sample: bool = True
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def greedy(cls, max_new_tokens: int = 256) -> "DecodingConfig":
        """Create greedy decoding config for reproducible baselines."""
        return cls(
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            max_new_tokens=max_new_tokens
        )


@dataclass
class ModelConfig:
    """Model identification and configuration."""
    model_id: str
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    family: ModelFamily = ModelFamily.OTHER
    tokenizer_hash: Optional[str] = None
    
    # Capability flags
    has_attention_output: bool = True
    has_kv_cache_access: bool = True
    has_hidden_states: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["family"] = self.family.value
        return d
    
    @classmethod
    def from_model_id(cls, model_id: str, **kwargs) -> "ModelConfig":
        """Infer family and capabilities from model ID."""
        model_id_lower = model_id.lower()
        
        if "gemma" in model_id_lower:
            family = ModelFamily.GEMMA
        elif "llama" in model_id_lower:
            family = ModelFamily.LLAMA
        elif "mistral" in model_id_lower or "mixtral" in model_id_lower:
            family = ModelFamily.MISTRAL
        elif "qwen" in model_id_lower:
            family = ModelFamily.QWEN
        elif "phi" in model_id_lower:
            family = ModelFamily.PHI
        else:
            family = ModelFamily.OTHER
            
        return cls(model_id=model_id, family=family, **kwargs)


@dataclass
class RunConfig:
    """Configuration for an experiment run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    experiment_name: str = ""
    experiment_version: str = "1.0"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_id=""))
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    prompt_set_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "model": self.model.to_dict(),
            "decoding": self.decoding.to_dict(),
            "prompt_set_id": self.prompt_set_id,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "notes": self.notes,
        }


@dataclass
class BehaviorDelta:
    """Behavioral changes between baseline and adversarial outputs."""
    semantic_similarity: float = 1.0  # 0-1, embedding cosine similarity
    semantic_changed: bool = False    # True if similarity < threshold
    
    baseline_refused: bool = False
    adversarial_refused: bool = False
    refusal_flipped: bool = False     # True if refusal state changed
    
    baseline_complied: bool = False
    adversarial_complied: bool = False
    compliance_flipped: bool = False
    
    tool_call_changed: bool = False   # For agentic scenarios
    
    classification: str = "none"      # none, benign_variance, refusal_bypass, etc.
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class StateDelta:
    """Internal state changes during processing."""
    # Hidden state metrics
    hidden_drift_norm: float = 0.0           # ||h_adv - h_baseline||
    hidden_drift_cosine: float = 1.0         # cos(h_adv, h_baseline)
    max_layer_drift: float = 0.0
    max_drift_layer_idx: int = -1
    
    # Attention metrics
    attention_entropy_baseline: float = 0.0
    attention_entropy_adversarial: float = 0.0
    attention_entropy_delta: float = 0.0
    attention_rerouted: bool = False
    
    # KV cache metrics (where available)
    kv_norm_growth: float = 0.0
    kv_similarity: float = 1.0
    
    # Layer-specific tracking
    per_layer_drift: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StabilityDelta:
    """Output quality and stability metrics."""
    # Degeneracy detection
    repetition_ratio: float = 0.0      # Repeated n-gram ratio
    unique_ngram_ratio: float = 1.0    # Unique n-grams / total n-grams
    loop_detected: bool = False
    
    # Quality metrics
    coherence_score: float = 1.0       # 0-1, output coherence
    perplexity_proxy: float = 0.0      # If available
    
    # Numerical stability
    has_nan: bool = False
    has_inf: bool = False
    max_activation_norm: float = 0.0
    norm_spike_detected: bool = False
    
    # Output characteristics
    output_length: int = 0
    is_truncated: bool = False
    is_empty: bool = False
    is_gibberish: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleRecord:
    """A single sample/turn record in an experiment."""
    # Identifiers
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt_id: str = ""
    turn: int = 0
    
    # Inputs
    prompt: str = ""
    prompt_variant: str = "baseline"  # baseline, adversarial, perturbed, etc.
    perturbation_id: Optional[str] = None
    
    # Outputs
    output: str = ""
    output_hash: str = ""  # For deduplication
    
    # Metrics
    behavior: BehaviorDelta = field(default_factory=BehaviorDelta)
    state: StateDelta = field(default_factory=StateDelta)
    stability: StabilityDelta = field(default_factory=StabilityDelta)
    
    # Metadata
    generation_time_ms: float = 0.0
    token_count: int = 0
    
    def __post_init__(self):
        if self.output and not self.output_hash:
            self.output_hash = hashlib.md5(self.output.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt_id": self.prompt_id,
            "turn": self.turn,
            "prompt": self.prompt,
            "prompt_variant": self.prompt_variant,
            "perturbation_id": self.perturbation_id,
            "output": self.output,
            "output_hash": self.output_hash,
            "behavior": self.behavior.to_dict(),
            "state": self.state.to_dict(),
            "stability": self.stability.to_dict(),
            "generation_time_ms": self.generation_time_ms,
            "token_count": self.token_count,
        }
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL-compatible string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ExperimentResult:
    """Complete result of an experiment run."""
    config: RunConfig
    samples: List[SampleRecord] = field(default_factory=list)
    
    # Aggregate metrics
    total_samples: int = 0
    refusal_rate: float = 0.0
    compliance_rate: float = 0.0
    refusal_bypass_count: int = 0
    
    # Summary statistics
    mean_semantic_similarity: float = 1.0
    mean_hidden_drift: float = 0.0
    mean_coherence: float = 1.0
    
    # Flags
    any_nan_detected: bool = False
    any_collapse_detected: bool = False
    
    def compute_aggregates(self):
        """Compute aggregate metrics from samples."""
        if not self.samples:
            return
            
        self.total_samples = len(self.samples)
        
        refusals = sum(1 for s in self.samples if s.behavior.adversarial_refused)
        compliances = sum(1 for s in self.samples if s.behavior.adversarial_complied)
        bypasses = sum(1 for s in self.samples if s.behavior.refusal_flipped and not s.behavior.adversarial_refused)
        
        self.refusal_rate = refusals / self.total_samples if self.total_samples > 0 else 0
        self.compliance_rate = compliances / self.total_samples if self.total_samples > 0 else 0
        self.refusal_bypass_count = bypasses
        
        self.mean_semantic_similarity = sum(s.behavior.semantic_similarity for s in self.samples) / self.total_samples
        self.mean_hidden_drift = sum(s.state.hidden_drift_norm for s in self.samples) / self.total_samples
        self.mean_coherence = sum(s.stability.coherence_score for s in self.samples) / self.total_samples
        
        self.any_nan_detected = any(s.stability.has_nan for s in self.samples)
        self.any_collapse_detected = any(s.stability.is_gibberish or s.stability.loop_detected for s in self.samples)
    
    def to_dict(self) -> Dict[str, Any]:
        self.compute_aggregates()
        return {
            "config": self.config.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
            "aggregates": {
                "total_samples": self.total_samples,
                "refusal_rate": self.refusal_rate,
                "compliance_rate": self.compliance_rate,
                "refusal_bypass_count": self.refusal_bypass_count,
                "mean_semantic_similarity": self.mean_semantic_similarity,
                "mean_hidden_drift": self.mean_hidden_drift,
                "mean_coherence": self.mean_coherence,
                "any_nan_detected": self.any_nan_detected,
                "any_collapse_detected": self.any_collapse_detected,
            }
        }
    
    def save_jsonl(self, path: str):
        """Save samples as JSONL file (one record per line)."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "w") as f:
            # Write config as first line
            f.write(json.dumps({"_type": "config", **self.config.to_dict()}) + "\n")
            # Write each sample
            for sample in self.samples:
                f.write(sample.to_jsonl() + "\n")
    
    @classmethod
    def load_jsonl(cls, path: str) -> "ExperimentResult":
        """Load from JSONL file."""
        samples = []
        config = None
        
        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record.get("_type") == "config":
                    del record["_type"]
                    # Reconstruct config (simplified - would need full reconstruction)
                    config = RunConfig(
                        run_id=record.get("run_id", ""),
                        experiment_name=record.get("experiment_name", ""),
                    )
                else:
                    # Reconstruct sample (simplified)
                    sample = SampleRecord(
                        sample_id=record.get("sample_id", ""),
                        prompt=record.get("prompt", ""),
                        output=record.get("output", ""),
                    )
                    samples.append(sample)
        
        result = cls(config=config or RunConfig(), samples=samples)
        result.compute_aggregates()
        return result





