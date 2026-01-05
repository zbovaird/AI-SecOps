"""
Multi-Model Loader with Hidden State Extraction

Loads open-weight models from HuggingFace and extracts hidden states
for CKA analysis. Supports memory-efficient sequential loading.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import gc

logger = logging.getLogger("transferability.model_loader")


def ensure_numpy_compatible(tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert tensor to float32 if it's bfloat16 or other unsupported types for NumPy.
    
    Args:
        tensor: Input tensor or array
        
    Returns:
        Tensor compatible with NumPy operations
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.float()
        # Also handle other potentially problematic types
        if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
            try:
                # Try to convert to float32
                return tensor.float()
            except:
                pass
    return tensor


@dataclass
class OpenModelConfig:
    """Configuration for an open-weight model."""
    model_id: str
    short_name: str
    num_layers: int = 0  # Auto-detected
    hidden_size: int = 0  # Auto-detected
    dtype: str = "bfloat16"
    
    # HuggingFace specific
    trust_remote_code: bool = False
    use_auth_token: bool = True


# Pre-configured models for the framework
SUPPORTED_MODELS: Dict[str, OpenModelConfig] = {
    "gemma2": OpenModelConfig(
        model_id="google/gemma-2-2b-it",
        short_name="gemma2",
        trust_remote_code=False,
    ),
    "mistral": OpenModelConfig(
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        short_name="mistral",
        trust_remote_code=False,
    ),
    "llama": OpenModelConfig(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        short_name="llama",
        trust_remote_code=False,
    ),
    "phi": OpenModelConfig(
        model_id="microsoft/phi-2",
        short_name="phi",
        trust_remote_code=True,
    ),
    "qwen": OpenModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        short_name="qwen",
        trust_remote_code=True,
    ),
}


@dataclass
class HiddenStateResult:
    """Result of hidden state extraction."""
    model_id: str
    short_name: str
    prompts: List[str]
    
    # layer_idx -> np.ndarray of shape (n_prompts, hidden_size)
    layer_states: Dict[int, Any] = field(default_factory=dict)
    
    # Model info
    num_layers: int = 0
    hidden_size: int = 0
    
    def to_numpy(self) -> Dict[int, Any]:
        """Convert all tensors to numpy arrays."""
        return {
            layer: ensure_numpy_compatible(states).cpu().numpy() if torch.is_tensor(states) else states
            for layer, states in self.layer_states.items()
        }


class ModelLoader:
    """
    Multi-model loader for CKA transferability analysis.
    
    Features:
    - Sequential model loading (memory efficient)
    - Hidden state extraction from all layers
    - Support for various model architectures
    - Automatic cleanup between models
    """
    
    def __init__(
        self,
        device: str = "auto",
        dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize model loader.
        
        Args:
            device: Device to load models on ("auto", "cuda", "cpu")
            dtype: Data type ("bfloat16", "float16", "float32")
            hf_token: HuggingFace authentication token
        """
        self.hf_token = hf_token
        self.dtype_str = dtype
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set dtype
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        
        # Current loaded model (for cleanup)
        self._current_model = None
        self._current_tokenizer = None
        
        logger.info(f"ModelLoader initialized: device={self.device}, dtype={dtype}")
    
    def _cleanup(self):
        """Clean up current model to free memory."""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
        if self._current_tokenizer is not None:
            del self._current_tokenizer
            self._current_tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_model(
        self,
        config: OpenModelConfig,
    ) -> Tuple[Any, Any]:
        """
        Load a single model and tokenizer.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Cleanup previous model
        self._cleanup()
        
        logger.info(f"Loading model: {config.model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            token=self.hf_token if config.use_auth_token else None,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        # Use device_map="auto" for GPU (automatically handles multi-GPU), None for CPU
        device_map = "auto" if self.device != "cpu" else None
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=config.trust_remote_code,
            token=self.hf_token if config.use_auth_token else None,
        )
        
        # Only move to CPU if explicitly requested (device_map=None already handles CPU)
        if self.device == "cpu" and device_map is None:
            model = model.to(self.device)
        
        model.eval()
        
        # Update config with detected values
        if hasattr(model.config, 'num_hidden_layers'):
            config.num_layers = model.config.num_hidden_layers
        if hasattr(model.config, 'hidden_size'):
            config.hidden_size = model.config.hidden_size
        
        logger.info(f"  Loaded: {config.num_layers} layers, {config.hidden_size} hidden size")
        
        self._current_model = model
        self._current_tokenizer = tokenizer
        
        return model, tokenizer
    
    def extract_hidden_states(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        config: OpenModelConfig,
        layers: Optional[List[int]] = None,
        pooling: str = "last",
    ) -> HiddenStateResult:
        """
        Extract hidden states from all specified layers.
        
        Args:
            model: The loaded model
            tokenizer: The tokenizer
            prompts: List of prompts to process
            config: Model configuration
            layers: Specific layers to extract (None = all)
            pooling: How to pool sequence dimension ("last", "mean", "max")
            
        Returns:
            HiddenStateResult with layer states
        """
        result = HiddenStateResult(
            model_id=config.model_id,
            short_name=config.short_name,
            prompts=prompts,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
        )
        
        if layers is None:
            layers = list(range(config.num_layers))
        
        logger.info(f"Extracting hidden states for {len(prompts)} prompts, {len(layers)} layers")
        
        all_hidden_states = {layer: [] for layer in layers}
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Forward pass with hidden states
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
                
                # Extract specified layers
                for layer_idx in layers:
                    if layer_idx < len(hidden_states):
                        layer_hidden = hidden_states[layer_idx][0]  # Remove batch dim
                        
                        # Pool across sequence dimension
                        if pooling == "last":
                            pooled = layer_hidden[-1]  # Last token
                        elif pooling == "mean":
                            pooled = layer_hidden.mean(dim=0)
                        elif pooling == "max":
                            pooled = layer_hidden.max(dim=0)[0]
                        else:
                            pooled = layer_hidden[-1]
                        
                        all_hidden_states[layer_idx].append(pooled.cpu())
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(prompts)} prompts")
        
        # Stack into tensors
        for layer_idx in layers:
            if all_hidden_states[layer_idx]:
                result.layer_states[layer_idx] = torch.stack(all_hidden_states[layer_idx])
        
        return result
    
    def load_and_extract(
        self,
        model_name: str,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        pooling: str = "last",
    ) -> HiddenStateResult:
        """
        Convenience method to load model and extract hidden states in one call.
        
        Args:
            model_name: Key in SUPPORTED_MODELS (e.g., "gemma2", "mistral")
            prompts: List of prompts
            layers: Specific layers (None = all)
            pooling: Pooling method
            
        Returns:
            HiddenStateResult
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")
        
        config = SUPPORTED_MODELS[model_name]
        model, tokenizer = self.load_model(config)
        
        result = self.extract_hidden_states(
            model, tokenizer, prompts, config, layers, pooling
        )
        
        return result
    
    def load_multiple_models(
        self,
        model_names: List[str],
        prompts: List[str],
        layers_per_model: Optional[Dict[str, List[int]]] = None,
        pooling: str = "last",
    ) -> Dict[str, HiddenStateResult]:
        """
        Load multiple models sequentially and extract hidden states.
        
        Memory efficient: only one model loaded at a time.
        
        Args:
            model_names: List of model keys
            prompts: List of prompts (same for all models)
            layers_per_model: Optional dict specifying layers per model
            pooling: Pooling method
            
        Returns:
            Dict mapping model_name -> HiddenStateResult
        """
        results = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*50}")
            
            layers = None
            if layers_per_model and model_name in layers_per_model:
                layers = layers_per_model[model_name]
            
            result = self.load_and_extract(model_name, prompts, layers, pooling)
            results[model_name] = result
            
            # Cleanup before next model
            self._cleanup()
        
        return results
    
    def get_supported_models(self) -> Dict[str, str]:
        """Get dictionary of supported model names and their HuggingFace IDs."""
        return {name: config.model_id for name, config in SUPPORTED_MODELS.items()}


def sample_layers(num_layers: int, n_samples: int = 5) -> List[int]:
    """
    Sample evenly spaced layers for efficient CKA computation.
    
    Args:
        num_layers: Total number of layers
        n_samples: Number of layers to sample
        
    Returns:
        List of layer indices
    """
    if num_layers <= n_samples:
        return list(range(num_layers))
    
    step = num_layers / n_samples
    return [int(i * step) for i in range(n_samples)]
