"""
Model Adapter - Standardized Model Loading and Capability Detection

Provides a unified interface for loading and interacting with different
model architectures. Handles capability detection (attention, KV-cache,
hidden states) for graceful degradation.

Red Team Value:
- Enables apples-to-apples comparison across model families
- Automatic capability detection for experiment selection
- Standardized generation interface
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .schema import ModelFamily, ModelConfig
from .logging import get_logger

logger = logging.getLogger("redteam.core.model_adapter")


class ModelCapability(Enum):
    """Capabilities that models may or may not support."""
    HIDDEN_STATES = "hidden_states"
    ATTENTIONS = "attentions"
    KV_CACHE = "kv_cache"
    CHAT_TEMPLATE = "chat_template"
    LOGITS = "logits"


@dataclass
class ModelCapabilities:
    """Detected capabilities for a model."""
    hidden_states: bool = False
    attentions: bool = False
    kv_cache: bool = False
    chat_template: bool = False
    logits: bool = True  # All causal LMs have this
    
    # Architecture info
    num_layers: int = 0
    num_heads: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_states": self.hidden_states,
            "attentions": self.attentions,
            "kv_cache": self.kv_cache,
            "chat_template": self.chat_template,
            "logits": self.logits,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }
    
    def supports(self, capability: ModelCapability) -> bool:
        """Check if model supports a capability."""
        return getattr(self, capability.value, False)


@dataclass
class GenerationResult:
    """Standardized generation result."""
    text: str
    input_tokens: int
    output_tokens: int
    
    # Optional outputs
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    past_key_values: Optional[Any] = None
    logits: Optional[Any] = None
    
    # Metadata
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_time_ms": self.generation_time_ms,
            "has_hidden_states": self.hidden_states is not None,
            "has_attentions": self.attentions is not None,
            "has_kv_cache": self.past_key_values is not None,
        }


class ModelAdapter:
    """
    Unified model adapter for red team experiments.
    
    Handles:
    - Model loading with appropriate dtype/device
    - Capability detection
    - Standardized generation interface
    - Chat template handling
    
    Usage:
        adapter = ModelAdapter.load("google/gemma-2-2b-it")
        result = adapter.generate("Hello!", return_hidden_states=True)
    """
    
    # Known model families for better defaults
    KNOWN_FAMILIES = {
        "gemma": ModelFamily.GEMMA,
        "llama": ModelFamily.LLAMA,
        "mistral": ModelFamily.MISTRAL,
        "qwen": ModelFamily.QWEN,
        "phi": ModelFamily.PHI,
        "gpt2": ModelFamily.GPT2,
    }
    
    def __init__(
        self,
        model,
        tokenizer,
        config: ModelConfig,
        capabilities: ModelCapabilities,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.capabilities = capabilities
        self._logger = get_logger("model_adapter")
    
    @classmethod
    def load(
        cls,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "ModelAdapter":
        """
        Load a model with automatic configuration.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ("auto", "cuda", "mps", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional args for model loading
            
        Returns:
            ModelAdapter instance
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger = get_logger("model_adapter")
        logger.info(f"Loading model: {model_id}")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Determine dtype
        if dtype == "auto":
            if device in ["cuda", "mps"]:
                dtype = "bfloat16"
            else:
                dtype = "float32"
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)
        
        logger.info(f"Using device: {device}, dtype: {dtype}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }
        
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        if device == "mps":
            model = model.to(device)
        
        model.eval()
        
        # Detect model family
        model_family = ModelFamily.OTHER
        model_id_lower = model_id.lower()
        for name, family in cls.KNOWN_FAMILIES.items():
            if name in model_id_lower:
                model_family = family
                break
        
        # Create config
        config = ModelConfig(
            model_id=model_id,
            family=model_family,
            dtype=dtype,
            device=device,
            num_layers=getattr(model.config, 'num_hidden_layers', 0),
            hidden_size=getattr(model.config, 'hidden_size', 0),
        )
        
        # Detect capabilities
        capabilities = cls._detect_capabilities(model, tokenizer)
        
        logger.info(f"Model loaded: {config.num_layers} layers, {capabilities.num_heads} heads")
        logger.info(f"Capabilities: hidden_states={capabilities.hidden_states}, "
                   f"attentions={capabilities.attentions}, kv_cache={capabilities.kv_cache}")
        
        return cls(model, tokenizer, config, capabilities)
    
    @classmethod
    def _detect_capabilities(
        cls,
        model,
        tokenizer,
    ) -> ModelCapabilities:
        """Detect model capabilities by running test inference."""
        import torch
        
        caps = ModelCapabilities()
        
        # Get architecture info from config
        config = model.config
        caps.num_layers = getattr(config, 'num_hidden_layers', 0)
        caps.num_heads = getattr(config, 'num_attention_heads', 0)
        caps.hidden_size = getattr(config, 'hidden_size', 0)
        caps.vocab_size = getattr(config, 'vocab_size', 0)
        
        # Check chat template
        caps.chat_template = hasattr(tokenizer, 'apply_chat_template')
        
        # Test inference to check capabilities
        try:
            test_input = tokenizer("test", return_tensors="pt")
            if hasattr(model, 'device'):
                test_input = {k: v.to(model.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                output = model(
                    **test_input,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=True,
                    return_dict=True,
                )
            
            # Check hidden states
            caps.hidden_states = (
                hasattr(output, 'hidden_states') and 
                output.hidden_states is not None
            )
            
            # Check attentions
            caps.attentions = (
                hasattr(output, 'attentions') and 
                output.attentions is not None
            )
            
            # Check KV cache
            caps.kv_cache = (
                hasattr(output, 'past_key_values') and 
                output.past_key_values is not None
            )
            
        except Exception as e:
            logger.warning(f"Capability detection failed: {e}")
        
        return caps
    
    def format_prompt(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format messages using chat template or fallback."""
        if self.capabilities.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # Fallback format
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_kv_cache: bool = False,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate response with optional internal state capture.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            return_hidden_states: Whether to return hidden states
            return_attentions: Whether to return attention weights
            return_kv_cache: Whether to return KV cache
            seed: Random seed for reproducibility
            
        Returns:
            GenerationResult with text and optional internal states
        """
        import torch
        import time
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        
        result = GenerationResult(
            text="",
            input_tokens=input_length,
            output_tokens=0,
        )
        
        try:
            start_time = time.time()
            
            with torch.no_grad():
                # First, get internal states if requested
                if return_hidden_states or return_attentions or return_kv_cache:
                    forward_output = self.model(
                        **inputs,
                        output_hidden_states=return_hidden_states and self.capabilities.hidden_states,
                        output_attentions=return_attentions and self.capabilities.attentions,
                        use_cache=return_kv_cache and self.capabilities.kv_cache,
                        return_dict=True,
                    )
                    
                    if return_hidden_states and self.capabilities.hidden_states:
                        result.hidden_states = forward_output.hidden_states
                    
                    if return_attentions and self.capabilities.attentions:
                        result.attentions = forward_output.attentions
                    
                    if return_kv_cache and self.capabilities.kv_cache:
                        result.past_key_values = forward_output.past_key_values
                    
                    result.logits = forward_output.logits
                
                # Generate response
                gen_output = self.model.generate(**inputs, **gen_kwargs)
            
            generation_time = time.time() - start_time
            
            # Decode
            output_ids = gen_output[0][input_length:]
            result.text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            result.output_tokens = len(output_ids)
            result.generation_time_ms = generation_time * 1000
            
        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            result.text = f"[ERROR: {e}]"
        
        return result
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """Generate response for a chat conversation."""
        prompt = self.format_prompt(messages)
        return self.generate(prompt, **kwargs)
    
    def get_layer_names(self) -> List[str]:
        """Get names of transformer layers for hooking."""
        layer_names = []
        
        for name, module in self.model.named_modules():
            # Look for transformer blocks
            if any(pattern in name for pattern in [
                '.layers.', '.h.', '.blocks.', '.decoder.layers.'
            ]):
                # Skip sub-modules, only get the block itself
                parts = name.split('.')
                if parts[-1].isdigit():
                    layer_names.append(name)
        
        return layer_names
    
    def __repr__(self) -> str:
        return (
            f"ModelAdapter(model_id={self.config.model_id}, "
            f"family={self.config.family.value}, "
            f"layers={self.capabilities.num_layers}, "
            f"device={self.config.device})"
        )


def load_model(model_id: str, **kwargs) -> ModelAdapter:
    """Convenience function to load a model."""
    return ModelAdapter.load(model_id, **kwargs)
