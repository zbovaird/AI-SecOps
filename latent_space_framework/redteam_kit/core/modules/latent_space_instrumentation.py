"""
Latent Space Instrumentation Module
Provides hooks for capturing activations, gradients, and KV-cache at each layer

Usage for Red Teaming:
---------------------
This module instruments transformer models to capture internal representations
at every layer, enabling latent space analysis, vulnerability detection, and
collapse induction experiments.

Example Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from redteam_kit.core.modules.latent_space_instrumentation import ModelInstrumentation
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    # Instrument model
    instrumentation = ModelInstrumentation(model, tokenizer=tokenizer, device="cuda")
    instrumentation.register_all_hooks()
    
    # Run inference
    inputs = tokenizer("Test prompt", return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    
    # Access activations
    activations = instrumentation.get_activations()
    print(f"Captured {len(activations)} layer activations")
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from collections import defaultdict
import h5py
import os
from pathlib import Path
from ..utils import ensure_numpy_compatible


class ModelInstrumentation:
    """Instrument model to capture activations, gradients, and KV-cache"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        storage_path: Optional[str] = None,
        capture_gradients: bool = False,
        compress_storage: bool = True
    ):
        """
        Initialize model instrumentation
        
        Args:
            model: PyTorch model to instrument
            tokenizer: Optional tokenizer for decoding/encoding
            device: Optional device (e.g., 'cuda', 'cpu')
            storage_path: Optional path for HDF5 storage
            capture_gradients: Whether to capture gradients (requires requires_grad=True)
            compress_storage: Whether to use compression in HDF5 storage
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.storage_path = storage_path
        self.capture_gradients = capture_gradients
        self.compress_storage = compress_storage
        
        # Storage for activations
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.gradients: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # KV-cache tracking
        self.kv_cache: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.attention_outputs: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        
        # Metadata
        self.layer_names: List[str] = []
        self.capture_count = 0
        
    def register_all_hooks(self, layer_types: Optional[List[type]] = None):
        """
        Register forward hooks for all layers
        
        Args:
            layer_types: Optional list of layer types to hook (default: all)
        """
        if layer_types is None:
            layer_types = [
                nn.Embedding,
                nn.Linear,
                nn.LayerNorm,
                nn.MultiheadAttention,
            ]
        
        # Also hook transformer blocks if available
        for name, module in self.model.named_modules():
            # Skip if already registered
            if any(name in hook_name for hook_name in self.layer_names):
                continue
                
            # Register hook for matching layer types
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                self._register_forward_hook(name, module)
            
            # Special handling for transformer blocks (common patterns)
            if 'transformer' in name.lower() or 'block' in name.lower():
                self._register_forward_hook(name, module)
            
            # Hook attention layers specifically
            if 'attention' in name.lower() or 'attn' in name.lower():
                self._register_attention_hook(name, module)
        
        # Hook model forward for overall tracking
        self._register_model_forward_hook()
    
    def _register_forward_hook(self, name: str, module: nn.Module):
        """Register forward hook for a specific module with safe handling"""
        def hook_fn(module, input, output):
            try:
                # Handle None outputs
                if output is None:
                    return
                
                # Handle tuple outputs (common in transformers)
                if isinstance(output, tuple):
                    if len(output) == 0:
                        return
                    # Usually (hidden_states, ...) or (hidden_states, attention_outputs, ...)
                    activation = output[0]
                    if activation is None:
                        return
                    activation = activation.detach().clone()
                else:
                    activation = output.detach().clone()
                
                # Convert bfloat16 to float32 for compatibility
                if activation.dtype == torch.bfloat16:
                    activation = activation.float()
                
                self.activations[name].append(activation)
                if name not in self.layer_names:
                    self.layer_names.append(name)
                
                # Store metadata
                if self.storage_path:
                    self._save_activation_to_disk(name, activation)
            except Exception:
                # Silently skip problematic hooks
                pass
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def _register_attention_hook(self, name: str, module: nn.Module):
        """Register hook specifically for attention mechanisms with safe None handling"""
        def attention_hook_fn(module, input, output):
            try:
                # Handle None outputs
                if output is None:
                    return
                
                # Capture Q, K, V if available
                if hasattr(module, 'qkv') or hasattr(module, 'in_proj_weight'):
                    # Try to extract Q, K, V from input
                    if isinstance(input, tuple) and len(input) > 0:
                        hidden_states = input[0]
                        if hidden_states is not None:
                            # Convert bfloat16 if needed
                            if hidden_states.dtype == torch.bfloat16:
                                hidden_states = hidden_states.float()
                            self.attention_outputs[name]['hidden_states'] = hidden_states.detach().clone()
                
                # Capture attention output safely
                if isinstance(output, tuple):
                    # Usually (attn_output, attn_weights, ...)
                    if len(output) > 0 and output[0] is not None:
                        attn_output = output[0]
                        if attn_output.dtype == torch.bfloat16:
                            attn_output = attn_output.float()
                        self.attention_outputs[name]['output'] = attn_output.detach().clone()
                    if len(output) > 1 and output[1] is not None:
                        attn_weights = output[1]
                        if attn_weights.dtype == torch.bfloat16:
                            attn_weights = attn_weights.float()
                        self.attention_outputs[name]['weights'] = attn_weights.detach().clone()
                else:
                    if output is not None:
                        if output.dtype == torch.bfloat16:
                            output = output.float()
                        self.attention_outputs[name]['output'] = output.detach().clone()
            except Exception:
                # Silently skip problematic hooks
                pass
        
        hook = module.register_forward_hook(attention_hook_fn)
        self.hooks.append(hook)
    
    def _register_model_forward_hook(self):
        """Register hook at model level for overall tracking"""
        def model_hook_fn(module, input, output):
            self.capture_count += 1
            # Store input/output shapes for debugging
            if not hasattr(self, 'input_shapes'):
                self.input_shapes = []
            if isinstance(input, tuple):
                self.input_shapes.append([inp.shape if hasattr(inp, 'shape') else None for inp in input])
        
        hook = self.model.register_forward_hook(model_hook_fn)
        self.hooks.append(hook)
    
    def register_gradient_hooks(self):
        """Register backward hooks for gradient capture"""
        if not self.capture_gradients:
            raise ValueError("capture_gradients must be True to register gradient hooks")
        
        def gradient_hook_fn(name):
            def hook(module, grad_input, grad_output):
                if grad_output is not None:
                    if isinstance(grad_output, tuple):
                        grad = grad_output[0].detach().clone()
                    else:
                        grad = grad_output.detach().clone()
                    self.gradients[name].append(grad)
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_full_backward_hook(gradient_hook_fn(name))
                self.hooks.append(hook)
    
    def _save_activation_to_disk(self, name: str, activation: torch.Tensor):
        """Save activation to HDF5 file"""
        if not self.storage_path:
            return
        
        # Create directory if needed
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy (with bfloat16 support)
        if isinstance(activation, torch.Tensor):
            activation_np = ensure_numpy_compatible(activation).cpu().numpy()
        else:
            activation_np = np.array(activation)
        
        # Save to HDF5
        with h5py.File(self.storage_path, 'a') as f:
            dataset_name = f"{name}/activation_{self.capture_count}"
            if self.compress_storage:
                f.create_dataset(
                    dataset_name,
                    data=activation_np,
                    compression='gzip',
                    compression_opts=9
                )
            else:
                f.create_dataset(dataset_name, data=activation_np)
    
    def get_activations(self, layer_name: Optional[str] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Get captured activations
        
        Args:
            layer_name: Optional specific layer name, otherwise returns all
        
        Returns:
            Dictionary of layer names to activation lists
        """
        if layer_name:
            return {layer_name: self.activations.get(layer_name, [])}
        return dict(self.activations)
    
    def get_gradients(self, layer_name: Optional[str] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Get captured gradients
        
        Args:
            layer_name: Optional specific layer name, otherwise returns all
        
        Returns:
            Dictionary of layer names to gradient lists
        """
        if layer_name:
            return {layer_name: self.gradients.get(layer_name, [])}
        return dict(self.gradients)
    
    def get_attention_outputs(self, layer_name: Optional[str] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get captured attention outputs
        
        Args:
            layer_name: Optional specific layer name, otherwise returns all
        
        Returns:
            Dictionary of attention outputs per layer
        """
        if layer_name:
            return {layer_name: self.attention_outputs.get(layer_name, {})}
        return dict(self.attention_outputs)
    
    def clear_activations(self):
        """Clear all stored activations"""
        self.activations.clear()
        self.gradients.clear()
        self.attention_outputs.clear()
        self.kv_cache.clear()
        self.capture_count = 0
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def remove_all_hooks(self):
        """
        Remove all hooks including those registered directly on model modules.
        This ensures a clean state when re-instrumenting models.
        """
        # Remove hooks registered through this instrumentation
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        self.hooks.clear()
        
        # Also clear hooks from model modules directly
        for name, module in self.model.named_modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()
        
        # Reset internal state
        self.activations.clear()
        self.gradients.clear()
        self.attention_outputs.clear()
        self.kv_cache.clear()
        self.layer_names.clear()
        self.capture_count = 0
    
    def get_layer_names(self) -> List[str]:
        """Get list of all instrumented layer names"""
        return list(set(self.layer_names))
    
    def get_layer_statistics(self, layer_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific layer
        
        Args:
            layer_name: Name of the layer
        
        Returns:
            Dictionary with statistics (mean, std, shape, etc.)
        """
        if layer_name not in self.activations:
            return {}
        
        activations = self.activations[layer_name]
        if not activations:
            return {}
        
        # Concatenate all activations for this layer
        stacked = torch.cat([a.flatten() for a in activations])
        
        stats = {
            'mean': float(stacked.mean().item()),
            'std': float(stacked.std().item()),
            'min': float(stacked.min().item()),
            'max': float(stacked.max().item()),
            'shape': list(activations[0].shape),
            'num_captures': len(activations),
            'dtype': str(activations[0].dtype)
        }
        
        return stats
    
    def export_activations(self, output_path: str, format: str = "numpy"):
        """
        Export activations to disk
        
        Args:
            output_path: Path to save activations
            format: Format to save ('numpy', 'hdf5', 'pt')
        """
        if format == 'numpy':
            np.savez_compressed(output_path, **{
                name: [ensure_numpy_compatible(a).cpu().numpy() for a in acts]
                for name, acts in self.activations.items()
            })
        elif format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                for name, acts in self.activations.items():
                    grp = f.create_group(name)
                    for i, act in enumerate(acts):
                        grp.create_dataset(
                            f'activation_{i}',
                            data=ensure_numpy_compatible(act).cpu().numpy(),
                            compression='gzip' if self.compress_storage else None
                        )
        elif format == 'pt':
            torch.save({
                name: [a.cpu() for a in acts]
                for name, acts in self.activations.items()
            }, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

