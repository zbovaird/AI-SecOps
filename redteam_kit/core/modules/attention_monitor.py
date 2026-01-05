"""
Attention & KV-Cache Monitor Module
Captures Q, K, V vectors and tracks attention distributions

Usage for Red Teaming:
---------------------
This module monitors attention mechanisms to identify heads that amplify
adversarial signals, track KV-cache evolution, and measure attention-induced
distortions in latent space.

Example Usage:
    from core.modules.attention_monitor import AttentionMonitor
    
    monitor = AttentionMonitor(model)
    monitor.register_attention_hooks()
    
    # Run inference
    outputs = model(inputs)
    
    # Get attention data
    attention_data = monitor.get_attention_data()
    kv_cache = monitor.get_kv_cache()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
from ..utils import safe_detach_clone, ensure_numpy_compatible


class AttentionMonitor:
    """Monitor attention mechanisms and KV-cache"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention monitor
        
        Args:
            model: PyTorch model to monitor
        """
        self.model = model
        
        # Storage
        self.qkv_data: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        self.attention_weights: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.kv_cache: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.attention_outputs: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        # Hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Metadata
        self.layer_names: List[str] = []
        self.num_heads: Dict[str, int] = {}
        self.head_dim: Dict[str, int] = {}
    
    def register_attention_hooks(self):
        """Register hooks for all attention layers"""
        for name, module in self.model.named_modules():
            # Check if it's an attention module
            if self._is_attention_module(module):
                self._register_attention_hook(name, module)
                self.layer_names.append(name)
    
    def _is_attention_module(self, module: nn.Module) -> bool:
        """Check if module is an attention module"""
        module_type = type(module).__name__.lower()
        return (
            'attention' in module_type or
            'attn' in module_type or
            hasattr(module, 'qkv') or
            hasattr(module, 'in_proj_weight') or
            isinstance(module, nn.MultiheadAttention)
        )
    
    def _register_attention_hook(self, name: str, module: nn.Module):
        """Register hook for a specific attention module with safe None handling"""
        def forward_hook(module, input, output):
            # Try to extract Q, K, V (optional, may fail for some architectures)
            try:
                self._extract_qkv(name, module, input)
            except Exception:
                pass  # QKV extraction is optional
            
            # Extract attention weights and outputs safely
            if isinstance(output, tuple):
                if len(output) > 0:
                    output_tensor = safe_detach_clone(output[0])
                    if output_tensor is not None:
                        self.attention_outputs[name].append(output_tensor)
                
                # Only try to get weights if they exist and are not None
                if len(output) > 1:
                    weight_tensor = safe_detach_clone(output[1])
                    if weight_tensor is not None:
                        self.attention_weights[name].append(weight_tensor)
            else:
                output_tensor = safe_detach_clone(output)
                if output_tensor is not None:
                    self.attention_outputs[name].append(output_tensor)
        
        hook = module.register_forward_hook(forward_hook)
        self.hooks.append(hook)
    
    def _extract_qkv(self, name: str, module: nn.Module, input: Tuple):
        """Extract Q, K, V vectors from attention module"""
        # Method 1: MultiheadAttention with separate projections
        if isinstance(module, nn.MultiheadAttention):
            if len(input) > 0:
                query = input[0]
                if len(input) > 1:
                    key = input[1]
                else:
                    key = query
                if len(input) > 2:
                    value = input[2]
                else:
                    value = query
                
                # Project to Q, K, V
                if hasattr(module, 'in_proj_weight'):
                    # Combined projection
                    qkv = torch.nn.functional.linear(query, module.in_proj_weight, module.in_proj_bias)
                    head_dim = module.embed_dim // module.num_heads
                    qkv = qkv.view(query.shape[0], query.shape[1], 3, module.num_heads, head_dim)
                    q, k, v = qkv.chunk(3, dim=2)
                    q = q.squeeze(2)
                    k = k.squeeze(2)
                    v = v.squeeze(2)
                else:
                    # Separate projections
                    q = torch.nn.functional.linear(query, module.q_proj_weight, module.q_proj_bias)
                    k = torch.nn.functional.linear(key, module.k_proj_weight, module.k_proj_bias)
                    v = torch.nn.functional.linear(value, module.v_proj_weight, module.v_proj_bias)
                
                self.qkv_data[name]['q'].append(q.detach().clone())
                self.qkv_data[name]['k'].append(k.detach().clone())
                self.qkv_data[name]['v'].append(v.detach().clone())
                
                self.num_heads[name] = module.num_heads
                self.head_dim[name] = module.embed_dim // module.num_heads
        
        # Method 2: Custom attention with qkv attribute
        elif hasattr(module, 'qkv'):
            qkv = module.qkv(input[0] if isinstance(input, tuple) else input)
            # Assume shape is (batch, seq, 3, heads, head_dim)
            if len(qkv.shape) == 5:
                q, k, v = qkv.chunk(3, dim=2)
                q = q.squeeze(2)
                k = k.squeeze(2)
                v = v.squeeze(2)
            else:
                # Try other shapes
                qkv_flat = qkv.view(qkv.shape[0], qkv.shape[1], 3, -1)
                q, k, v = qkv_flat.chunk(3, dim=2)
            
            self.qkv_data[name]['q'].append(q.detach().clone())
            self.qkv_data[name]['k'].append(k.detach().clone())
            self.qkv_data[name]['v'].append(v.detach().clone())
    
    def get_attention_data(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get captured attention data
        
        Args:
            layer_name: Optional specific layer name
        
        Returns:
            Dictionary with Q, K, V, attention weights, and outputs
        """
        if layer_name:
            return {
                'qkv': dict(self.qkv_data.get(layer_name, {})),
                'attention_weights': self.attention_weights.get(layer_name, []),
                'outputs': self.attention_outputs.get(layer_name, [])
            }
        
        return {
            'qkv': {name: dict(data) for name, data in self.qkv_data.items()},
            'attention_weights': dict(self.attention_weights),
            'outputs': dict(self.attention_outputs)
        }
    
    def get_kv_cache(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get KV-cache data
        
        Args:
            layer_name: Optional specific layer name
        
        Returns:
            Dictionary with KV-cache
        """
        if layer_name:
            return dict(self.kv_cache.get(layer_name, {}))
        return dict(self.kv_cache)
    
    def update_kv_cache(self, layer_name: str, key: torch.Tensor, value: torch.Tensor):
        """
        Update KV-cache for a layer (for multi-turn conversations)
        
        Args:
            layer_name: Layer name
            key: Key tensor
            value: Value tensor
        """
        if layer_name not in self.kv_cache:
            self.kv_cache[layer_name] = {'key': key.clone(), 'value': value.clone()}
        else:
            # Concatenate with existing cache
            self.kv_cache[layer_name]['key'] = torch.cat([
                self.kv_cache[layer_name]['key'],
                key.clone()
            ], dim=-2)
            self.kv_cache[layer_name]['value'] = torch.cat([
                self.kv_cache[layer_name]['value'],
                value.clone()
            ], dim=-2)
    
    def analyze_attention_distribution(
        self,
        attention_weights: torch.Tensor,
        head_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Analyze attention weight distribution
        
        Args:
            attention_weights: Attention weight tensor (batch, heads, seq, seq) or (batch, seq, seq)
            head_idx: Optional specific head index
        
        Returns:
            Dictionary with distribution metrics
        """
        # Handle different shapes
        if len(attention_weights.shape) == 4:
            # (batch, heads, seq, seq)
            if head_idx is not None:
                weights = attention_weights[:, head_idx, :, :]
            else:
                weights = attention_weights.mean(dim=1)  # Average over heads
        elif len(attention_weights.shape) == 3:
            # (batch, seq, seq)
            weights = attention_weights
        else:
            raise ValueError(f"Unexpected attention weight shape: {attention_weights.shape}")
        
        # Flatten for analysis (ensure numpy compatible)
        weights_flat = ensure_numpy_compatible(weights.flatten()).detach().cpu().numpy()
        
        # Compute metrics
        mean_attention = float(weights_flat.mean())
        max_attention = float(weights_flat.max())
        min_attention = float(weights_flat.min())
        
        # Entropy of attention distribution
        # Normalize to probability distribution
        weights_normalized = weights_flat + 1e-10
        weights_normalized = weights_normalized / weights_normalized.sum()
        attention_entropy = float(entropy(weights_normalized))
        
        # Peakiness (concentration measure)
        # Higher values = more peaked
        peakiness = float((weights_flat ** 2).sum())
        
        return {
            'mean': mean_attention,
            'max': max_attention,
            'min': min_attention,
            'entropy': attention_entropy,
            'peakiness': peakiness
        }
    
    def identify_susceptible_heads(
        self,
        layer_name: str,
        entropy_threshold: float = 1.0,
        peakiness_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Identify attention heads that may be susceptible to adversarial signals
        
        Args:
            layer_name: Layer name to analyze
            entropy_threshold: Threshold for low entropy (peaked attention)
            peakiness_threshold: Threshold for high peakiness
        
        Returns:
            List of susceptible head information
        """
        if layer_name not in self.attention_weights:
            return []
        
        susceptible_heads = []
        attention_weights_list = self.attention_weights[layer_name]
        
        if not attention_weights_list:
            return []
        
        # Analyze each head
        num_heads = self.num_heads.get(layer_name, 1)
        
        for head_idx in range(num_heads):
            head_metrics = []
            
            for attn_weights in attention_weights_list:
                metrics = self.analyze_attention_distribution(attn_weights, head_idx=head_idx)
                head_metrics.append(metrics)
            
            # Average metrics across captures
            avg_entropy = np.mean([m['entropy'] for m in head_metrics])
            avg_peakiness = np.mean([m['peakiness'] for m in head_metrics])
            
            is_susceptible = (
                avg_entropy < entropy_threshold or
                avg_peakiness > peakiness_threshold
            )
            
            if is_susceptible:
                susceptible_heads.append({
                    'layer_name': layer_name,
                    'head_idx': head_idx,
                    'avg_entropy': float(avg_entropy),
                    'avg_peakiness': float(avg_peakiness),
                    'metrics': head_metrics
                })
        
        return susceptible_heads
    
    def clear_data(self):
        """Clear all captured data"""
        self.qkv_data.clear()
        self.attention_weights.clear()
        self.kv_cache.clear()
        self.attention_outputs.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks"""
        self.remove_hooks()






