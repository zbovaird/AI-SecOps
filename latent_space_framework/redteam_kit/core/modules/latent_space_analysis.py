"""
Latent Space Analysis Module
Computes variance, entropy, SVD, Jacobians, and identifies vulnerability basins

Usage for Red Teaming:
---------------------
This module analyzes latent representations to identify vulnerability basins,
track dimensionality collapse, and compute sensitivity metrics for collapse
induction experiments.

Example Usage:
    from core.modules.latent_space_analysis import LatentSpaceAnalyzer
    
    analyzer = LatentSpaceAnalyzer()
    
    # Analyze layer activations
    stats = analyzer.analyze_layer(activations)
    print(f"Variance: {stats['variance']}, Entropy: {stats['entropy']}")
    
    # Compute Jacobian
    jacobian = analyzer.compute_jacobian(model, layer_name, inputs)
    
    # Identify vulnerability basins
    basins = analyzer.identify_vulnerability_basins(all_layer_stats)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.linalg import svd
from scipy.stats import entropy
import warnings
import json
import os
import inspect
from pathlib import Path
from datetime import datetime
from ..utils import ensure_numpy_compatible, convert_to_native


class LatentSpaceAnalyzer:
    """Analyze latent space properties and identify vulnerabilities"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize latent space analyzer
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def analyze_layer(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        compute_svd: bool = True,
        compute_entropy: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single layer's activations with bfloat16 support
        
        Args:
            activations: Layer activations (any shape)
            compute_svd: Whether to compute SVD for dimensionality
            compute_entropy: Whether to compute entropy
        
        Returns:
            Dictionary with analysis statistics
        """
        # Handle None or empty inputs
        if activations is None:
            return {'error': 'None activations provided'}
        
        # Convert to tensor if needed
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations)
        elif isinstance(activations, (list, tuple)):
            if len(activations) == 0:
                return {'error': 'Empty activations list'}
            activations = torch.stack(activations)
        
        # Convert bfloat16 to float32 for compatibility
        activations = ensure_numpy_compatible(activations)
        
        # Check for empty tensors
        if activations.numel() == 0:
            return {'error': 'Empty tensor'}
        
        # Flatten for some computations
        activations_flat = activations.flatten()
        
        stats = {
            'shape': list(activations.shape),
            'mean': float(activations.mean().item()),
            'std': float(activations.std().item()),
            'variance': float(activations.var().item()),
            'min': float(activations.min().item()),
            'max': float(activations.max().item()),
            'norm': float(activations.norm().item()),
        }
        
        # Compute entropy
        if compute_entropy:
            try:
                # Ensure numpy compatible before histogram
                activations_np = ensure_numpy_compatible(activations_flat).detach().cpu().numpy()
                hist, _ = np.histogram(activations_np, bins=50)
                hist = hist + 1e-10  # Avoid log(0)
                hist = hist / hist.sum()
                stats['entropy'] = float(entropy(hist))
            except Exception as e:
                stats['entropy'] = None
                stats['entropy_error'] = str(e)
        
        # Compute SVD for dimensionality analysis
        if compute_svd:
            svd_stats = self._compute_svd_stats(activations)
            stats.update(svd_stats)
        
        return stats
    
    def _compute_svd_stats(self, activations: torch.Tensor) -> Dict[str, Any]:
        """Compute SVD-based statistics"""
        # Ensure bfloat16 is converted
        activations = ensure_numpy_compatible(activations)
        
        # Reshape to (samples, features)
        if len(activations.shape) > 2:
            # Flatten spatial dimensions
            activations_2d = activations.view(activations.shape[0], -1)
        else:
            activations_2d = activations
        
        # Center the data
        activations_centered = activations_2d - activations_2d.mean(dim=0, keepdims=True)
        
        # Compute SVD
        try:
            U, s, Vt = torch.linalg.svd(activations_centered, full_matrices=False)
            s_np = ensure_numpy_compatible(s).detach().cpu().numpy()
        except Exception as e:
            warnings.warn(f"SVD computation failed: {e}")
            return {
                'effective_rank': 0,
                'singular_values': [],
                'top_singular_values': [],
                'singular_value_ratio': 0.0
            }
        
        # Effective rank (number of singular values > threshold)
        threshold = s_np.max() * 1e-6
        effective_rank = np.sum(s_np > threshold)
        
        # Top singular values
        top_k = min(10, len(s_np))
        top_singular_values = s_np[:top_k].tolist()
        
        # Ratio of top singular value to sum
        if len(s_np) > 0 and s_np.sum() > 0:
            singular_value_ratio = float(s_np[0] / s_np.sum())
        else:
            singular_value_ratio = 0.0
        
        return {
            'effective_rank': int(effective_rank),
            'full_rank': activations_2d.shape[1],
            'singular_values': s_np.tolist(),
            'top_singular_values': top_singular_values,
            'singular_value_ratio': singular_value_ratio,
            'rank_deficiency': activations_2d.shape[1] - effective_rank
        }
    
    def compute_jacobian(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        output_dim: Optional[int] = None,
        method: str = 'autograd',
        batch_size: int = 10
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix for a layer
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to compute Jacobian for
            inputs: Input tensor
            output_dim: Optional output dimension (if None, uses full output)
            method: Method ('autograd' or 'finite_diff')
            batch_size: Batch size for autograd method (default: 10)
        
        Returns:
            Jacobian matrix
        """
        if method == 'autograd':
            return self._compute_jacobian_autograd(model, layer_name, inputs, output_dim, batch_size)
        elif method == 'finite_diff':
            return self._compute_jacobian_finite_diff(model, layer_name, inputs, output_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_jacobian_autograd(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        output_dim: Optional[int],
        batch_size: int = 10,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute Jacobian using autograd with memory-efficient batched processing
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to compute Jacobian for
            inputs: Input tensor (can be input_ids or embeddings)
            output_dim: Optional output dimension (if None, uses full output)
            batch_size: Batch size for processing (default: 10)
            position_embeddings: Optional tuple of (cos, sin) tensors for RoPE (required for Gemma2)
        
        Returns:
            Jacobian matrix
        """
        # Enable expandable segments to reduce fragmentation
        if torch.cuda.is_available():
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            torch.cuda.empty_cache()
        
        # Check if inputs are embeddings (float dtype) or input_ids (int dtype)
        # If embeddings, we need to skip the embedding layer in forward pass
        is_embeddings = inputs.dtype.is_floating_point
        
        # Ensure inputs can have gradients (must be float)
        if not inputs.dtype.is_floating_point:
            raise ValueError(f"Inputs must be floating point tensors (embeddings), got dtype {inputs.dtype}. "
                           f"Convert input_ids to embeddings first using model's embedding layer.")
        
        inputs.requires_grad_(True)
        
        # Get layer
        try:
            layer = dict(model.named_modules())[layer_name]
        except KeyError:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # If inputs are embeddings, use hooks to intercept embedding layer
        if is_embeddings:
            # Define hook to capture target layer output
            layer_output = None
            
            def capture_hook(module, input, output):
                nonlocal layer_output
                # Check for tuple output (common in transformers)
                if isinstance(output, tuple):
                    layer_output = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    layer_output = output.last_hidden_state
                else:
                    layer_output = output
            
            # Register hook on target layer
            handle = layer.register_forward_hook(capture_hook)
            
            try:
                # Call model with inputs_embeds
                # Rely on model to generate position_ids automatically
                # This avoids shape mismatches with some models (Gemma)
                model(inputs_embeds=inputs, output_hidden_states=True)
                
                output = layer_output
                
                if output is None:
                     raise ValueError(f"Failed to capture output for layer {layer_name}. Hook did not trigger.")

            finally:
                handle.remove()

        else:
            # Inputs are input_ids - use similar hook approach for consistency and safety
            # instead of manual navigation
            
            # Define hook to capture target layer output
            layer_output = None
            
            def capture_hook(module, input, output):
                nonlocal layer_output
                if isinstance(output, tuple):
                    layer_output = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    layer_output = output.last_hidden_state
                else:
                    layer_output = output
                    
            # Register hook on target layer
            handle = layer.register_forward_hook(capture_hook)
            
            try:
                model(input_ids=inputs, output_hidden_states=True)
                output = layer_output
                 
                if output is None:
                     raise ValueError(f"Failed to capture output for layer {layer_name}. Hook did not trigger.")
                     
            finally:
                handle.remove()
        
        # Ensure output is a tensor and extract last token position for sequence outputs
        # This handles (batch, seq_len, hidden_dim) -> (batch, hidden_dim) for last token
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Output is not a tensor: {type(output)}")
        
        # For sequence outputs (3D: batch, seq_len, hidden_dim), use last token position
        # This is standard for transformer layer analysis
        if len(output.shape) == 3:
            # Shape: (batch, seq_len, hidden_dim) -> (batch, hidden_dim) for last token
            output = output[:, -1, :]  # Extract last token position
        elif len(output.shape) == 2 and output.shape[0] > 1:
            # If batch dimension > 1, take last element
            output = output[-1]
        
        # Flatten output (full Jacobian - no dimension limiting for steering capability)
        # Now output should be 1D or 2D (batch, hidden_dim) or (hidden_dim,)
        if output_dim is None:
            output_flat = output.flatten()
        else:
            output_flat = output.view(-1)[:output_dim]
        
        # Ensure inputs are properly shaped for gradient computation
        # Flatten inputs for Jacobian computation (input_size = total elements)
        input_size = inputs.numel()
        output_size = len(output_flat)
        
        # Compute Jacobian in batches to reduce memory
        jacobian = torch.zeros(output_size, input_size, device=inputs.device, dtype=torch.float32)
        
        for batch_start in range(0, output_size, batch_size):
            batch_end = min(batch_start + batch_size, output_size)
            
            # Process batch
            for i in range(batch_start, batch_end):
                if inputs.grad is not None:
                    inputs.grad.zero_()
                
                # Only retain graph if not last element in batch
                retain = (i < batch_end - 1)
                
                # Ensure output_flat[i] is a scalar for backward()
                if output_flat[i].dim() > 0:
                    # If not scalar, sum to make it scalar
                    output_flat[i].sum().backward(retain_graph=retain)
                else:
                    output_flat[i].backward(retain_graph=retain)
                
                # Flatten inputs.grad to match input_size
                if inputs.grad is not None:
                    jacobian[i] = inputs.grad.flatten().detach()
                else:
                    # If no gradient, fill with zeros
                    jacobian[i] = torch.zeros(input_size, device=inputs.device)
            
            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clean up
        inputs.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jacobian
    
    def _compute_jacobian_finite_diff(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        output_dim: Optional[int],
        eps: float = 1e-5
    ) -> torch.Tensor:
        """Compute Jacobian using finite differences"""
        # Get baseline output
        with torch.no_grad():
            x = inputs
            for name, module in model.named_modules():
                if name == layer_name:
                    break
                x = module(x)
            layer = dict(model.named_modules())[layer_name]
            output_baseline = layer(x)
            if output_dim is None:
                output_baseline_flat = output_baseline.flatten()
            else:
                output_baseline_flat = output_baseline.view(-1)[:output_dim]
        
        # Compute finite differences
        jacobian = torch.zeros(len(output_baseline_flat), inputs.numel())
        inputs_flat = inputs.flatten()
        
        for i in range(inputs.numel()):
            inputs_perturbed = inputs.clone()
            inputs_perturbed_flat = inputs_perturbed.flatten()
            inputs_perturbed_flat[i] += eps
            inputs_perturbed = inputs_perturbed_flat.view(inputs.shape)
            
            with torch.no_grad():
                x = inputs_perturbed
                for name, module in model.named_modules():
                    if name == layer_name:
                        break
                    x = module(x)
                layer = dict(model.named_modules())[layer_name]
                output_perturbed = layer(x)
                if output_dim is None:
                    output_perturbed_flat = output_perturbed.flatten()
                else:
                    output_perturbed_flat = output_perturbed.view(-1)[:output_dim]
            
            jacobian[:, i] = (output_perturbed_flat - output_baseline_flat) / eps
        
        return jacobian
    
    def compute_jacobian_determinant(
        self,
        jacobian: torch.Tensor
    ) -> float:
        """
        Compute determinant of Jacobian (measure of volume preservation)
        
        Args:
            jacobian: Jacobian matrix
        
        Returns:
            Determinant value
        """
        # For non-square matrices, use pseudo-determinant or singular values
        if jacobian.shape[0] == jacobian.shape[1]:
            det = torch.det(jacobian)
            return float(det.item())
        else:
            # Use product of singular values
            s = torch.linalg.svdvals(jacobian)
            pseudo_det = s.prod()
            return float(pseudo_det.item())
    
    def compute_jacobian_stats_only(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        batch_size: int = 10,
        compute_singular_values: bool = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute full Jacobian but only return statistics (memory-efficient)
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to compute Jacobian for
            inputs: Input tensor
            batch_size: Batch size for processing (default: 10)
            compute_singular_values: Whether to compute SVD (default: True, top 10 only)
            position_embeddings: Optional tuple of (cos, sin) tensors for RoPE (required for Gemma2)
        
        Returns:
            Dictionary with statistics (determinant, norms, singular values, etc.)
        """
        # Compute full Jacobian
        jacobian = self._compute_jacobian_autograd(model, layer_name, inputs, None, batch_size, position_embeddings)
        
        # Extract statistics immediately
        stats = {
            'shape': list(jacobian.shape),
            'frobenius_norm': float(torch.norm(jacobian, p='fro').item()),
            'max': float(jacobian.max().item()),
            'min': float(jacobian.min().item()),
            'mean': float(jacobian.mean().item()),
            'std': float(jacobian.std().item()),
        }
        
        # Compute determinant
        try:
            if jacobian.shape[0] == jacobian.shape[1]:
                stats['determinant'] = float(torch.det(jacobian).item())
            else:
                # Use pseudo-determinant (product of singular values)
                s = torch.linalg.svdvals(jacobian)
                stats['determinant'] = float(s.prod().item())
        except Exception as e:
            warnings.warn(f"Failed to compute determinant for {layer_name}: {e}")
            stats['determinant'] = None
        
        # Compute singular values (top 10 for memory efficiency)
        if compute_singular_values:
            try:
                singular_values = torch.linalg.svdvals(jacobian)
                top_k = min(10, len(singular_values))
                stats['top_singular_values'] = singular_values[:top_k].cpu().tolist()
                stats['spectral_norm'] = float(singular_values[0].item())
                
                # Condition number
                if len(singular_values) > 1 and singular_values[-1] > 0:
                    stats['condition_number'] = float((singular_values[0] / singular_values[-1]).item())
                else:
                    stats['condition_number'] = None
            except Exception as e:
                warnings.warn(f"Failed to compute SVD for {layer_name}: {e}")
                stats['top_singular_values'] = []
                stats['spectral_norm'] = None
                stats['condition_number'] = None
        else:
            stats['top_singular_values'] = []
            stats['spectral_norm'] = None
            stats['condition_number'] = None
        
        # Delete Jacobian and clear cache
        del jacobian
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return stats
    
    def compute_jacobians_for_basins_stats_only(
        self,
        model: nn.Module,
        vulnerability_basins: List[Dict[str, Any]],
        inputs: torch.Tensor,
        batch_size: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute Jacobian statistics for all vulnerability basins (memory-efficient)
        
        Args:
            model: PyTorch model
            vulnerability_basins: List of vulnerability basin dictionaries
            inputs: Input tensor
            batch_size: Batch size per Jacobian (default: 10)
            save_path: Optional JSON save path
        
        Returns:
            Dictionary mapping layer names to statistics
        """
        # Enable expandable segments
        if torch.cuda.is_available():
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            torch.cuda.empty_cache()
        
        # --- GEMMA 2 FIX: Generate RoPE embeddings required for Gemma 2 decoder layers ---
        # Generate position_embeddings once at the start using model.model.rotary_emb
        position_embeddings = None
        device = next(model.parameters()).device
        
        # Access the model's rotary embedding module
        rotary_emb = None
        if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
            rotary_emb = model.model.rotary_emb
        elif hasattr(model, 'rotary_emb'):
            rotary_emb = model.rotary_emb
        
        if rotary_emb is not None:
            seq_length = inputs.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate position_embeddings using rotary_emb(inputs, position_ids)
            try:
                cos, sin = rotary_emb(inputs, position_ids)
                position_embeddings = (cos, sin)
            except Exception as e:
                warnings.warn(f"Could not generate position_embeddings: {e}")
                position_embeddings = None
        
        results = {}
        total_basins = len(vulnerability_basins)
        
        print(f"Computing Jacobian statistics for {total_basins} vulnerability basins...")
        
        for idx, basin in enumerate(vulnerability_basins):
            layer_name = basin['layer_name']
            print(f"Processing basin {idx+1}/{total_basins}: {layer_name}")
            
            try:
                # Check for dimension mismatch before processing
                # Get the target layer to check its input dimension requirements
                module_dict = dict(model.named_modules())
                target_layer = module_dict.get(layer_name)
                
                if target_layer is not None:
                    # Removed restrictive dimension check that caused false positives
                    pass
                
                # Compute Jacobian statistics (not full matrix)
                # Pass position_embeddings down to the computation
                stats = self.compute_jacobian_stats_only(
                    model, layer_name, inputs, batch_size, 
                    compute_singular_values=True,
                    position_embeddings=position_embeddings
                )
                
                # Store results
                results[layer_name] = {
                    'basin_info': basin,
                    'jacobian_stats': stats
                }
                
                print(f"  ✓ Completed {layer_name} (determinant: {stats.get('determinant', 'N/A')})")
                
            except Exception as e:
                warnings.warn(f"Failed to compute Jacobian for {layer_name}: {e}")
                results[layer_name] = {
                    'basin_info': basin,
                    'error': str(e),
                    'jacobian_stats': None
                }
                print(f"  ✗ Error: {str(e)}")
            
            # Clear cache between basins
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results if path provided
        if save_path:
            self._save_jacobian_results(results, save_path)
        
        return results
    
    def iterative_perturbation_steering(
        self,
        model: nn.Module,
        layer_name: str,
        initial_inputs: torch.Tensor,
        tokenizer: Any,
        target_determinant: float = 0.0,
        max_iterations: int = 50,
        step_size: float = 0.01,
        convergence_threshold: float = 1e-6,
        trend_window: int = 5,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Iteratively perturb inputs to steer Jacobian determinant toward zero
        
        Args:
            model: PyTorch model
            layer_name: Target layer name
            initial_inputs: Initial input tensor (token IDs)
            tokenizer: Tokenizer for decoding
            target_determinant: Target determinant (default: 0.0)
            max_iterations: Maximum iterations (default: 50)
            step_size: Gradient step size (default: 0.01)
            convergence_threshold: Stop if determinant change < threshold (default: 1e-6)
            trend_window: Number of iterations to check trend (default: 5)
            batch_size: Batch size for Jacobian computation (default: 10)
        
        Returns:
            Dictionary with steering results including perturbations
        """
        # Enable expandable segments
        if torch.cuda.is_available():
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            torch.cuda.empty_cache()
        
        # Get embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'embed_tokens'):
            embed_layer = model.embed_tokens
        else:
            raise ValueError("Could not find embedding layer in model")
        
        # Initialize embeddings (requires grad for perturbation)
        current_inputs = initial_inputs.clone().detach()
        current_embeddings = embed_layer(current_inputs).detach()
        initial_embeddings = current_embeddings.clone()
        
        current_embeddings.requires_grad_(True)
        
        determinant_history = []
        perturbation_history = []
        
        converged = False
        
        print(f"Steering Jacobian determinant toward {target_determinant} for {layer_name}...")
        
        for iteration in range(max_iterations):
            try:
                # Compute full Jacobian w.r.t. embeddings for determinant
                # Forward pass to get layer output
                current_embeddings.requires_grad_(True)
                hidden_states = current_embeddings
                
                # Navigate to target layer
                for name, module in model.named_modules():
                    if name == layer_name:
                        break
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    hidden_states = module(hidden_states)
                
                # Get layer output
                target_layer = dict(model.named_modules())[layer_name]
                layer_output = target_layer(hidden_states)
                output_flat = layer_output.flatten()
                
                # Compute full Jacobian in batches (memory-efficient)
                embed_size = current_embeddings.numel()
                output_size = len(output_flat)
                
                # For very large Jacobians, we might need to use a subset
                # But try full first
                if output_size * embed_size > 1e8:  # > 100M elements
                    # Use subset for determinant estimation
                    sample_size = min(500, output_size)
                    indices = torch.randperm(output_size)[:sample_size]
                    output_sample = output_flat[indices]
                    compute_full = False
                else:
                    output_sample = output_flat
                    compute_full = True
                
                # Compute Jacobian
                jacobian = torch.zeros(len(output_sample), embed_size,
                                      device=current_embeddings.device, dtype=torch.float32)
                
                for i in range(len(output_sample)):
                    if current_embeddings.grad is not None:
                        current_embeddings.grad.zero_()
                    output_sample[i].backward(retain_graph=(i < len(output_sample) - 1))
                    jacobian[i] = current_embeddings.grad.flatten().detach()
                
                # Compute determinant (or pseudo-determinant)
                if jacobian.shape[0] == jacobian.shape[1]:
                    det = torch.det(jacobian)
                else:
                    # Use product of singular values
                    s = torch.linalg.svdvals(jacobian)
                    det = s.prod()
                
                determinant = float(det.item())
                determinant_history.append(determinant)
                
                # Store det for gradient computation
                det_tensor = det.detach().clone()
                
                # Delete Jacobian immediately (only keep determinant)
                del jacobian
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Store perturbation
                perturbation_delta = current_embeddings - initial_embeddings
                perturbation_norm = float(perturbation_delta.norm().item())
                
                # Decode current inputs
                try:
                    current_text = tokenizer.decode(current_inputs[0], skip_special_tokens=True)
                except:
                    current_text = None
                
                perturbation_history.append({
                    'iteration': iteration,
                    'determinant': determinant,
                    'perturbation_norm': perturbation_norm,
                    'token_sequence': current_inputs[0].cpu().tolist(),
                    'text': current_text,
                    'embedding_delta': ensure_numpy_compatible(perturbation_delta).detach().cpu().numpy().tolist() if iteration > 0 else None
                })
                
                print(f"  Iteration {iteration+1}/{max_iterations}: determinant={determinant:.6e}, "
                      f"perturbation_norm={perturbation_norm:.6f}")
                
                # Check convergence
                if abs(determinant - target_determinant) < convergence_threshold:
                    converged = True
                    print(f"  ✓ Converged to target determinant")
                    break
                
                # Check trend (is determinant trending toward zero?)
                if len(determinant_history) >= trend_window:
                    recent_dets = determinant_history[-trend_window:]
                    if all(recent_dets[i] > recent_dets[i+1] for i in range(len(recent_dets)-1)):
                        # Still trending down, continue
                        pass
                    elif abs(recent_dets[-1] - recent_dets[0]) < convergence_threshold:
                        # Not making progress, might be stuck
                        print(f"  ⚠️  Determinant not trending toward zero (stuck)")
                        break
                
                # Update embeddings to reduce determinant
                # Recompute forward pass with gradient tracking for update
                current_embeddings.requires_grad_(True)
                hidden_states_grad = current_embeddings
                
                for name, module in model.named_modules():
                    if name == layer_name:
                        break
                    if isinstance(hidden_states_grad, tuple):
                        hidden_states_grad = hidden_states_grad[0]
                    hidden_states_grad = module(hidden_states_grad)
                
                target_layer_grad = dict(model.named_modules())[layer_name]
                layer_output_grad = target_layer_grad(hidden_states_grad)
                output_flat_grad = layer_output_grad.flatten()
                
                # Use smaller subset for gradient computation (memory efficiency)
                if len(output_flat_grad) > 500:
                    grad_sample_size = 500
                    grad_indices = torch.randperm(len(output_flat_grad))[:grad_sample_size]
                    output_for_grad = output_flat_grad[grad_indices]
                else:
                    output_for_grad = output_flat_grad
                
                # Compute Jacobian for gradient (smaller subset)
                jacobian_grad = torch.zeros(len(output_for_grad), embed_size,
                                           device=current_embeddings.device)
                
                for i in range(len(output_for_grad)):
                    if current_embeddings.grad is not None:
                        current_embeddings.grad.zero_()
                    output_for_grad[i].backward(retain_graph=(i < len(output_for_grad) - 1))
                    jacobian_grad[i] = current_embeddings.grad.flatten().detach()
                
                # Compute determinant approximation for gradient
                if jacobian_grad.shape[0] == jacobian_grad.shape[1]:
                    det_for_grad = torch.det(jacobian_grad)
                else:
                    s = torch.linalg.svdvals(jacobian_grad)
                    det_for_grad = s.prod()
                
                # Compute gradient of determinant w.r.t. embeddings
                current_embeddings.grad.zero_()
                det_for_grad.backward()
                
                # Update embeddings (gradient descent on determinant)
                with torch.no_grad():
                    # Step in direction that reduces determinant
                    current_embeddings = current_embeddings - step_size * current_embeddings.grad.sign()
                    current_embeddings = current_embeddings.detach()
                
                # Clean up
                del jacobian_grad, det_for_grad, output_for_grad, hidden_states_grad, layer_output_grad, output_flat_grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                warnings.warn(f"Error in iteration {iteration}: {e}")
                break
        
        # Final perturbation
        final_perturbation_delta = current_embeddings - initial_embeddings
        final_perturbation_norm = float(final_perturbation_delta.norm().item())
        
        try:
            final_text = tokenizer.decode(current_inputs[0], skip_special_tokens=True)
        except:
            final_text = None
        
        result = {
            'layer_name': layer_name,
            'initial_determinant': determinant_history[0] if determinant_history else None,
            'final_determinant': determinant_history[-1] if determinant_history else None,
            'iterations': len(determinant_history),
            'converged': converged,
            'determinant_history': determinant_history,
            'perturbations': perturbation_history,
            'final_perturbation': {
                'embedding_delta': ensure_numpy_compatible(final_perturbation_delta).detach().cpu().numpy().tolist(),
                'token_sequence': current_inputs[0].cpu().tolist(),
                'text': final_text,
                'perturbation_norm': final_perturbation_norm
            },
            'initial_embeddings': ensure_numpy_compatible(initial_embeddings).detach().cpu().numpy().tolist(),
            'final_embeddings': ensure_numpy_compatible(current_embeddings).detach().cpu().numpy().tolist(),
            'metadata': {
                'target_determinant': target_determinant,
                'step_size': step_size,
                'trend_window': trend_window,
                'max_iterations': max_iterations
            }
        }
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _save_jacobian_results(self, results: Dict[str, Dict[str, Any]], save_path: str):
        """Save Jacobian results to JSON file"""
        # Convert to JSON-serializable format
        save_data = {
            'metadata': {
                'total_basins': len(results),
                'computation_date': datetime.now().isoformat(),
                'device': self.device
            },
            'results': {}
        }
        
        for layer_name, result in results.items():
            save_data['results'][layer_name] = {
                'basin_info': result.get('basin_info', {}),
                'jacobian_stats': result.get('jacobian_stats', {}),
                'error': result.get('error', None)
            }
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(convert_to_native(save_data), f, indent=2)
        
        print(f"\n✓ Saved Jacobian results to {save_path}")
    
    def analyze_all_layers(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all layers
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
        
        Returns:
            Dictionary mapping layer names to analysis statistics
        """
        results = {}
        for layer_name, activations in layer_activations.items():
            results[layer_name] = self.analyze_layer(activations)
        return results
    
    def identify_vulnerability_basins(
        self,
        layer_stats: Dict[str, Dict[str, Any]],
        variance_threshold: float = 0.01,
        entropy_threshold: float = 2.0,
        rank_deficiency_threshold: int = 10,
        singular_value_ratio_threshold: float = 0.95,
        require_multiple_criteria: bool = False,
        min_criteria_count: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Identify layers that may be vulnerability basins
        
        Args:
            layer_stats: Dictionary of layer statistics from analyze_all_layers
            variance_threshold: Threshold for low variance (lower = stricter)
            entropy_threshold: Threshold for low entropy (lower = stricter)
            rank_deficiency_threshold: Threshold for rank deficiency (higher = stricter)
            singular_value_ratio_threshold: Threshold for singular value ratio (higher = stricter, default 0.95)
            require_multiple_criteria: If True, require multiple criteria to be met (AND logic)
            min_criteria_count: Minimum number of criteria that must be met (default: 2)
        
        Returns:
            List of vulnerability basin candidates
        """
        basins = []
        
        for layer_name, stats in layer_stats.items():
            criteria_met = []
            reasons = []
            
            # Check variance
            if stats.get('variance', float('inf')) < variance_threshold:
                criteria_met.append('variance')
                reasons.append(f"Low variance: {stats['variance']:.6f}")
            
            # Check entropy
            if stats.get('entropy', float('inf')) < entropy_threshold:
                criteria_met.append('entropy')
                reasons.append(f"Low entropy: {stats['entropy']:.3f}")
            
            # Check rank deficiency
            if stats.get('rank_deficiency', 0) > rank_deficiency_threshold:
                criteria_met.append('rank_deficiency')
                reasons.append(f"Rank deficiency: {stats['rank_deficiency']}")
            
            # Check singular value ratio (high = near-nilpotent)
            # Made configurable and stricter by default (0.95 instead of 0.9)
            if stats.get('singular_value_ratio', 0) > singular_value_ratio_threshold:
                criteria_met.append('singular_value_ratio')
                reasons.append(f"High singular value ratio: {stats['singular_value_ratio']:.3f}")
            
            # Determine if this layer is a basin
            is_basin = False
            if require_multiple_criteria:
                # AND logic: require multiple criteria
                if len(criteria_met) >= min_criteria_count:
                    is_basin = True
            else:
                # OR logic: any criterion is sufficient
                if len(criteria_met) > 0:
                    is_basin = True
            
            if is_basin:
                basins.append({
                    'layer_name': layer_name,
                    'reasons': reasons,
                    'stats': stats,
                    'criteria_met': criteria_met,
                    'criteria_count': len(criteria_met)
                })
        
        return basins
    
    def track_dimensionality_evolution(
        self,
        activations_sequence: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
        layer_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Track how dimensionality evolves over a sequence
        
        Args:
            activations_sequence: List of activation dictionaries over time
            layer_name: Layer to track
        
        Returns:
            Dictionary with evolution metrics
        """
        effective_ranks = []
        variances = []
        entropies = []
        
        for activations in activations_sequence:
            if layer_name in activations:
                stats = self.analyze_layer(activations[layer_name])
                effective_ranks.append(stats.get('effective_rank', 0))
                variances.append(stats.get('variance', 0))
                entropies.append(stats.get('entropy', 0))
        
        return {
            'effective_rank': np.array(effective_ranks),
            'variance': np.array(variances),
            'entropy': np.array(entropies)
        }
    
    def detect_collapse_signature(
        self,
        evolution_metrics: Dict[str, np.ndarray],
        collapse_threshold: float = 0.5
    ) -> bool:
        """
        Detect if collapse signature is present
        
        Args:
            evolution_metrics: Metrics from track_dimensionality_evolution
            collapse_threshold: Threshold for collapse detection
        
        Returns:
            True if collapse signature detected
        """
        # Check if effective rank drops significantly
        if 'effective_rank' in evolution_metrics:
            ranks = evolution_metrics['effective_rank']
            if len(ranks) > 1:
                rank_drop = (ranks[0] - ranks[-1]) / ranks[0] if ranks[0] > 0 else 0
                if rank_drop > collapse_threshold:
                    return True
        
        # Check if variance drops significantly
        if 'variance' in evolution_metrics:
            variances = evolution_metrics['variance']
            if len(variances) > 1:
                variance_drop = (variances[0] - variances[-1]) / variances[0] if variances[0] > 0 else 0
                if variance_drop > collapse_threshold:
                    return True
        
        return False






