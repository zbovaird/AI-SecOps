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
        Analyze a single layer's activations
        
        Args:
            activations: Layer activations (any shape)
            compute_svd: Whether to compute SVD for dimensionality
            compute_entropy: Whether to compute entropy
        
        Returns:
            Dictionary with analysis statistics
        """
        # Convert to tensor if needed
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations)
        
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
            # Discretize for entropy computation
            hist, _ = np.histogram(activations_flat.detach().cpu().numpy(), bins=50)
            hist = hist + 1e-10  # Avoid log(0)
            hist = hist / hist.sum()
            stats['entropy'] = float(entropy(hist))
        
        # Compute SVD for dimensionality analysis
        if compute_svd:
            svd_stats = self._compute_svd_stats(activations)
            stats.update(svd_stats)
        
        return stats
    
    def _compute_svd_stats(self, activations: torch.Tensor) -> Dict[str, Any]:
        """Compute SVD-based statistics"""
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
            s_np = s.detach().cpu().numpy()
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
        method: str = 'autograd'
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix for a layer
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to compute Jacobian for
            inputs: Input tensor
            output_dim: Optional output dimension (if None, uses full output)
            method: Method ('autograd' or 'finite_diff')
        
        Returns:
            Jacobian matrix
        """
        if method == 'autograd':
            return self._compute_jacobian_autograd(model, layer_name, inputs, output_dim)
        elif method == 'finite_diff':
            return self._compute_jacobian_finite_diff(model, layer_name, inputs, output_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_jacobian_autograd(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        output_dim: Optional[int]
    ) -> torch.Tensor:
        """Compute Jacobian using autograd"""
        inputs.requires_grad_(True)
        
        # Get layer
        layer = dict(model.named_modules())[layer_name]
        
        # Forward pass to layer
        x = inputs
        for name, module in model.named_modules():
            if name == layer_name:
                break
            x = module(x)
        
        # Get output
        output = layer(x)
        
        # Flatten output
        if output_dim is None:
            output_flat = output.flatten()
        else:
            output_flat = output.view(-1)[:output_dim]
        
        # Compute Jacobian
        jacobian = torch.zeros(len(output_flat), inputs.numel())
        
        for i in range(len(output_flat)):
            if inputs.grad is not None:
                inputs.grad.zero_()
            output_flat[i].backward(retain_graph=True)
            jacobian[i] = inputs.grad.flatten()
        
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
        rank_deficiency_threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify layers that may be vulnerability basins
        
        Args:
            layer_stats: Dictionary of layer statistics from analyze_all_layers
            variance_threshold: Threshold for low variance
            entropy_threshold: Threshold for low entropy
            rank_deficiency_threshold: Threshold for rank deficiency
        
        Returns:
            List of vulnerability basin candidates
        """
        basins = []
        
        for layer_name, stats in layer_stats.items():
            is_basin = False
            reasons = []
            
            # Check variance
            if stats.get('variance', float('inf')) < variance_threshold:
                is_basin = True
                reasons.append(f"Low variance: {stats['variance']:.6f}")
            
            # Check entropy
            if stats.get('entropy', float('inf')) < entropy_threshold:
                is_basin = True
                reasons.append(f"Low entropy: {stats['entropy']:.3f}")
            
            # Check rank deficiency
            if stats.get('rank_deficiency', 0) > rank_deficiency_threshold:
                is_basin = True
                reasons.append(f"Rank deficiency: {stats['rank_deficiency']}")
            
            # Check singular value ratio (high = near-nilpotent)
            if stats.get('singular_value_ratio', 0) > 0.9:
                is_basin = True
                reasons.append(f"High singular value ratio: {stats['singular_value_ratio']:.3f}")
            
            if is_basin:
                basins.append({
                    'layer_name': layer_name,
                    'reasons': reasons,
                    'stats': stats
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


