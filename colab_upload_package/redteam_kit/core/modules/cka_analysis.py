"""
Centered Kernel Alignment (CKA) Analysis Module
Computes CKA similarity between layers and models

Usage for Red Teaming:
---------------------
CKA measures similarity between representations in different layers or models.
This helps identify vulnerability basins, track latent space evolution, and
assess cross-model transferability.

Example Usage:
    from core.modules.cka_analysis import CKAAnalyzer
    
    analyzer = CKAAnalyzer()
    
    # Compute CKA between two layers
    cka_score = analyzer.compute_cka(activations_layer1, activations_layer2)
    
    # Compute full similarity matrix
    similarity_matrix = analyzer.compute_similarity_matrix(all_layer_activations)
    
    # Compare two models
    cross_model_cka = analyzer.compute_cross_model_cka(
        model1_activations, model2_activations
    )
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from scipy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils import ensure_numpy_compatible


class CKAAnalyzer:
    """Compute and analyze Centered Kernel Alignment"""
    
    def __init__(self, kernel_type: str = 'linear', use_rbf: bool = False):
        """
        Initialize CKA analyzer
        
        Args:
            kernel_type: Type of kernel ('linear', 'rbf', 'polynomial')
            use_rbf: Whether to use RBF kernel (more expensive but more accurate)
        """
        self.kernel_type = kernel_type
        self.use_rbf = use_rbf
    
    def compute_cka(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray],
        debiased: bool = False
    ) -> float:
        """
        Compute Centered Kernel Alignment between two representations
        
        Args:
            X: First representation (samples x features)
            Y: Second representation (samples x features)
            debiased: Whether to use debiased CKA
        
        Returns:
            CKA score between 0 and 1 (1 = identical, 0 = orthogonal)
        """
        # Convert to numpy if needed (with bfloat16 support)
        if isinstance(X, torch.Tensor):
            X = ensure_numpy_compatible(X).detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = ensure_numpy_compatible(Y).detach().cpu().numpy()
        
        # Flatten if needed (handle multi-dimensional activations)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        if len(Y.shape) > 2:
            Y = Y.reshape(Y.shape[0], -1)
        
        # Ensure same number of samples
        min_samples = min(X.shape[0], Y.shape[0])
        X = X[:min_samples]
        Y = Y[:min_samples]
        
        # Center the data
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        
        if debiased:
            return self._compute_debiased_cka(X_centered, Y_centered)
        else:
            return self._compute_standard_cka(X_centered, Y_centered)
    
    def _compute_standard_cka(
        self,
        X_centered: np.ndarray,
        Y_centered: np.ndarray
    ) -> float:
        """Compute standard CKA"""
        # Compute Gram matrices
        K = X_centered @ X_centered.T
        L = Y_centered @ Y_centered.T
        
        # Center the Gram matrices
        K_centered = self._center_gram_matrix(K)
        L_centered = self._center_gram_matrix(L)
        
        # Compute CKA
        numerator = np.trace(K_centered @ L_centered)
        denominator = np.sqrt(np.trace(K_centered @ K_centered) * np.trace(L_centered @ L_centered))
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def _compute_debiased_cka(
        self,
        X_centered: np.ndarray,
        Y_centered: np.ndarray
    ) -> float:
        """Compute debiased CKA (better for small sample sizes)"""
        n = X_centered.shape[0]
        
        # Compute Gram matrices
        K = X_centered @ X_centered.T
        L = Y_centered @ Y_centered.T
        
        # Center Gram matrices
        K_centered = self._center_gram_matrix(K)
        L_centered = self._center_gram_matrix(L)
        
        # Debiased computation
        # Remove diagonal terms
        K_centered_debiased = K_centered.copy()
        L_centered_debiased = L_centered.copy()
        np.fill_diagonal(K_centered_debiased, 0)
        np.fill_diagonal(L_centered_debiased, 0)
        
        numerator = np.trace(K_centered_debiased @ L_centered_debiased)
        denominator = np.sqrt(
            np.trace(K_centered_debiased @ K_centered_debiased) *
            np.trace(L_centered_debiased @ L_centered_debiased)
        )
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def _center_gram_matrix(self, K: np.ndarray) -> np.ndarray:
        """Center a Gram matrix"""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    def compute_similarity_matrix(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        layer_names: Optional[List[str]] = None,
        debiased: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise CKA similarity matrix for all layers
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
            layer_names: Optional list of layer names (uses dict keys if None)
            debiased: Whether to use debiased CKA
        
        Returns:
            Tuple of (similarity_matrix, layer_names)
        """
        if layer_names is None:
            layer_names = list(layer_activations.keys())
        
        n_layers = len(layer_names)
        similarity_matrix = np.zeros((n_layers, n_layers))
        
        # Compute pairwise similarities
        for i, name_i in enumerate(layer_names):
            for j, name_j in enumerate(layer_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    # Compute CKA
                    cka_score = self.compute_cka(
                        layer_activations[name_i],
                        layer_activations[name_j],
                        debiased=debiased
                    )
                    similarity_matrix[i, j] = cka_score
                    similarity_matrix[j, i] = cka_score  # Symmetric
        
        return similarity_matrix, layer_names
    
    def compute_cross_model_cka(
        self,
        model1_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        model2_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        model1_layers: Optional[List[str]] = None,
        model2_layers: Optional[List[str]] = None,
        debiased: bool = False
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute CKA between layers of two different models
        
        Args:
            model1_activations: Activations from first model
            model2_activations: Activations from second model
            model1_layers: Optional layer names for model1
            model2_layers: Optional layer names for model2
            debiased: Whether to use debiased CKA
        
        Returns:
            Tuple of (similarity_matrix, model1_layer_names, model2_layer_names)
        """
        if model1_layers is None:
            model1_layers = list(model1_activations.keys())
        if model2_layers is None:
            model2_layers = list(model2_activations.keys())
        
        n1 = len(model1_layers)
        n2 = len(model2_layers)
        similarity_matrix = np.zeros((n1, n2))
        
        # Compute cross-model similarities
        for i, layer1 in enumerate(model1_layers):
            for j, layer2 in enumerate(model2_layers):
                cka_score = self.compute_cka(
                    model1_activations[layer1],
                    model2_activations[layer2],
                    debiased=debiased
                )
                similarity_matrix[i, j] = cka_score
        
        return similarity_matrix, model1_layers, model2_layers
    
    def visualize_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        layer_names: List[str],
        output_path: Optional[str] = None,
        title: str = "Layer Similarity Matrix (CKA)",
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize CKA similarity matrix as heatmap
        
        Args:
            similarity_matrix: CKA similarity matrix
            layer_names: List of layer names
            output_path: Optional path to save figure
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=layer_names,
            yticklabels=layer_names,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            vmin=0,
            vmax=1,
            square=True
        )
        
        plt.title(title)
        plt.xlabel('Layer')
        plt.ylabel('Layer')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_cross_model_similarity(
        self,
        similarity_matrix: np.ndarray,
        model1_layers: List[str],
        model2_layers: List[str],
        output_path: Optional[str] = None,
        title: str = "Cross-Model Similarity (CKA)",
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Visualize cross-model CKA similarity matrix
        
        Args:
            similarity_matrix: Cross-model CKA matrix
            model1_layers: Layer names for model 1
            model2_layers: Layer names for model 2
            output_path: Optional path to save figure
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            similarity_matrix,
            xticklabels=model2_layers,
            yticklabels=model1_layers,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            vmin=0,
            vmax=1
        )
        
        plt.title(title)
        plt.xlabel('Model 2 Layers')
        plt.ylabel('Model 1 Layers')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def find_most_similar_layers(
        self,
        similarity_matrix: np.ndarray,
        layer_names: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Find most similar layer pairs
        
        Args:
            similarity_matrix: CKA similarity matrix
            layer_names: List of layer names
            top_k: Number of top pairs to return
        
        Returns:
            List of (layer1, layer2, cka_score) tuples
        """
        # Get upper triangle (excluding diagonal)
        n = len(layer_names)
        pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((
                    layer_names[i],
                    layer_names[j],
                    similarity_matrix[i, j]
                ))
        
        # Sort by CKA score
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def track_cka_drift(
        self,
        activations_sequence: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
        reference_layer: str,
        target_layer: str,
        debiased: bool = False
    ) -> np.ndarray:
        """
        Track CKA drift over a sequence of activations
        
        Args:
            activations_sequence: List of activation dictionaries over time
            reference_layer: Reference layer name
            target_layer: Target layer name to track
            debiased: Whether to use debiased CKA
        
        Returns:
            Array of CKA scores over time
        """
        cka_scores = []
        
        for activations in activations_sequence:
            if reference_layer in activations and target_layer in activations:
                cka = self.compute_cka(
                    activations[reference_layer],
                    activations[target_layer],
                    debiased=debiased
                )
                cka_scores.append(cka)
        
        return np.array(cka_scores)




