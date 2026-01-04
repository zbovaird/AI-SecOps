"""
Centered Kernel Alignment (CKA) Implementation

CKA measures representational similarity between neural network layers,
enabling comparison of internal representations across different models.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited" (2019)
    https://arxiv.org/abs/1905.00414
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("transferability.cka")


def centering_matrix(n: int) -> np.ndarray:
    """
    Create centering matrix H = I - (1/n) * 1 * 1^T
    
    Args:
        n: Size of the matrix
        
    Returns:
        n x n centering matrix
    """
    return np.eye(n) - np.ones((n, n)) / n


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC measures statistical dependence between two kernel matrices.
    
    Args:
        K: First kernel matrix (n x n)
        L: Second kernel matrix (n x n)
        
    Returns:
        HSIC value
    """
    n = K.shape[0]
    H = centering_matrix(n)
    
    # Center the kernel matrices
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # HSIC = (1/(n-1)^2) * trace(K_centered @ L_centered)
    return np.trace(K_centered @ L_centered) / ((n - 1) ** 2)


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """
    Compute linear kernel matrix K = X @ X^T
    
    Args:
        X: Feature matrix (n_samples x n_features)
        
    Returns:
        Kernel matrix (n_samples x n_samples)
    """
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix.
    
    Args:
        X: Feature matrix (n_samples x n_features)
        sigma: RBF bandwidth. If None, uses median heuristic.
        
    Returns:
        Kernel matrix (n_samples x n_samples)
    """
    # Compute pairwise squared distances
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) + \
               np.sum(X ** 2, axis=1, keepdims=True).T - \
               2 * X @ X.T
    
    # Use median heuristic for sigma if not provided
    if sigma is None:
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]) / 2)
        if sigma == 0:
            sigma = 1.0
    
    return np.exp(-sq_dists / (2 * sigma ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA between two representation matrices.
    
    Linear CKA is invariant to orthogonal transformations and isotropic scaling,
    making it suitable for comparing representations across different models.
    
    Args:
        X: First representation matrix (n_samples x n_features_x)
        Y: Second representation matrix (n_samples x n_features_y)
        
    Returns:
        CKA similarity score in [0, 1]
    """
    # Ensure same number of samples
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    
    # Compute kernel matrices
    K = linear_kernel(X)
    L = linear_kernel(Y)
    
    # Compute CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    if hsic_kk * hsic_ll == 0:
        return 0.0
    
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Compute RBF (kernel) CKA between two representation matrices.
    
    RBF CKA captures non-linear similarities between representations.
    
    Args:
        X: First representation matrix (n_samples x n_features_x)
        Y: Second representation matrix (n_samples x n_features_y)
        sigma: RBF bandwidth. If None, uses median heuristic.
        
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)
    
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    if hsic_kk * hsic_ll == 0:
        return 0.0
    
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


@dataclass
class CKAResult:
    """Result of CKA comparison between two models."""
    model_a: str
    model_b: str
    
    # Layer-wise CKA scores (layer_a, layer_b) -> score
    layer_cka: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Aggregated scores
    mean_cka: float = 0.0
    max_cka: float = 0.0
    
    # Best layer alignment
    best_layer_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_cka": self.mean_cka,
            "max_cka": self.max_cka,
            "best_layer_pairs": self.best_layer_pairs,
            "layer_cka_sample": dict(list(self.layer_cka.items())[:20]),
        }


class CKA:
    """
    CKA Analysis Engine for comparing model representations.
    
    Supports:
    - Linear and RBF CKA
    - Layer-wise comparison
    - Batch processing for memory efficiency
    - Multiple aggregation strategies
    """
    
    def __init__(
        self,
        kernel: str = "linear",
        sigma: Optional[float] = None,
        batch_size: int = 100,
    ):
        """
        Initialize CKA analyzer.
        
        Args:
            kernel: "linear" or "rbf"
            sigma: RBF bandwidth (only for kernel="rbf")
            batch_size: Batch size for processing large datasets
        """
        self.kernel = kernel
        self.sigma = sigma
        self.batch_size = batch_size
        
        if kernel == "linear":
            self._cka_func = linear_cka
        elif kernel == "rbf":
            self._cka_func = lambda X, Y: rbf_cka(X, Y, sigma)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        logger.info(f"CKA initialized with {kernel} kernel")
    
    def compare_representations(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """
        Compare two representation matrices.
        
        Args:
            X: First representations (n_samples x n_features)
            Y: Second representations (n_samples x n_features)
            
        Returns:
            CKA similarity score
        """
        # Convert to float32 for memory efficiency
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        
        return self._cka_func(X, Y)
    
    def compare_layer_representations(
        self,
        model_a_states: Dict[int, np.ndarray],
        model_b_states: Dict[int, np.ndarray],
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
    ) -> CKAResult:
        """
        Compare hidden states across all layers between two models.
        
        Args:
            model_a_states: Dict mapping layer_idx -> hidden states (n_samples x hidden_size)
            model_b_states: Dict mapping layer_idx -> hidden states
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            CKAResult with layer-wise and aggregated scores
        """
        result = CKAResult(model_a=model_a_name, model_b=model_b_name)
        
        layers_a = sorted(model_a_states.keys())
        layers_b = sorted(model_b_states.keys())
        
        logger.info(f"Computing CKA between {model_a_name} ({len(layers_a)} layers) "
                   f"and {model_b_name} ({len(layers_b)} layers)")
        
        all_scores = []
        
        for layer_a in layers_a:
            for layer_b in layers_b:
                X = model_a_states[layer_a]
                Y = model_b_states[layer_b]
                
                # Ensure same number of samples
                min_samples = min(X.shape[0], Y.shape[0])
                X = X[:min_samples]
                Y = Y[:min_samples]
                
                score = self.compare_representations(X, Y)
                result.layer_cka[(layer_a, layer_b)] = score
                all_scores.append((layer_a, layer_b, score))
        
        if all_scores:
            scores_only = [s[2] for s in all_scores]
            result.mean_cka = float(np.mean(scores_only))
            result.max_cka = float(np.max(scores_only))
            
            # Best layer pairs (top 5)
            sorted_scores = sorted(all_scores, key=lambda x: x[2], reverse=True)
            result.best_layer_pairs = sorted_scores[:5]
        
        return result
    
    def compute_similarity_matrix(
        self,
        model_states: Dict[str, Dict[int, np.ndarray]],
        use_mean: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute CKA similarity matrix between all model pairs.
        
        Args:
            model_states: Dict mapping model_name -> (layer_idx -> hidden states)
            use_mean: If True, use mean CKA across layers; else use max
            
        Returns:
            Tuple of (similarity_matrix, model_names)
        """
        model_names = list(model_states.keys())
        n_models = len(model_names)
        
        similarity_matrix = np.eye(n_models)  # Diagonal is 1
        
        logger.info(f"Computing {n_models}x{n_models} CKA similarity matrix...")
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                result = self.compare_layer_representations(
                    model_states[model_names[i]],
                    model_states[model_names[j]],
                    model_names[i],
                    model_names[j],
                )
                
                score = result.mean_cka if use_mean else result.max_cka
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score
                
                logger.info(f"  {model_names[i]} vs {model_names[j]}: {score:.4f}")
        
        return similarity_matrix, model_names


def compute_cka_batch(
    X_batches: List[np.ndarray],
    Y_batches: List[np.ndarray],
    kernel: str = "linear",
) -> float:
    """
    Compute CKA using batched processing for memory efficiency.
    
    Useful when full representations don't fit in memory.
    
    Args:
        X_batches: List of batch representations from model X
        Y_batches: List of batch representations from model Y
        kernel: "linear" or "rbf"
        
    Returns:
        CKA similarity score
    """
    # Concatenate batches
    X = np.vstack(X_batches)
    Y = np.vstack(Y_batches)
    
    if kernel == "linear":
        return linear_cka(X, Y)
    else:
        return rbf_cka(X, Y)
