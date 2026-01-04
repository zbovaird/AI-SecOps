"""
Phase 1: Open Model CKA Computation

Computes CKA similarity matrix between all open-weight models
to identify which models have similar internal representations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

from ..core.cka import CKA, CKAResult
from ..core.model_loader import ModelLoader, SUPPORTED_MODELS, sample_layers

logger = logging.getLogger("transferability.experiments.open_model_cka")


@dataclass
class CKAMatrix:
    """CKA similarity matrix between models."""
    model_names: List[str]
    similarity_matrix: np.ndarray  # n_models x n_models
    
    # Detailed pairwise results
    pairwise_results: Dict[str, CKAResult] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    prompts_used: int = 0
    layers_sampled: int = 0
    kernel: str = "linear"
    
    def to_dict(self) -> Dict:
        return {
            "model_names": self.model_names,
            "similarity_matrix": self.similarity_matrix.tolist(),
            "timestamp": self.timestamp,
            "prompts_used": self.prompts_used,
            "layers_sampled": self.layers_sampled,
            "kernel": self.kernel,
            "pairwise_summary": {
                k: {"mean_cka": v.mean_cka, "max_cka": v.max_cka}
                for k, v in self.pairwise_results.items()
            },
        }
    
    def get_most_similar_pair(self) -> Tuple[str, str, float]:
        """Get the most similar model pair (excluding self-similarity)."""
        n = len(self.model_names)
        max_sim = -1
        best_pair = ("", "")
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] > max_sim:
                    max_sim = self.similarity_matrix[i, j]
                    best_pair = (self.model_names[i], self.model_names[j])
        
        return best_pair[0], best_pair[1], max_sim
    
    def get_similarity(self, model_a: str, model_b: str) -> float:
        """Get similarity between two models."""
        try:
            i = self.model_names.index(model_a)
            j = self.model_names.index(model_b)
            return self.similarity_matrix[i, j]
        except ValueError:
            return 0.0
    
    def rank_by_similarity_to(self, target_model: str) -> List[Tuple[str, float]]:
        """Rank all models by similarity to a target model."""
        try:
            target_idx = self.model_names.index(target_model)
        except ValueError:
            return []
        
        similarities = []
        for i, model in enumerate(self.model_names):
            if model != target_model:
                similarities.append((model, self.similarity_matrix[target_idx, i]))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "=" * 60,
            "CKA SIMILARITY MATRIX",
            "=" * 60,
            f"Models: {', '.join(self.model_names)}",
            f"Kernel: {self.kernel}",
            f"Prompts: {self.prompts_used}, Layers sampled: {self.layers_sampled}",
            "",
            "Similarity Matrix:",
        ]
        
        # Header
        header = "          " + "  ".join(f"{m[:8]:>8}" for m in self.model_names)
        lines.append(header)
        
        # Rows
        for i, model in enumerate(self.model_names):
            row = f"{model[:8]:>8}  " + "  ".join(
                f"{self.similarity_matrix[i, j]:>8.3f}"
                for j in range(len(self.model_names))
            )
            lines.append(row)
        
        # Most similar pair
        m1, m2, sim = self.get_most_similar_pair()
        lines.extend([
            "",
            f"Most similar pair: {m1} <-> {m2} (CKA = {sim:.3f})",
        ])
        
        return "\n".join(lines)


class OpenModelCKAExperiment:
    """
    Phase 1 Experiment: Compute CKA between open-weight models.
    
    This establishes the structural similarity baseline between models
    that we can later correlate with attack transferability.
    """
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        kernel: str = "linear",
        n_layer_samples: int = 5,
    ):
        """
        Initialize experiment.
        
        Args:
            hf_token: HuggingFace token
            device: Device for model loading
            dtype: Data type for models
            kernel: CKA kernel ("linear" or "rbf")
            n_layer_samples: Number of layers to sample per model
        """
        self.loader = ModelLoader(device=device, dtype=dtype, hf_token=hf_token)
        self.cka = CKA(kernel=kernel)
        self.kernel = kernel
        self.n_layer_samples = n_layer_samples
        
        logger.info(f"OpenModelCKAExperiment initialized: kernel={kernel}")
    
    def run(
        self,
        model_names: List[str],
        prompts: List[str],
        save_path: Optional[str] = None,
    ) -> CKAMatrix:
        """
        Run the CKA experiment across specified models.
        
        Args:
            model_names: List of model keys (e.g., ["gemma2", "mistral", "llama"])
            prompts: Prompts to use for hidden state extraction
            save_path: Optional path to save results
            
        Returns:
            CKAMatrix with similarity scores
        """
        logger.info(f"Starting CKA experiment with {len(model_names)} models, {len(prompts)} prompts")
        
        # Validate model names
        for name in model_names:
            if name not in SUPPORTED_MODELS:
                raise ValueError(f"Unknown model: {name}. Supported: {list(SUPPORTED_MODELS.keys())}")
        
        # Extract hidden states from all models
        model_states: Dict[str, Dict[int, np.ndarray]] = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Extracting states from: {model_name}")
            logger.info(f"{'='*50}")
            
            result = self.loader.load_and_extract(
                model_name,
                prompts,
                pooling="last",
            )
            
            # Sample layers for efficiency
            all_layers = sorted(result.layer_states.keys())
            sampled_layers = sample_layers(len(all_layers), self.n_layer_samples)
            sampled_layer_indices = [all_layers[i] for i in sampled_layers if i < len(all_layers)]
            
            # Convert to numpy and keep only sampled layers
            states_numpy = result.to_numpy()
            model_states[model_name] = {
                layer: states_numpy[layer]
                for layer in sampled_layer_indices
                if layer in states_numpy
            }
            
            logger.info(f"  Extracted {len(model_states[model_name])} layers")
            
            # Cleanup
            self.loader._cleanup()
        
        # Compute CKA matrix
        logger.info("\nComputing CKA similarity matrix...")
        similarity_matrix, names = self.cka.compute_similarity_matrix(
            model_states,
            use_mean=True,
        )
        
        # Build detailed pairwise results
        pairwise_results = {}
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i < j:
                    key = f"{name_a}_vs_{name_b}"
                    result = self.cka.compare_layer_representations(
                        model_states[name_a],
                        model_states[name_b],
                        name_a,
                        name_b,
                    )
                    pairwise_results[key] = result
        
        # Create result
        cka_matrix = CKAMatrix(
            model_names=names,
            similarity_matrix=similarity_matrix,
            pairwise_results=pairwise_results,
            timestamp=datetime.now().isoformat(),
            prompts_used=len(prompts),
            layers_sampled=self.n_layer_samples,
            kernel=self.kernel,
        )
        
        logger.info("\n" + cka_matrix.summary())
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(cka_matrix.to_dict(), f, indent=2)
            logger.info(f"\nResults saved to: {save_path}")
        
        return cka_matrix
    
    def run_subset(
        self,
        model_a: str,
        model_b: str,
        prompts: List[str],
    ) -> CKAResult:
        """
        Run CKA between just two models.
        
        Useful for quick pairwise comparisons.
        
        Args:
            model_a: First model key
            model_b: Second model key
            prompts: Prompts to use
            
        Returns:
            CKAResult
        """
        # Load and extract from model A
        result_a = self.loader.load_and_extract(model_a, prompts, pooling="last")
        states_a = result_a.to_numpy()
        self.loader._cleanup()
        
        # Load and extract from model B
        result_b = self.loader.load_and_extract(model_b, prompts, pooling="last")
        states_b = result_b.to_numpy()
        self.loader._cleanup()
        
        # Compute CKA
        return self.cka.compare_layer_representations(
            states_a,
            states_b,
            model_a,
            model_b,
        )


# Convenience prompts for CKA computation
CKA_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about nature.",
    "What are the main causes of climate change?",
    "Describe the process of machine learning.",
    "How does the internet work?",
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise?",
    "Describe the water cycle.",
    "How do vaccines work?",
    "What is artificial intelligence?",
    "Explain the theory of relativity.",
    "What causes earthquakes?",
    "How do computers process information?",
    "What is the history of the internet?",
    "Explain how electricity works.",
    "What are renewable energy sources?",
    "How does the human brain work?",
    "What is blockchain technology?",
]
