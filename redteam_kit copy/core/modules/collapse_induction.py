"""
Collapse Induction Module
Searches for sequences that minimize variance and induce latent-space collapse

Usage for Red Teaming:
---------------------
This module searches for input sequences that steer the model toward vulnerability
basins, minimize layer variance, reduce Jacobian determinants, and potentially
induce latent-space collapse.

Example Usage:
    from core.modules.collapse_induction import CollapseInduction
    
    inducer = CollapseInduction(model, tokenizer, instrumentation)
    
    # Search for collapse candidates
    candidates = inducer.search_collapse_sequences(
        initial_prompts=["Test prompt"],
        target_layers=["layer_10", "layer_15"]
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import random
from tqdm import tqdm

from .latent_space_analysis import LatentSpaceAnalyzer
from .adversarial_perturbation import AdversarialPerturbationEngine


class CollapseInduction:
    """Search for sequences that induce latent-space collapse"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        instrumentation: Any,
        device: Optional[str] = None
    ):
        """
        Initialize collapse induction module
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer
            instrumentation: ModelInstrumentation instance
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.instrumentation = instrumentation
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.analyzer = LatentSpaceAnalyzer(device=self.device)
        self.perturbation_engine = AdversarialPerturbationEngine(
            model, tokenizer, device=self.device
        )
        
        # Storage
        self.collapse_candidates: List[Dict[str, Any]] = []
        self.search_history: List[Dict[str, Any]] = []
    
    def search_collapse_sequences(
        self,
        initial_prompts: List[str],
        target_layers: Optional[List[str]] = None,
        num_candidates: int = 10,
        max_iterations: int = 100,
        variance_threshold: float = 0.01,
        search_strategy: str = "gradient"
    ) -> List[Dict[str, Any]]:
        """
        Search for input sequences that minimize variance in target layers
        
        Args:
            initial_prompts: Starting prompts to mutate
            target_layers: Optional list of target layers (uses all if None)
            num_candidates: Number of candidate sequences to find
            max_iterations: Maximum search iterations
            variance_threshold: Target variance threshold
            search_strategy: Search strategy ('gradient', 'random', 'evolutionary')
        
        Returns:
            List of collapse candidate sequences
        """
        if target_layers is None:
            target_layers = self.instrumentation.get_layer_names()
        
        candidates = []
        
        if search_strategy == "gradient":
            candidates = self._gradient_search(
                initial_prompts, target_layers, num_candidates, max_iterations, variance_threshold
            )
        elif search_strategy == "random":
            candidates = self._random_search(
                initial_prompts, target_layers, num_candidates, max_iterations, variance_threshold
            )
        elif search_strategy == "evolutionary":
            candidates = self._evolutionary_search(
                initial_prompts, target_layers, num_candidates, max_iterations, variance_threshold
            )
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        self.collapse_candidates.extend(candidates)
        return candidates
    
    def _gradient_search(
        self,
        initial_prompts: List[str],
        target_layers: List[str],
        num_candidates: int,
        max_iterations: int,
        variance_threshold: float
    ) -> List[Dict[str, Any]]:
        """Gradient-based search for variance minimization"""
        candidates = []
        
        for prompt in tqdm(initial_prompts, desc="Gradient search"):
            best_variance = float('inf')
            best_sequence = None
            
            # Tokenize input
            inputs = self.perturbation_engine._tokenize_input(prompt).to(self.device)
            inputs.requires_grad_(True)
            
            for iteration in range(max_iterations):
                # Forward pass
                self.instrumentation.clear_activations()
                outputs = self.model(inputs)
                
                # Get activations
                activations = self.instrumentation.get_activations()
                
                # Compute variance for target layers
                total_variance = 0.0
                layer_variances = {}
                
                for layer_name in target_layers:
                    if layer_name in activations:
                        layer_acts = activations[layer_name]
                        if isinstance(layer_acts, list):
                            layer_acts = layer_acts[-1]
                        
                        stats = self.analyzer.analyze_layer(layer_acts)
                        variance = stats['variance']
                        total_variance += variance
                        layer_variances[layer_name] = variance
                
                avg_variance = total_variance / len(target_layers) if target_layers else 0
                
                # Check if this is a good candidate
                if avg_variance < variance_threshold:
                    candidates.append({
                        'sequence': self.tokenizer.decode(inputs[0], skip_special_tokens=True),
                        'variance': avg_variance,
                        'layer_variances': layer_variances,
                        'iteration': iteration,
                        'method': 'gradient'
                    })
                    
                    if len(candidates) >= num_candidates:
                        break
                
                # Update best
                if avg_variance < best_variance:
                    best_variance = avg_variance
                    best_sequence = inputs.clone()
                
                # Compute gradient for variance minimization
                self.model.zero_grad()
                # Use negative variance as loss (minimize variance)
                loss = -torch.tensor(avg_variance, device=self.device, requires_grad=True)
                loss.backward()
                
                # Update input (gradient ascent on variance minimization)
                with torch.no_grad():
                    # Small step in direction that reduces variance
                    step_size = 0.01
                    inputs = inputs - step_size * inputs.grad.sign()
                    inputs = torch.clamp(inputs, 0, self.tokenizer.vocab_size - 1)
                    inputs.requires_grad_(True)
            
            # Add best found
            if best_sequence is not None and best_variance < float('inf'):
                candidates.append({
                    'sequence': self.tokenizer.decode(best_sequence[0], skip_special_tokens=True),
                    'variance': best_variance,
                    'method': 'gradient'
                })
        
        return candidates[:num_candidates]
    
    def _random_search(
        self,
        initial_prompts: List[str],
        target_layers: List[str],
        num_candidates: int,
        max_iterations: int,
        variance_threshold: float
    ) -> List[Dict[str, Any]]:
        """Random search for variance minimization"""
        candidates = []
        
        for prompt in tqdm(initial_prompts, desc="Random search"):
            for iteration in range(max_iterations):
                # Randomly mutate prompt
                mutated_prompt = self._mutate_prompt(prompt)
                
                # Tokenize and run
                inputs = self.perturbation_engine._tokenize_input(mutated_prompt).to(self.device)
                
                self.instrumentation.clear_activations()
                with torch.no_grad():
                    _ = self.model(inputs)
                
                activations = self.instrumentation.get_activations()
                
                # Compute variance
                total_variance = 0.0
                layer_variances = {}
                
                for layer_name in target_layers:
                    if layer_name in activations:
                        layer_acts = activations[layer_name]
                        if isinstance(layer_acts, list):
                            layer_acts = layer_acts[-1]
                        
                        stats = self.analyzer.analyze_layer(layer_acts)
                        variance = stats['variance']
                        total_variance += variance
                        layer_variances[layer_name] = variance
                
                avg_variance = total_variance / len(target_layers) if target_layers else 0
                
                # Check if candidate
                if avg_variance < variance_threshold:
                    candidates.append({
                        'sequence': mutated_prompt,
                        'variance': avg_variance,
                        'layer_variances': layer_variances,
                        'iteration': iteration,
                        'method': 'random'
                    })
                    
                    if len(candidates) >= num_candidates:
                        break
        
        return candidates[:num_candidates]
    
    def _evolutionary_search(
        self,
        initial_prompts: List[str],
        target_layers: List[str],
        num_candidates: int,
        max_iterations: int,
        variance_threshold: float
    ) -> List[Dict[str, Any]]:
        """Evolutionary search for variance minimization"""
        # Initialize population
        population = [(prompt, None) for prompt in initial_prompts]
        population_size = len(initial_prompts) * 5
        
        candidates = []
        
        for generation in tqdm(range(max_iterations), desc="Evolutionary search"):
            # Evaluate population
            evaluated_population = []
            
            for prompt, _ in population:
                inputs = self.perturbation_engine._tokenize_input(prompt).to(self.device)
                
                self.instrumentation.clear_activations()
                with torch.no_grad():
                    _ = self.model(inputs)
                
                activations = self.instrumentation.get_activations()
                
                # Compute fitness (negative variance)
                total_variance = 0.0
                for layer_name in target_layers:
                    if layer_name in activations:
                        layer_acts = activations[layer_name]
                        if isinstance(layer_acts, list):
                            layer_acts = layer_acts[-1]
                        stats = self.analyzer.analyze_layer(layer_acts)
                        total_variance += stats['variance']
                
                avg_variance = total_variance / len(target_layers) if target_layers else 0
                fitness = -avg_variance  # Negative because we want to minimize
                
                evaluated_population.append((prompt, fitness, avg_variance))
                
                # Check if candidate
                if avg_variance < variance_threshold:
                    candidates.append({
                        'sequence': prompt,
                        'variance': avg_variance,
                        'generation': generation,
                        'method': 'evolutionary'
                    })
            
            if len(candidates) >= num_candidates:
                break
            
            # Select top performers
            evaluated_population.sort(key=lambda x: x[1], reverse=True)
            elite = evaluated_population[:len(evaluated_population) // 2]
            
            # Create new generation
            new_population = [p for p, _, _ in elite]
            
            while len(new_population) < population_size:
                # Crossover
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1[0], parent2[0])
                
                # Mutation
                if random.random() < 0.3:
                    child = self._mutate_prompt(child)
                
                new_population.append(child)
            
            population = [(p, None) for p in new_population]
        
        return candidates[:num_candidates]
    
    def _mutate_prompt(self, prompt: str) -> str:
        """Mutate a prompt randomly"""
        # Simple mutation: add/remove words, change order
        words = prompt.split()
        
        if len(words) == 0:
            return prompt
        
        mutation_type = random.choice(['shuffle', 'add', 'remove', 'replace'])
        
        if mutation_type == 'shuffle' and len(words) > 1:
            random.shuffle(words)
            return ' '.join(words)
        elif mutation_type == 'add':
            words.insert(random.randint(0, len(words)), random.choice(['the', 'a', 'an', 'and', 'or']))
            return ' '.join(words)
        elif mutation_type == 'remove' and len(words) > 1:
            words.pop(random.randint(0, len(words) - 1))
            return ' '.join(words)
        else:
            # Replace random word
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(['test', 'example', 'sample', 'data'])
            return ' '.join(words)
    
    def _crossover(self, prompt1: str, prompt2: str) -> str:
        """Crossover two prompts"""
        words1 = prompt1.split()
        words2 = prompt2.split()
        
        if len(words1) == 0 or len(words2) == 0:
            return prompt1
        
        # Take first half from prompt1, second half from prompt2
        split_point = len(words1) // 2
        child = words1[:split_point] + words2[split_point:]
        
        return ' '.join(child)
    
    def minimize_jacobian_determinant(
        self,
        prompt: str,
        target_layer: str,
        max_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Search for input that minimizes Jacobian determinant
        
        Args:
            prompt: Starting prompt
            target_layer: Target layer name
            max_iterations: Maximum iterations
        
        Returns:
            Dictionary with best sequence and determinant value
        """
        from .latent_space_analysis import LatentSpaceAnalyzer
        
        inputs = self.perturbation_engine._tokenize_input(prompt).to(self.device)
        inputs.requires_grad_(True)
        
        best_determinant = float('inf')
        best_sequence = None
        
        for iteration in range(max_iterations):
            # Compute Jacobian
            jacobian = self.analyzer.compute_jacobian(
                self.model,
                target_layer,
                inputs,
                method='autograd'
            )
            
            # Compute determinant
            det = self.analyzer.compute_jacobian_determinant(jacobian)
            
            if det < best_determinant:
                best_determinant = det
                best_sequence = inputs.clone()
            
            # Update input to minimize determinant
            self.model.zero_grad()
            loss = torch.tensor(det, device=self.device, requires_grad=True)
            loss.backward()
            
            with torch.no_grad():
                step_size = 0.01
                inputs = inputs - step_size * inputs.grad.sign()
                inputs = torch.clamp(inputs, 0, self.tokenizer.vocab_size - 1)
                inputs.requires_grad_(True)
        
        return {
            'sequence': self.tokenizer.decode(best_sequence[0], skip_special_tokens=True) if best_sequence is not None else prompt,
            'determinant': best_determinant,
            'target_layer': target_layer
        }
    
    def get_collapse_candidates(self) -> List[Dict[str, Any]]:
        """Get all collapse candidates"""
        return self.collapse_candidates
    
    def clear_candidates(self):
        """Clear collapse candidates"""
        self.collapse_candidates.clear()


