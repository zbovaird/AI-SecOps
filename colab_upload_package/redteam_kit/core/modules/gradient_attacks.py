"""
Gradient-Based Adversarial Attacks for Latent Space Red Teaming
Implements FGSM, PGD, and other gradient attacks on embeddings

Usage:
    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine
    
    engine = GradientAttackEngine(model, tokenizer, instrumentation)
    results = engine.attack_prompt(
        prompt="Test prompt",
        attack_type="fgsm",
        target_layers=["model.layers.1", "model.layers.2"]
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import warnings

try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        BasicIterativeMethod,
        MomentumIterativeMethod,
        AutoProjectedGradientDescent
    )
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    warnings.warn("IBM ART not available. Install with: pip install adversarial-robustness-toolbox")


class GradientAttackEngine:
    """
    Gradient-based adversarial attack engine for latent space red teaming.
    Works on embeddings (not token IDs) and tracks impact on vulnerable layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        instrumentation: Optional[Any] = None,
        device: Optional[str] = None
    ):
        """
        Initialize gradient attack engine
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer for the model
            instrumentation: ModelInstrumentation instance for tracking activations
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.instrumentation = instrumentation
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        # Store original embeddings for reference
        self.embedding_layer = None
        self._find_embedding_layer()
    
    def _find_embedding_layer(self):
        """Find the embedding layer in the model"""
        for name, module in self.model.named_modules():
            if 'embed' in name.lower() and isinstance(module, nn.Embedding):
                self.embedding_layer = module
                break
        
        if self.embedding_layer is None:
            # Try common names
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                self.embedding_layer = self.model.model.embed_tokens
            elif hasattr(self.model, 'embed_tokens'):
                self.embedding_layer = self.model.embed_tokens
    
    def attack_prompt(
        self,
        prompt: str,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        max_iter: int = 10,
        alpha: Optional[float] = None,
        target_layers: Optional[List[str]] = None,
        target_loss: str = "cross_entropy"
    ) -> Dict[str, Any]:
        """
        Perform gradient attack on a prompt
        
        Args:
            prompt: Input prompt text
            attack_type: Type of attack ('fgsm', 'pgd', 'bim', 'mim', 'auto_pgd')
            epsilon: Perturbation budget
            max_iter: Maximum iterations (for iterative attacks)
            alpha: Step size (default: epsilon / max_iter)
            target_layers: Optional list of layers to target for loss
            target_loss: Loss type ('cross_entropy', 'targeted', 'untargeted')
        
        Returns:
            Dictionary with attack results and layer impacts
        """
        if self.embedding_layer is None:
            raise ValueError("Could not find embedding layer in model")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs['input_ids']
        
        # Get baseline activations
        baseline_activations = {}
        if self.instrumentation:
            self.instrumentation.activations.clear()
            with torch.no_grad():
                _ = self.model(**inputs)
            baseline_activations = {
                name: acts[-1].clone() if isinstance(acts, list) and acts else acts.clone()
                for name, acts in self.instrumentation.activations.items()
            }
        
        # Get embeddings
        embeddings = self.embedding_layer(input_ids)
        original_embeddings = embeddings.clone()
        
        # Perform attack
        if attack_type.lower() == "fgsm":
            adversarial_embeddings = self._fgsm_attack(
                embeddings, input_ids, epsilon, target_layers, target_loss
            )
        elif attack_type.lower() == "pgd":
            adversarial_embeddings = self._pgd_attack(
                embeddings, input_ids, epsilon, max_iter, alpha, target_layers, target_loss
            )
        elif attack_type.lower() == "bim":
            adversarial_embeddings = self._bim_attack(
                embeddings, input_ids, epsilon, max_iter, alpha, target_layers, target_loss
            )
        elif attack_type.lower() == "mim":
            adversarial_embeddings = self._mim_attack(
                embeddings, input_ids, epsilon, max_iter, alpha, target_layers, target_loss
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Get adversarial activations
        adversarial_activations = {}
        if self.instrumentation:
            # Create fake input_ids from adversarial embeddings (nearest neighbor)
            # This is approximate - in practice you'd use a projection method
            with torch.no_grad():
                # Find nearest token embeddings
                adv_input_ids = self._embeddings_to_tokens(adversarial_embeddings, original_embeddings, input_ids)
                
                self.instrumentation.activations.clear()
                adv_inputs = {'input_ids': adv_input_ids}
                _ = self.model(**adv_inputs)
                
                adversarial_activations = {
                    name: acts[-1].clone() if isinstance(acts, list) and acts else acts.clone()
                    for name, acts in self.instrumentation.activations.items()
                }
        
        # Compute layer impacts
        layer_impacts = {}
        if baseline_activations and adversarial_activations:
            for layer_name in baseline_activations.keys():
                if layer_name in adversarial_activations:
                    baseline = baseline_activations[layer_name]
                    adversarial = adversarial_activations[layer_name]
                    
                    # Compute metrics
                    diff = (adversarial - baseline).abs()
                    layer_impacts[layer_name] = {
                        'mean_diff': float(diff.mean().item()),
                        'max_diff': float(diff.max().item()),
                        'norm_diff': float(torch.norm(diff).item()),
                        'relative_change': float(torch.norm(diff).item() / (torch.norm(baseline).item() + 1e-8))
                    }
        
        # Decode adversarial prompt (approximate)
        adversarial_prompt = self._decode_adversarial_embeddings(adversarial_embeddings, input_ids)
        
        return {
            'original_prompt': prompt,
            'adversarial_prompt': adversarial_prompt,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'layer_impacts': layer_impacts,
            'embedding_perturbation_norm': float(torch.norm(adversarial_embeddings - original_embeddings).item()),
            'baseline_activations': {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                                    for k, v in baseline_activations.items()} if baseline_activations else {},
            'adversarial_activations': {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                                        for k, v in adversarial_activations.items()} if adversarial_activations else {}
        }
    
    def _fgsm_attack(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
        target_layers: Optional[List[str]],
        target_loss: str
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        embeddings.requires_grad_(True)
        
        # Forward pass with adversarial embeddings
        # We need to replace embeddings in the forward pass
        outputs = self._forward_with_embeddings(embeddings, input_ids)
        
        # Compute loss
        loss = self._compute_loss(outputs, input_ids, target_layers, target_loss)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        perturbation = epsilon * embeddings.grad.sign()
        adversarial_embeddings = embeddings + perturbation
        
        return adversarial_embeddings.detach()
    
    def _pgd_attack(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
        max_iter: int,
        alpha: Optional[float],
        target_layers: Optional[List[str]],
        target_loss: str
    ) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        if alpha is None:
            alpha = epsilon / max_iter
        
        original_embeddings = embeddings.clone()
        adversarial_embeddings = embeddings.clone()
        adversarial_embeddings.requires_grad_(True)
        
        for iteration in range(max_iter):
            # Forward pass
            outputs = self._forward_with_embeddings(adversarial_embeddings, input_ids)
            
            # Compute loss
            loss = self._compute_loss(outputs, input_ids, target_layers, target_loss)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial embeddings
            with torch.no_grad():
                perturbation = alpha * adversarial_embeddings.grad.sign()
                adversarial_embeddings = adversarial_embeddings + perturbation
                
                # Project to epsilon ball
                delta = adversarial_embeddings - original_embeddings
                delta = torch.clamp(delta, -epsilon, epsilon)
                adversarial_embeddings = original_embeddings + delta
            
            adversarial_embeddings.requires_grad_(True)
        
        return adversarial_embeddings.detach()
    
    def _bim_attack(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
        max_iter: int,
        alpha: Optional[float],
        target_layers: Optional[List[str]],
        target_loss: str
    ) -> torch.Tensor:
        """Basic Iterative Method (BIM) - same as PGD but with smaller steps"""
        if alpha is None:
            alpha = epsilon / max_iter
        
        return self._pgd_attack(embeddings, input_ids, epsilon, max_iter, alpha, target_layers, target_loss)
    
    def _mim_attack(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        epsilon: float,
        max_iter: int,
        alpha: Optional[float],
        target_layers: Optional[List[str]],
        target_loss: str
    ) -> torch.Tensor:
        """Momentum Iterative Method (MIM)"""
        if alpha is None:
            alpha = epsilon / max_iter
        
        original_embeddings = embeddings.clone()
        adversarial_embeddings = embeddings.clone()
        adversarial_embeddings.requires_grad_(True)
        
        momentum = torch.zeros_like(embeddings)
        decay_factor = 1.0
        
        for iteration in range(max_iter):
            # Forward pass
            outputs = self._forward_with_embeddings(adversarial_embeddings, input_ids)
            
            # Compute loss
            loss = self._compute_loss(outputs, input_ids, target_layers, target_loss)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update momentum
            with torch.no_grad():
                momentum = decay_factor * momentum + adversarial_embeddings.grad / torch.norm(adversarial_embeddings.grad)
                
                # Update adversarial embeddings
                perturbation = alpha * momentum.sign()
                adversarial_embeddings = adversarial_embeddings + perturbation
                
                # Project to epsilon ball
                delta = adversarial_embeddings - original_embeddings
                delta = torch.clamp(delta, -epsilon, epsilon)
                adversarial_embeddings = original_embeddings + delta
            
            adversarial_embeddings.requires_grad_(True)
        
        return adversarial_embeddings.detach()
    
    def _forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using custom embeddings"""
        # Create a hook to replace embeddings
        def embedding_hook(module, input, output):
            # Return our adversarial embeddings instead of normal embeddings
            return embeddings
        
        hook = self.embedding_layer.register_forward_hook(embedding_hook)
        
        try:
            # Forward pass - handle different model architectures
            if hasattr(self.model, 'model'):
                # Hugging Face models (e.g., Gemma 2)
                model_outputs = self.model.model(inputs_embeds=embeddings)
                
                # Extract hidden states from BaseModelOutputWithPast or tuple
                if hasattr(model_outputs, 'last_hidden_state'):
                    hidden_states = model_outputs.last_hidden_state
                elif isinstance(model_outputs, tuple):
                    hidden_states = model_outputs[0]
                else:
                    hidden_states = model_outputs
                
                # Get logits from model
                if hasattr(self.model, 'lm_head'):
                    outputs = self.model.lm_head(hidden_states)
                elif hasattr(self.model, 'head'):
                    outputs = self.model.head(hidden_states)
                else:
                    # If no lm_head, return hidden states (for models without separate head)
                    outputs = hidden_states
            else:
                # Direct model call
                model_outputs = self.model(input_ids=input_ids)
                
                # Extract logits from BaseModelOutputWithPast or tuple
                if hasattr(model_outputs, 'logits'):
                    outputs = model_outputs.logits
                elif isinstance(model_outputs, tuple):
                    outputs = model_outputs[0]
                else:
                    outputs = model_outputs
        finally:
            hook.remove()
        
        return outputs
    
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        input_ids: torch.Tensor,
        target_layers: Optional[List[str]],
        target_loss: str
    ) -> torch.Tensor:
        """Compute loss for gradient attack"""
        if target_loss == "cross_entropy":
            # Standard cross-entropy loss
            logits = outputs[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        elif target_loss == "targeted":
            # Targeted loss (maximize probability of target tokens)
            # For now, use negative cross-entropy
            logits = outputs[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = -loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        elif target_loss == "untargeted":
            # Untargeted loss (minimize probability of original tokens)
            logits = outputs[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        else:
            raise ValueError(f"Unknown target_loss: {target_loss}")
        
        # Add layer-specific loss if target_layers specified
        if target_layers and self.instrumentation:
            layer_loss = 0.0
            for layer_name in target_layers:
                if layer_name in self.instrumentation.activations:
                    acts = self.instrumentation.activations[layer_name]
                    if isinstance(acts, list) and acts:
                        layer_act = acts[-1]
                        # Maximize variance (steer toward vulnerability)
                        layer_loss -= layer_act.var()
            loss = loss + 0.1 * layer_loss
        
        return loss
    
    def _embeddings_to_tokens(
        self,
        adversarial_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
        original_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Convert adversarial embeddings back to token IDs (nearest neighbor)"""
        # Find nearest token embeddings
        embedding_weights = self.embedding_layer.weight.data  # [vocab_size, embed_dim]
        
        # Reshape for comparison
        adv_flat = adversarial_embeddings.view(-1, adversarial_embeddings.size(-1))
        
        # Compute distances
        distances = torch.cdist(adv_flat, embedding_weights)
        
        # Find nearest neighbors
        nearest_tokens = distances.argmin(dim=-1)
        
        # Reshape back
        adv_input_ids = nearest_tokens.view(adversarial_embeddings.shape[:2])
        
        return adv_input_ids.to(original_input_ids.device)
    
    def _decode_adversarial_embeddings(
        self,
        adversarial_embeddings: torch.Tensor,
        original_input_ids: torch.Tensor
    ) -> str:
        """Decode adversarial embeddings to text (approximate)"""
        # Convert embeddings to tokens
        adv_input_ids = self._embeddings_to_tokens(adversarial_embeddings, None, original_input_ids)
        
        # Decode
        try:
            decoded = self.tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
            return decoded
        except:
            return "[Unable to decode adversarial embeddings]"
    
    def batch_attack(
        self,
        prompts: List[str],
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        max_iter: int = 10,
        target_layers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform attacks on multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.attack_prompt(
                prompt=prompt,
                attack_type=attack_type,
                epsilon=epsilon,
                max_iter=max_iter,
                target_layers=target_layers
            )
            results.append(result)
        return results
