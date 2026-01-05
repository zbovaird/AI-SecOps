"""
Adversarial Perturbation Engine
Integrates with IBM ART for gradient-based attacks and latent-space distortion

Usage for Red Teaming:
---------------------
This module uses adversarial attacks to probe latent space boundaries, maximize
representational distortion, and search for inputs that steer toward vulnerability
basins.

Example Usage:
    from core.modules.adversarial_perturbation import AdversarialPerturbationEngine
    
    engine = AdversarialPerturbationEngine(model, tokenizer)
    
    # Generate adversarial perturbation
    adversarial_input = engine.generate_perturbation(
        original_input="Test prompt",
        attack_type="pgd",
        target_layer="layer_10"
    )
    
    # Track perturbation propagation
    propagation = engine.track_propagation(adversarial_input)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import defaultdict
import warnings

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    warnings.warn("IBM ART not available. Install with: pip install adversarial-robustness-toolbox")


class AdversarialPerturbationEngine:
    """Generate adversarial perturbations using ART and track propagation"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[str] = None,
        loss_fn: Optional[Callable] = None
    ):
        """
        Initialize adversarial perturbation engine
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer for the model
            device: Device to use ('cuda', 'cpu', or None for auto)
            loss_fn: Optional custom loss function
        """
        if not ART_AVAILABLE:
            raise ImportError("IBM ART is required. Install with: pip install adversarial-robustness-toolbox")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Loss function (default: cross-entropy on logits)
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        
        # ART classifier wrapper
        self.art_classifier = None
        self._setup_art_classifier()
        
        # Propagation tracking
        self.propagation_history: List[Dict[str, Any]] = []
    
    def _setup_art_classifier(self):
        """Setup ART PyTorch classifier"""
        # Create a wrapper function for model predictions
        def model_predict(x: np.ndarray) -> np.ndarray:
            """Predict function for ART"""
            self.model.eval()
            with torch.no_grad():
                # Convert numpy to tensor
                if isinstance(x, np.ndarray):
                    x_tensor = torch.from_numpy(x).to(self.device)
                else:
                    x_tensor = x.to(self.device)
                
                # Forward pass
                outputs = self.model(x_tensor)
                
                # Return logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                return logits.detach().cpu().numpy()
        
        # Create ART classifier
        try:
            self.art_classifier = PyTorchClassifier(
                model=self.model,
                loss=self.loss_fn,
                input_shape=(1,),  # Will be adjusted based on input
                nb_classes=2,  # Will be adjusted based on model
                device_type=str(self.device)
            )
        except Exception as e:
            warnings.warn(f"Failed to create ART classifier: {e}. Some attacks may not work.")
    
    def generate_perturbation(
        self,
        original_input: Union[str, torch.Tensor],
        attack_type: str = "fgsm",
        target_layer: Optional[str] = None,
        epsilon: float = 0.1,
        max_iter: int = 10,
        **attack_kwargs
    ) -> Dict[str, Any]:
        """
        Generate adversarial perturbation
        
        Args:
            original_input: Original input (text or tensor)
            attack_type: Type of attack ('fgsm', 'pgd', 'cw')
            target_layer: Optional target layer for layer-specific attacks
            epsilon: Perturbation magnitude
            max_iter: Maximum iterations for iterative attacks
            **attack_kwargs: Additional attack parameters
        
        Returns:
            Dictionary with adversarial input and metadata
        """
        # Convert text to tensor if needed
        if isinstance(original_input, str):
            inputs = self._tokenize_input(original_input)
        else:
            inputs = original_input
        
        inputs = inputs.to(self.device)
        
        # Generate perturbation based on attack type
        if attack_type.lower() == "fgsm":
            adversarial_input = self._fgsm_attack(inputs, epsilon, **attack_kwargs)
        elif attack_type.lower() == "pgd":
            adversarial_input = self._pgd_attack(inputs, epsilon, max_iter, **attack_kwargs)
        elif attack_type.lower() == "cw":
            adversarial_input = self._cw_attack(inputs, **attack_kwargs)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Compute perturbation
        perturbation = adversarial_input - inputs
        
        return {
            'original_input': inputs.detach().cpu(),
            'adversarial_input': adversarial_input.detach().cpu(),
            'perturbation': perturbation.detach().cpu(),
            'perturbation_norm': float(perturbation.norm().item()),
            'attack_type': attack_type,
            'epsilon': epsilon,
            'target_layer': target_layer
        }
    
    def _tokenize_input(self, text: str) -> torch.Tensor:
        """Tokenize text input"""
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return encoded['input_ids']
    
    def _fgsm_attack(
        self,
        inputs: torch.Tensor,
        epsilon: float,
        **kwargs
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        # Create dummy target (for loss computation)
        # In practice, you'd want a specific target
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        loss = self.loss_fn(logits, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        perturbation = epsilon * inputs.grad.sign()
        adversarial_input = inputs + perturbation
        
        return adversarial_input.detach()
    
    def _pgd_attack(
        self,
        inputs: torch.Tensor,
        epsilon: float,
        max_iter: int = 10,
        alpha: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        if alpha is None:
            alpha = epsilon / max_iter
        
        adversarial_input = inputs.clone()
        adversarial_input.requires_grad_(True)
        
        for _ in range(max_iter):
            # Forward pass
            outputs = self.model(adversarial_input)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Compute loss
            target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
            loss = self.loss_fn(logits, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial input
            with torch.no_grad():
                perturbation = alpha * adversarial_input.grad.sign()
                adversarial_input = adversarial_input + perturbation
                
                # Project to epsilon ball
                delta = adversarial_input - inputs
                delta = torch.clamp(delta, -epsilon, epsilon)
                adversarial_input = inputs + delta
            
            adversarial_input.requires_grad_(True)
        
        return adversarial_input.detach()
    
    def _cw_attack(
        self,
        inputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Carlini-Wagner L2 attack (simplified version)"""
        # Simplified CW attack - full implementation would use ART's CWL2Method
        # For now, use PGD as approximation
        epsilon = kwargs.get('epsilon', 0.1)
        max_iter = kwargs.get('max_iter', 10)
        return self._pgd_attack(inputs, epsilon, max_iter)
    
    def track_propagation(
        self,
        adversarial_input: torch.Tensor,
        instrumentation: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Track how perturbation propagates through layers
        
        Args:
            adversarial_input: Adversarial input tensor
            instrumentation: Optional ModelInstrumentation instance
        
        Returns:
            Dictionary with propagation metrics
        """
        if instrumentation is None:
            # Run forward pass and capture activations manually
            activations = {}
            hooks = []
            
            def get_activation(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        activations[name] = output[0].detach().clone()
                    else:
                        activations[name] = output.detach().clone()
                return hook
            
            # Register hooks
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    hooks.append(module.register_forward_hook(get_activation(name)))
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(adversarial_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        else:
            # Use instrumentation
            with torch.no_grad():
                _ = self.model(adversarial_input)
            activations = instrumentation.get_activations()
        
        # Compute propagation metrics
        propagation_metrics = {}
        for layer_name, activation in activations.items():
            if isinstance(activation, list):
                activation = activation[-1]  # Use last capture
            
            metrics = {
                'mean': float(activation.mean().item()),
                'std': float(activation.std().item()),
                'norm': float(activation.norm().item()),
                'max': float(activation.max().item()),
                'min': float(activation.min().item())
            }
            propagation_metrics[layer_name] = metrics
        
        # Store in history
        propagation_entry = {
            'input': adversarial_input.detach().cpu(),
            'layer_metrics': propagation_metrics,
            'timestamp': torch.cuda.Event().query() if torch.cuda.is_available() else None
        }
        self.propagation_history.append(propagation_entry)
        
        return propagation_entry
    
    def maximize_latent_distortion(
        self,
        original_input: Union[str, torch.Tensor],
        target_layer: str,
        max_iterations: int = 50,
        epsilon: float = 0.1
    ) -> Dict[str, Any]:
        """
        Search for perturbation that maximizes distortion in target layer
        
        Args:
            original_input: Original input
            target_layer: Target layer name
            max_iterations: Maximum search iterations
            epsilon: Perturbation budget
        
        Returns:
            Dictionary with best perturbation and distortion metrics
        """
        if isinstance(original_input, str):
            inputs = self._tokenize_input(original_input)
        else:
            inputs = original_input
        
        inputs = inputs.to(self.device)
        
        best_perturbation = None
        max_distortion = -float('inf')
        
        # Hook to capture target layer activation
        target_activation_original = None
        target_activation_adversarial = None
        
        def capture_target_layer(name):
            def hook(module, input, output):
                nonlocal target_activation_original, target_activation_adversarial
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                if target_activation_original is None:
                    target_activation_original = activation.detach().clone()
                else:
                    target_activation_adversarial = activation.detach().clone()
            return hook
        
        # Register hook
        target_module = dict(self.model.named_modules())[target_layer]
        hook = target_module.register_forward_hook(capture_target_layer(target_layer))
        
        try:
            # Baseline forward pass
            with torch.no_grad():
                _ = self.model(inputs)
            
            baseline_activation = target_activation_original.clone()
            
            # Iterative search
            for iteration in range(max_iterations):
                # Generate perturbation
                perturbation_result = self.generate_perturbation(
                    inputs,
                    attack_type="pgd",
                    epsilon=epsilon,
                    max_iter=5
                )
                
                adversarial_input = perturbation_result['adversarial_input'].to(self.device)
                
                # Forward pass with adversarial input
                with torch.no_grad():
                    _ = self.model(adversarial_input)
                
                # Compute distortion
                if target_activation_adversarial is not None:
                    distortion = torch.norm(
                        target_activation_adversarial - baseline_activation
                    ).item()
                    
                    if distortion > max_distortion:
                        max_distortion = distortion
                        best_perturbation = perturbation_result
                
                target_activation_adversarial = None
            
        finally:
            hook.remove()
        
        return {
            'best_perturbation': best_perturbation,
            'max_distortion': max_distortion,
            'target_layer': target_layer
        }
    
    def get_propagation_history(self) -> List[Dict[str, Any]]:
        """Get all propagation history"""
        return self.propagation_history
    
    def clear_history(self):
        """Clear propagation history"""
        self.propagation_history.clear()






