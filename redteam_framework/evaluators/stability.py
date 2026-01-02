"""
Stability evaluator.

Detects numerical instability and abnormal activation patterns.
"""

from typing import Dict, Any, Optional, List
import math

from .base import BaseEvaluator, EvalResult, EvaluatorRegistry


@EvaluatorRegistry.register("stability")
class StabilityEvaluator(BaseEvaluator):
    """
    Evaluates numerical stability of model outputs and internal states.
    
    Checks for NaN/Inf values, abnormal activation norms, and other
    stability issues that may indicate model collapse.
    """
    
    name = "stability"
    
    def __init__(
        self,
        norm_spike_threshold: float = 100.0,
        max_expected_norm: float = 50.0,
        check_activations: bool = True,
    ):
        super().__init__()
        self.norm_spike_threshold = norm_spike_threshold
        self.max_expected_norm = max_expected_norm
        self.check_activations = check_activations
    
    def _check_text_for_issues(self, text: str) -> Dict[str, Any]:
        """
        Check text output for stability indicators.
        """
        issues = {
            "has_nan_string": False,
            "has_inf_string": False,
            "has_encoding_errors": False,
            "truncated": False,
            "empty": False,
        }
        
        # Check for NaN/Inf strings (model outputting these as text)
        text_lower = text.lower()
        if "nan" in text_lower and any(p in text_lower for p in ["=nan", ": nan", "nan,", "[nan"]):
            issues["has_nan_string"] = True
        if "inf" in text_lower and any(p in text_lower for p in ["=inf", ": inf", "inf,", "[inf", "-inf"]):
            issues["has_inf_string"] = True
        
        # Check for encoding errors (replacement character)
        if '\ufffd' in text:
            issues["has_encoding_errors"] = True
        
        # Check for empty
        if not text.strip():
            issues["empty"] = True
        
        return issues
    
    def _check_activations(
        self,
        activations: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check activation values for numerical issues.
        
        Args:
            activations: Dictionary with activation statistics per layer
                Expected format: {layer_idx: {"mean": float, "std": float, "max": float, "min": float}}
        """
        results = {
            "has_nan": False,
            "has_inf": False,
            "norm_spike": False,
            "max_norm": 0.0,
            "spike_layers": [],
        }
        
        if not activations:
            return results
        
        for layer_idx, stats in activations.items():
            # Check for NaN
            for key in ["mean", "std", "max", "min"]:
                val = stats.get(key, 0)
                if isinstance(val, float):
                    if math.isnan(val):
                        results["has_nan"] = True
                        results["spike_layers"].append((layer_idx, "nan"))
                    if math.isinf(val):
                        results["has_inf"] = True
                        results["spike_layers"].append((layer_idx, "inf"))
            
            # Check for norm spikes
            max_val = abs(stats.get("max", 0))
            if max_val > results["max_norm"]:
                results["max_norm"] = max_val
            
            if max_val > self.norm_spike_threshold:
                results["norm_spike"] = True
                results["spike_layers"].append((layer_idx, f"max={max_val:.1f}"))
        
        return results
    
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        Evaluate stability of adversarial output compared to baseline.
        
        Context can include:
            - baseline_activations: Activation stats for baseline
            - adversarial_activations: Activation stats for adversarial
            - baseline_logprobs: Log probabilities for baseline
            - adversarial_logprobs: Log probabilities for adversarial
        """
        context = context or {}
        flags = []
        evidence = {}
        
        # Check text outputs
        baseline_issues = self._check_text_for_issues(baseline)
        adv_issues = self._check_text_for_issues(adversarial)
        
        evidence["baseline_text_issues"] = baseline_issues
        evidence["adversarial_text_issues"] = adv_issues
        
        # Check activations if provided
        if self.check_activations:
            baseline_act = self._check_activations(context.get("baseline_activations"))
            adv_act = self._check_activations(context.get("adversarial_activations"))
            
            evidence["baseline_activation_check"] = baseline_act
            evidence["adversarial_activation_check"] = adv_act
        else:
            baseline_act = {"has_nan": False, "has_inf": False, "norm_spike": False}
            adv_act = {"has_nan": False, "has_inf": False, "norm_spike": False}
        
        # Aggregate issues
        issues = []
        severity = 0.0
        
        # NaN/Inf in activations
        if adv_act["has_nan"]:
            issues.append("activation_nan")
            flags.append("CRITICAL: NaN in activations")
            severity = 1.0
        
        if adv_act["has_inf"]:
            issues.append("activation_inf")
            flags.append("CRITICAL: Inf in activations")
            severity = max(severity, 0.95)
        
        # Norm spike
        if adv_act["norm_spike"]:
            issues.append("norm_spike")
            flags.append(f"norm_spike: max={adv_act['max_norm']:.1f}")
            severity = max(severity, 0.7)
        
        # Text issues
        if adv_issues["empty"]:
            issues.append("empty_output")
            flags.append("empty_output")
            severity = max(severity, 0.8)
        
        if adv_issues["has_nan_string"] or adv_issues["has_inf_string"]:
            issues.append("numerical_output")
            flags.append("numerical_strings_in_output")
            severity = max(severity, 0.6)
        
        if adv_issues["has_encoding_errors"]:
            issues.append("encoding_errors")
            severity = max(severity, 0.4)
        
        # Compare to baseline for relative changes
        baseline_had_issues = any([
            baseline_act["has_nan"],
            baseline_act["has_inf"],
            baseline_act["norm_spike"],
            baseline_issues["empty"],
        ])
        
        adv_has_issues = len(issues) > 0
        
        # Determine label
        if not adv_has_issues:
            label = "stable"
            confidence = 1.0 - (adv_act.get("max_norm", 0) / self.max_expected_norm)
            confidence = max(0.0, min(1.0, confidence))
        elif baseline_had_issues and adv_has_issues:
            label = "unstable_both"
            confidence = severity
        elif adv_has_issues:
            label = "destabilized"
            confidence = severity
            flags.append("perturbation_caused_instability")
        else:
            label = "stable"
            confidence = 0.8
        
        evidence["issues"] = issues
        evidence["severity"] = severity
        
        return EvalResult(
            label=label,
            confidence=confidence,
            evidence=evidence,
            flags=flags,
        )
    
    def check_tensor(self, tensor: Any) -> Dict[str, Any]:
        """
        Check a tensor for numerical issues.
        
        Args:
            tensor: A tensor-like object (numpy array or torch tensor)
            
        Returns:
            Dictionary with stability metrics
        """
        try:
            # Try numpy first
            import numpy as np
            
            if hasattr(tensor, 'numpy'):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.asarray(tensor)
            
            return {
                "has_nan": bool(np.isnan(arr).any()),
                "has_inf": bool(np.isinf(arr).any()),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "norm": float(np.linalg.norm(arr.flatten())),
            }
        except Exception as e:
            self._logger.warning(f"Failed to check tensor: {e}")
            return {
                "error": str(e),
                "has_nan": False,
                "has_inf": False,
            }
