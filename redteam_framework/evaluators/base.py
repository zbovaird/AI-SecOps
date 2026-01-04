"""
Base evaluator interface and registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol, Type
import logging

logger = logging.getLogger("redteam.evaluators")


@dataclass
class EvalResult:
    """
    Standardized evaluation result.
    
    Attributes:
        label: Classification label (e.g., "refusal", "compliance", "degraded")
        confidence: Confidence score 0-1
        evidence: Supporting data for the classification
        flags: Warning flags or edge cases detected
    """
    label: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "flags": self.flags,
        }
    
    @property
    def is_positive(self) -> bool:
        """Whether this result indicates a positive detection."""
        return self.confidence >= 0.5
    
    def __repr__(self):
        flags_str = f", flags={self.flags}" if self.flags else ""
        return f"EvalResult(label={self.label!r}, confidence={self.confidence:.2f}{flags_str})"


class Evaluator(Protocol):
    """
    Protocol for evaluators.
    
    Evaluators analyze baseline and adversarial outputs to detect changes.
    """
    
    name: str
    
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        Evaluate baseline vs adversarial output.
        
        Args:
            baseline: Baseline model output (no perturbation)
            adversarial: Adversarial model output (with perturbation)
            context: Additional context (prompt, model info, etc.)
            
        Returns:
            EvalResult with classification and confidence
        """
        ...


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators with common functionality.
    """
    
    name: str = "base"
    
    def __init__(self, **config):
        """Initialize with optional configuration."""
        self.config = config
        self._logger = logging.getLogger(f"redteam.evaluators.{self.name}")
    
    @abstractmethod
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate baseline vs adversarial output."""
        pass
    
    def _log_eval(self, result: EvalResult, **extra):
        """Log evaluation result."""
        self._logger.debug(
            f"{self.name}: {result.label} (conf={result.confidence:.2f})",
            extra={"eval_result": result.to_dict(), **extra}
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


class CompositeEvaluator(BaseEvaluator):
    """
    Combines multiple evaluators into a single evaluation.
    """
    
    name = "composite"
    
    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        aggregation: str = "any",  # "any", "all", "majority", "weighted"
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.evaluators = evaluators
        self.aggregation = aggregation
        self.weights = weights or {}
        
    def evaluate(
        self,
        baseline: str,
        adversarial: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Run all evaluators and aggregate results."""
        results = {}
        for evaluator in self.evaluators:
            try:
                results[evaluator.name] = evaluator.evaluate(baseline, adversarial, context)
            except Exception as e:
                self._logger.error(f"Evaluator {evaluator.name} failed: {e}")
                results[evaluator.name] = EvalResult(
                    label="error",
                    confidence=0.0,
                    flags=[f"evaluator_error: {e}"]
                )
        
        # Aggregate based on strategy
        if self.aggregation == "any":
            # Positive if any evaluator is positive
            positives = [r for r in results.values() if r.is_positive]
            if positives:
                # Use highest confidence positive
                best = max(positives, key=lambda r: r.confidence)
                return EvalResult(
                    label=best.label,
                    confidence=best.confidence,
                    evidence={"sub_results": {k: v.to_dict() for k, v in results.items()}},
                    flags=list(set(f for r in results.values() for f in r.flags))
                )
            else:
                return EvalResult(
                    label="none",
                    confidence=1.0 - max(r.confidence for r in results.values()),
                    evidence={"sub_results": {k: v.to_dict() for k, v in results.items()}},
                )
                
        elif self.aggregation == "all":
            # Positive only if all evaluators agree
            if all(r.is_positive for r in results.values()):
                min_conf = min(r.confidence for r in results.values())
                labels = list(set(r.label for r in results.values()))
                return EvalResult(
                    label=labels[0] if len(labels) == 1 else "mixed",
                    confidence=min_conf,
                    evidence={"sub_results": {k: v.to_dict() for k, v in results.items()}},
                )
            else:
                return EvalResult(
                    label="none",
                    confidence=1.0,
                    evidence={"sub_results": {k: v.to_dict() for k, v in results.items()}},
                )
                
        elif self.aggregation == "weighted":
            # Weighted average of confidences
            total_weight = sum(self.weights.get(e.name, 1.0) for e in self.evaluators)
            weighted_conf = sum(
                results[e.name].confidence * self.weights.get(e.name, 1.0)
                for e in self.evaluators
            ) / total_weight
            
            # Label from highest weighted confidence
            best_name = max(
                results.keys(),
                key=lambda k: results[k].confidence * self.weights.get(k, 1.0)
            )
            return EvalResult(
                label=results[best_name].label,
                confidence=weighted_conf,
                evidence={"sub_results": {k: v.to_dict() for k, v in results.items()}},
            )
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class EvaluatorRegistry:
    """
    Registry for evaluator classes.
    """
    
    _evaluators: Dict[str, Type[BaseEvaluator]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an evaluator class."""
        def decorator(evaluator_cls: Type[BaseEvaluator]):
            cls._evaluators[name] = evaluator_cls
            return evaluator_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseEvaluator]:
        """Get evaluator class by name."""
        if name not in cls._evaluators:
            raise KeyError(f"Unknown evaluator: {name}. Available: {list(cls._evaluators.keys())}")
        return cls._evaluators[name]
    
    @classmethod
    def create(cls, name: str, **config) -> BaseEvaluator:
        """Create evaluator instance by name."""
        return cls.get(name)(**config)
    
    @classmethod
    def available(cls) -> List[str]:
        """List available evaluator names."""
        return list(cls._evaluators.keys())





