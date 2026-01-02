"""
Base perturbation interface and registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type
import hashlib
import logging

logger = logging.getLogger("redteam.perturbations")


@dataclass
class PerturbationResult:
    """
    Result of applying a perturbation.
    
    Attributes:
        original: Original input
        perturbed: Perturbed output
        perturbation_type: Type of perturbation applied
        perturbation_id: Unique ID for this specific perturbation
        metadata: Additional info about the perturbation
        similarity: Semantic similarity to original (if computed)
    """
    original: str
    perturbed: str
    perturbation_type: str
    perturbation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: Optional[float] = None
    
    def __post_init__(self):
        if not self.perturbation_id:
            # Generate ID from hash of perturbation
            content = f"{self.perturbation_type}:{self.original}:{self.perturbed}"
            self.perturbation_id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "perturbation_type": self.perturbation_type,
            "perturbation_id": self.perturbation_id,
            "metadata": self.metadata,
            "similarity": self.similarity,
        }
    
    @property
    def diff(self) -> Dict[str, Any]:
        """Get a summary of what changed."""
        return {
            "type": self.perturbation_type,
            "original_length": len(self.original),
            "perturbed_length": len(self.perturbed),
            "length_delta": len(self.perturbed) - len(self.original),
            "changed": self.original != self.perturbed,
        }


class Perturbation(ABC):
    """
    Abstract base class for perturbations.
    """
    
    name: str = "base"
    category: str = "unknown"  # "prompt", "latent", "token"
    
    def __init__(self, **config):
        self.config = config
        self._logger = logging.getLogger(f"redteam.perturbations.{self.name}")
    
    @abstractmethod
    def apply(self, text: str, **kwargs) -> PerturbationResult:
        """
        Apply perturbation to input text.
        
        Args:
            text: Input text to perturb
            **kwargs: Additional arguments specific to perturbation type
            
        Returns:
            PerturbationResult with original and perturbed text
        """
        pass
    
    def apply_batch(self, texts: List[str], **kwargs) -> List[PerturbationResult]:
        """
        Apply perturbation to multiple texts.
        
        Default implementation calls apply() for each text.
        Override for optimized batch processing.
        """
        return [self.apply(text, **kwargs) for text in texts]
    
    def validate(self, result: PerturbationResult) -> bool:
        """
        Validate that perturbation result is valid.
        
        Override to add custom validation logic.
        """
        # Basic validation
        if not result.perturbed:
            self._logger.warning("Perturbation produced empty output")
            return False
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category!r})"


class PerturbationRegistry:
    """
    Registry for perturbation classes.
    """
    
    _perturbations: Dict[str, Type[Perturbation]] = {}
    
    @classmethod
    def register(cls, name: str, category: str = "prompt"):
        """Decorator to register a perturbation class."""
        def decorator(perturbation_cls: Type[Perturbation]):
            perturbation_cls.name = name
            perturbation_cls.category = category
            cls._perturbations[name] = perturbation_cls
            return perturbation_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Perturbation]:
        """Get perturbation class by name."""
        if name not in cls._perturbations:
            raise KeyError(f"Unknown perturbation: {name}. Available: {list(cls._perturbations.keys())}")
        return cls._perturbations[name]
    
    @classmethod
    def create(cls, name: str, **config) -> Perturbation:
        """Create perturbation instance by name."""
        return cls.get(name)(**config)
    
    @classmethod
    def available(cls, category: Optional[str] = None) -> List[str]:
        """List available perturbation names, optionally filtered by category."""
        if category is None:
            return list(cls._perturbations.keys())
        return [
            name for name, p_cls in cls._perturbations.items()
            if p_cls.category == category
        ]
    
    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        """Get perturbations organized by category."""
        result: Dict[str, List[str]] = {}
        for name, p_cls in cls._perturbations.items():
            cat = p_cls.category
            if cat not in result:
                result[cat] = []
            result[cat].append(name)
        return result


class CompositePerturbation(Perturbation):
    """
    Combines multiple perturbations in sequence.
    """
    
    name = "composite"
    category = "meta"
    
    def __init__(self, perturbations: List[Perturbation]):
        super().__init__()
        self.perturbations = perturbations
    
    def apply(self, text: str, **kwargs) -> PerturbationResult:
        """Apply all perturbations in sequence."""
        current = text
        all_metadata = []
        
        for p in self.perturbations:
            result = p.apply(current, **kwargs)
            current = result.perturbed
            all_metadata.append({
                "perturbation": p.name,
                "metadata": result.metadata,
            })
        
        return PerturbationResult(
            original=text,
            perturbed=current,
            perturbation_type="composite",
            metadata={
                "sequence": [p.name for p in self.perturbations],
                "steps": all_metadata,
            },
        )


class RandomChoicePerturbation(Perturbation):
    """
    Randomly selects one perturbation from a set to apply.
    """
    
    name = "random_choice"
    category = "meta"
    
    def __init__(self, perturbations: List[Perturbation], seed: Optional[int] = None):
        super().__init__()
        self.perturbations = perturbations
        self.seed = seed
        
        import random
        self._rng = random.Random(seed)
    
    def apply(self, text: str, **kwargs) -> PerturbationResult:
        """Apply randomly selected perturbation."""
        selected = self._rng.choice(self.perturbations)
        result = selected.apply(text, **kwargs)
        
        # Update metadata to indicate random selection
        result.metadata["selected_from"] = [p.name for p in self.perturbations]
        result.perturbation_type = f"random:{result.perturbation_type}"
        
        return result
