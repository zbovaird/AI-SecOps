"""
Experiments for transferability analysis.

Phase 1: Open model CKA computation
Phase 2: Attack testing across all models
Phase 3: Correlation analysis and surrogate recommendations
"""

from .open_model_cka import OpenModelCKAExperiment, CKAMatrix
from .attack_testing import AttackTestingExperiment, ModelAttackResults
from .correlation import CorrelationAnalysis, TransferabilityReport

__all__ = [
    "OpenModelCKAExperiment",
    "CKAMatrix",
    "AttackTestingExperiment",
    "ModelAttackResults",
    "CorrelationAnalysis",
    "TransferabilityReport",
]
