"""
Phase 3: Correlation Analysis and Surrogate Recommendations

Correlates CKA similarity with attack transferability to:
1. Find the best surrogate model for closed targets
2. Predict attack success on closed models from open model results
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
from scipy import stats

from .open_model_cka import CKAMatrix
from .attack_testing import AllModelResults, ModelAttackResults

logger = logging.getLogger("transferability.experiments.correlation")


@dataclass
class SurrogateRecommendation:
    """Recommendation for surrogate model selection."""
    target_model: str  # The closed model we're targeting
    
    # Ranked surrogates (best first)
    ranked_surrogates: List[Tuple[str, float, float]]  # (model, cka_sim, attack_similarity)
    
    # Best surrogate
    best_surrogate: str = ""
    best_cka_similarity: float = 0.0
    best_attack_similarity: float = 0.0
    
    # Predicted attack success
    predicted_compliance_rate: float = 0.0
    actual_compliance_rate: float = 0.0
    prediction_error: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "target_model": self.target_model,
            "best_surrogate": self.best_surrogate,
            "best_cka_similarity": self.best_cka_similarity,
            "best_attack_similarity": self.best_attack_similarity,
            "predicted_compliance_rate": self.predicted_compliance_rate,
            "actual_compliance_rate": self.actual_compliance_rate,
            "prediction_error": self.prediction_error,
            "ranked_surrogates": [
                {"model": m, "cka_sim": c, "attack_sim": a}
                for m, c, a in self.ranked_surrogates[:5]
            ],
        }
    
    def summary(self) -> str:
        lines = [
            f"Target: {self.target_model}",
            f"  Best surrogate: {self.best_surrogate}",
            f"    CKA similarity: {self.best_cka_similarity:.3f}",
            f"    Attack similarity: {self.best_attack_similarity:.3f}",
            f"  Predicted compliance: {self.predicted_compliance_rate:.1%}",
            f"  Actual compliance: {self.actual_compliance_rate:.1%}",
            f"  Prediction error: {self.prediction_error:.1%}",
        ]
        
        if len(self.ranked_surrogates) > 1:
            lines.append("  Alternative surrogates:")
            for model, cka, attack in self.ranked_surrogates[1:3]:
                lines.append(f"    {model}: CKA={cka:.3f}, Attack={attack:.3f}")
        
        return "\n".join(lines)


@dataclass
class TransferabilityReport:
    """Complete transferability analysis report."""
    
    # Correlation between CKA and attack transfer
    cka_attack_correlation: float = 0.0
    correlation_p_value: float = 0.0
    
    # Surrogate recommendations per target
    surrogate_recommendations: Dict[str, SurrogateRecommendation] = field(default_factory=dict)
    
    # Overall transferability matrix
    # (open_model_i, target_j) -> predicted_success
    transferability_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    open_models_tested: List[str] = field(default_factory=list)
    closed_models_tested: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "cka_attack_correlation": self.cka_attack_correlation,
            "correlation_p_value": self.correlation_p_value,
            "timestamp": self.timestamp,
            "open_models_tested": self.open_models_tested,
            "closed_models_tested": self.closed_models_tested,
            "surrogate_recommendations": {
                k: v.to_dict() for k, v in self.surrogate_recommendations.items()
            },
            "transferability_matrix": self.transferability_matrix,
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TRANSFERABILITY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"CKA-Attack Correlation: {self.cka_attack_correlation:.3f} (p={self.correlation_p_value:.4f})",
            "",
            "Interpretation:",
        ]
        
        if self.cka_attack_correlation > 0.7:
            lines.append("  STRONG positive correlation - CKA is a good predictor of attack transfer")
        elif self.cka_attack_correlation > 0.4:
            lines.append("  MODERATE positive correlation - CKA provides useful guidance")
        elif self.cka_attack_correlation > 0.1:
            lines.append("  WEAK positive correlation - CKA offers limited prediction")
        else:
            lines.append("  NO significant correlation - CKA may not predict attack transfer")
        
        lines.extend([
            "",
            "SURROGATE RECOMMENDATIONS",
            "-" * 40,
        ])
        
        for target, rec in self.surrogate_recommendations.items():
            lines.append("")
            lines.append(rec.summary())
        
        return "\n".join(lines)


class CorrelationAnalysis:
    """
    Phase 3: Analyze correlation between CKA similarity and attack transferability.
    
    This is the key insight: if CKA correlates with attack transfer, we can:
    1. Use open models as surrogates for closed ones
    2. Predict which attacks will transfer to closed models
    """
    
    def __init__(self):
        logger.info("CorrelationAnalysis initialized")
    
    def compute_attack_similarity(
        self,
        results_a: ModelAttackResults,
        results_b: ModelAttackResults,
    ) -> float:
        """
        Compute attack success similarity between two models.
        
        Higher similarity means attacks succeed/fail on both models similarly.
        
        Args:
            results_a: Attack results for model A
            results_b: Attack results for model B
            
        Returns:
            Similarity score [0, 1]
        """
        # Build lookup by prompt_id
        a_outcomes = {r.prompt_id: r.is_complied for r in results_a.results}
        b_outcomes = {r.prompt_id: r.is_complied for r in results_b.results}
        
        # Find common prompts
        common_prompts = set(a_outcomes.keys()) & set(b_outcomes.keys())
        
        if not common_prompts:
            return 0.0
        
        # Count agreements
        agreements = sum(
            1 for p in common_prompts
            if a_outcomes[p] == b_outcomes[p]
        )
        
        return agreements / len(common_prompts)
    
    def compute_cka_attack_correlation(
        self,
        cka_matrix: CKAMatrix,
        attack_results: AllModelResults,
    ) -> Tuple[float, float]:
        """
        Compute correlation between CKA similarity and attack similarity.
        
        Args:
            cka_matrix: CKA similarity matrix between open models
            attack_results: Attack results for all models
            
        Returns:
            Tuple of (correlation, p_value)
        """
        # Get models that are in both CKA matrix and attack results
        common_models = [
            m for m in cka_matrix.model_names
            if m in attack_results.models
        ]
        
        if len(common_models) < 2:
            logger.warning("Need at least 2 common models for correlation")
            return 0.0, 1.0
        
        cka_similarities = []
        attack_similarities = []
        
        for i, model_a in enumerate(common_models):
            for j, model_b in enumerate(common_models):
                if i < j:  # Avoid duplicates
                    # Get CKA similarity
                    cka_sim = cka_matrix.get_similarity(model_a, model_b)
                    
                    # Compute attack similarity
                    attack_sim = self.compute_attack_similarity(
                        attack_results.models[model_a],
                        attack_results.models[model_b],
                    )
                    
                    cka_similarities.append(cka_sim)
                    attack_similarities.append(attack_sim)
        
        if len(cka_similarities) < 2:
            return 0.0, 1.0
        
        # Pearson correlation
        correlation, p_value = stats.pearsonr(cka_similarities, attack_similarities)
        
        return correlation, p_value
    
    def find_best_surrogate(
        self,
        target_model: str,
        cka_matrix: CKAMatrix,
        attack_results: AllModelResults,
    ) -> SurrogateRecommendation:
        """
        Find the best open-weight surrogate for a closed model.
        
        Args:
            target_model: The closed model key (e.g., "openai", "anthropic")
            cka_matrix: CKA matrix between open models
            attack_results: Attack results for all models
            
        Returns:
            SurrogateRecommendation
        """
        if target_model not in attack_results.models:
            logger.warning(f"Target model {target_model} not in attack results")
            return SurrogateRecommendation(target_model=target_model, ranked_surrogates=[])
        
        target_results = attack_results.models[target_model]
        
        # Compute attack similarity with each open model
        candidates = []
        
        for open_model in cka_matrix.model_names:
            if open_model in attack_results.models:
                attack_sim = self.compute_attack_similarity(
                    target_results,
                    attack_results.models[open_model],
                )
                
                # Get average CKA with other open models as proxy for "representativeness"
                avg_cka = np.mean([
                    cka_matrix.get_similarity(open_model, other)
                    for other in cka_matrix.model_names
                    if other != open_model
                ])
                
                candidates.append((open_model, avg_cka, attack_sim))
        
        # Rank by attack similarity (primary) and CKA (secondary)
        ranked = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
        
        rec = SurrogateRecommendation(
            target_model=target_model,
            ranked_surrogates=ranked,
        )
        
        if ranked:
            best = ranked[0]
            rec.best_surrogate = best[0]
            rec.best_cka_similarity = best[1]
            rec.best_attack_similarity = best[2]
            
            # Predict based on surrogate's compliance rate
            rec.predicted_compliance_rate = attack_results.models[best[0]].compliance_rate
            rec.actual_compliance_rate = target_results.compliance_rate
            rec.prediction_error = abs(rec.predicted_compliance_rate - rec.actual_compliance_rate)
        
        return rec
    
    def predict_attack_success(
        self,
        surrogate_model: str,
        target_model: str,
        cka_similarity: float,
        attack_results: AllModelResults,
        prompt_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Predict attack success on target based on surrogate results.
        
        Args:
            surrogate_model: The open model used as surrogate
            target_model: The closed target model
            cka_similarity: CKA similarity between models
            attack_results: Attack results
            prompt_id: Specific prompt (None = overall prediction)
            
        Returns:
            Dict with prediction and confidence
        """
        if surrogate_model not in attack_results.models:
            return {"prediction": 0.0, "confidence": 0.0}
        
        surrogate_results = attack_results.models[surrogate_model]
        
        if prompt_id:
            # Find specific prompt result
            for result in surrogate_results.results:
                if result.prompt_id == prompt_id:
                    # Prediction = surrogate outcome, confidence = CKA similarity
                    return {
                        "prediction": 1.0 if result.is_complied else 0.0,
                        "confidence": cka_similarity,
                        "surrogate_complied": result.is_complied,
                    }
            return {"prediction": 0.0, "confidence": 0.0}
        else:
            # Overall prediction = surrogate compliance rate, weighted by CKA
            return {
                "prediction": surrogate_results.compliance_rate,
                "confidence": cka_similarity,
            }
    
    def run(
        self,
        cka_matrix: CKAMatrix,
        attack_results: AllModelResults,
        closed_model_keys: List[str] = ["openai", "anthropic"],
        save_path: Optional[str] = None,
    ) -> TransferabilityReport:
        """
        Run full correlation analysis.
        
        Args:
            cka_matrix: CKA similarity matrix
            attack_results: Attack results for all models
            closed_model_keys: Keys for closed models in attack_results
            save_path: Optional path to save report
            
        Returns:
            TransferabilityReport
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Correlation Analysis")
        logger.info("=" * 60)
        
        report = TransferabilityReport(
            timestamp=datetime.now().isoformat(),
            open_models_tested=cka_matrix.model_names.copy(),
            closed_models_tested=[k for k in closed_model_keys if k in attack_results.models],
        )
        
        # Compute CKA-attack correlation
        correlation, p_value = self.compute_cka_attack_correlation(cka_matrix, attack_results)
        report.cka_attack_correlation = correlation
        report.correlation_p_value = p_value
        
        logger.info(f"\nCKA-Attack Correlation: {correlation:.3f} (p={p_value:.4f})")
        
        # Find surrogates for each closed model
        for closed_model in closed_model_keys:
            if closed_model in attack_results.models:
                rec = self.find_best_surrogate(closed_model, cka_matrix, attack_results)
                report.surrogate_recommendations[closed_model] = rec
                logger.info(f"\n{rec.summary()}")
        
        # Build transferability matrix
        for open_model in cka_matrix.model_names:
            if open_model in attack_results.models:
                report.transferability_matrix[open_model] = {}
                
                for closed_model in closed_model_keys:
                    if closed_model in attack_results.models:
                        # Predict based on attack similarity
                        attack_sim = self.compute_attack_similarity(
                            attack_results.models[open_model],
                            attack_results.models[closed_model],
                        )
                        report.transferability_matrix[open_model][closed_model] = attack_sim
        
        logger.info("\n" + report.summary())
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"\nReport saved to: {save_path}")
        
        return report
    
    def quick_analysis(
        self,
        cka_matrix: CKAMatrix,
        attack_results: AllModelResults,
    ) -> Dict[str, Any]:
        """
        Quick correlation analysis without full report.
        
        Args:
            cka_matrix: CKA matrix
            attack_results: Attack results
            
        Returns:
            Dict with key metrics
        """
        correlation, p_value = self.compute_cka_attack_correlation(cka_matrix, attack_results)
        
        return {
            "cka_attack_correlation": correlation,
            "correlation_p_value": p_value,
            "is_significant": p_value < 0.05,
            "interpretation": (
                "strong" if correlation > 0.7 else
                "moderate" if correlation > 0.4 else
                "weak" if correlation > 0.1 else
                "none"
            ),
        }
