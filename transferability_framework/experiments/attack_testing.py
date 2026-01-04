"""
Phase 2: Attack Testing Experiment

Runs standard attack suite against all models (open and closed)
to measure attack success rates for transferability correlation.
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

from ..core.attack_suite import AttackSuite, AttackResult, STANDARD_ATTACKS
from ..core.model_loader import ModelLoader, SUPPORTED_MODELS
from ..core.api_clients import OpenAIClient, AnthropicClient, APIResponse

logger = logging.getLogger("transferability.experiments.attack_testing")


@dataclass
class ModelAttackResults:
    """Attack results for a single model."""
    model_id: str
    model_type: str  # "open" or "closed"
    
    # Individual attack results
    results: List[AttackResult] = field(default_factory=list)
    
    # Aggregate metrics
    compliance_rate: float = 0.0
    refusal_rate: float = 0.0
    unclear_rate: float = 0.0
    
    # Per-category metrics
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    total_attacks: int = 0
    avg_response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "compliance_rate": self.compliance_rate,
            "refusal_rate": self.refusal_rate,
            "unclear_rate": self.unclear_rate,
            "category_metrics": self.category_metrics,
            "timestamp": self.timestamp,
            "total_attacks": self.total_attacks,
            "avg_response_time_ms": self.avg_response_time_ms,
            "results": [r.to_dict() for r in self.results[:10]],  # Sample
        }
    
    def summary(self) -> str:
        lines = [
            f"Model: {self.model_id} ({self.model_type})",
            f"  Compliance rate: {self.compliance_rate:.1%}",
            f"  Refusal rate: {self.refusal_rate:.1%}",
            f"  Unclear rate: {self.unclear_rate:.1%}",
            f"  Avg response time: {self.avg_response_time_ms:.0f}ms",
        ]
        
        if self.category_metrics:
            lines.append("  Per-category compliance:")
            for cat, metrics in self.category_metrics.items():
                lines.append(f"    {cat}: {metrics.get('compliance_rate', 0):.1%}")
        
        return "\n".join(lines)


@dataclass
class AllModelResults:
    """Attack results across all models."""
    models: Dict[str, ModelAttackResults] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "models": {k: v.to_dict() for k, v in self.models.items()},
        }
    
    def get_compliance_vector(self) -> Dict[str, float]:
        """Get compliance rates as a vector for correlation."""
        return {k: v.compliance_rate for k, v in self.models.items()}
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ATTACK TESTING RESULTS",
            "=" * 60,
            "",
        ]
        
        # Rank by compliance rate (higher = more vulnerable)
        ranked = sorted(
            self.models.items(),
            key=lambda x: x[1].compliance_rate,
            reverse=True,
        )
        
        for model_id, results in ranked:
            lines.append(results.summary())
            lines.append("")
        
        return "\n".join(lines)


class AttackTestingExperiment:
    """
    Phase 2 Experiment: Test attacks across all models.
    
    Runs the standard attack suite against:
    - Open-weight models (via HuggingFace)
    - Closed-weight models (via API)
    
    Results are used in Phase 3 to correlate with CKA similarity.
    """
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        """
        Initialize experiment.
        
        Args:
            hf_token: HuggingFace token
            openai_key: OpenAI API key (optional)
            anthropic_key: Anthropic API key (optional)
            device: Device for open models
            dtype: Data type for open models
        """
        self.loader = ModelLoader(device=device, dtype=dtype, hf_token=hf_token)
        self.attack_suite = AttackSuite()
        
        # Initialize API clients if keys provided
        self.openai_client = None
        self.anthropic_client = None
        
        if openai_key:
            self.openai_client = OpenAIClient(api_key=openai_key)
        if anthropic_key:
            self.anthropic_client = AnthropicClient(api_key=anthropic_key)
        
        logger.info(f"AttackTestingExperiment initialized")
        logger.info(f"  Open models: {list(SUPPORTED_MODELS.keys())}")
        logger.info(f"  OpenAI: {'enabled' if self.openai_client else 'disabled'}")
        logger.info(f"  Anthropic: {'enabled' if self.anthropic_client else 'disabled'}")
    
    def _test_open_model(
        self,
        model_name: str,
        max_new_tokens: int = 200,
    ) -> ModelAttackResults:
        """
        Test attacks on an open-weight model.
        
        Args:
            model_name: Model key (e.g., "gemma2")
            max_new_tokens: Max tokens to generate
            
        Returns:
            ModelAttackResults
        """
        logger.info(f"\nTesting open model: {model_name}")
        
        config = SUPPORTED_MODELS[model_name]
        model, tokenizer = self.loader.load_model(config)
        
        results = ModelAttackResults(
            model_id=config.model_id,
            model_type="open",
            timestamp=datetime.now().isoformat(),
        )
        
        attack_results = []
        total_time = 0.0
        
        for i, attack in enumerate(self.attack_suite.attacks):
            start_time = time.time()
            
            # Tokenize
            inputs = tokenizer(
                attack.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            try:
                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Decode (skip input tokens)
                input_length = inputs['input_ids'].shape[1]
                response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                
            except Exception as e:
                logger.warning(f"Generation error on attack {attack.id}: {e}")
                response = f"[ERROR: {str(e)}]"
            
            generation_time_ms = (time.time() - start_time) * 1000
            total_time += generation_time_ms
            
            # Create result
            attack_result = self.attack_suite.create_result(
                attack,
                config.model_id,
                response,
                generation_time_ms,
            )
            attack_results.append(attack_result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.attack_suite.attacks)} attacks")
        
        # Compute aggregate metrics
        metrics = self.attack_suite.compute_success_rate(attack_results)
        category_metrics = self.attack_suite.compute_category_success(attack_results)
        
        results.results = attack_results
        results.compliance_rate = metrics["compliance_rate"]
        results.refusal_rate = metrics["refusal_rate"]
        results.unclear_rate = metrics["unclear_rate"]
        results.category_metrics = category_metrics
        results.total_attacks = len(attack_results)
        results.avg_response_time_ms = total_time / len(attack_results) if attack_results else 0
        
        # Cleanup
        self.loader._cleanup()
        
        return results
    
    def _test_api_model(
        self,
        client: Any,
        model_id: str,
    ) -> ModelAttackResults:
        """
        Test attacks on an API-based model.
        
        Args:
            client: API client (OpenAI or Anthropic)
            model_id: Model identifier
            
        Returns:
            ModelAttackResults
        """
        logger.info(f"\nTesting API model: {model_id}")
        
        results = ModelAttackResults(
            model_id=model_id,
            model_type="closed",
            timestamp=datetime.now().isoformat(),
        )
        
        attack_results = []
        total_time = 0.0
        
        for i, attack in enumerate(self.attack_suite.attacks):
            # Generate via API
            api_response = client.generate(attack.prompt)
            total_time += api_response.generation_time_ms
            
            # Create attack result
            attack_result = self.attack_suite.create_result(
                attack,
                model_id,
                api_response.response,
                api_response.generation_time_ms,
            )
            attack_results.append(attack_result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.attack_suite.attacks)} attacks")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Compute metrics
        metrics = self.attack_suite.compute_success_rate(attack_results)
        category_metrics = self.attack_suite.compute_category_success(attack_results)
        
        results.results = attack_results
        results.compliance_rate = metrics["compliance_rate"]
        results.refusal_rate = metrics["refusal_rate"]
        results.unclear_rate = metrics["unclear_rate"]
        results.category_metrics = category_metrics
        results.total_attacks = len(attack_results)
        results.avg_response_time_ms = total_time / len(attack_results) if attack_results else 0
        
        return results
    
    def run_open_models(
        self,
        model_names: List[str],
        save_path: Optional[str] = None,
    ) -> AllModelResults:
        """
        Run attacks on open-weight models only.
        
        Args:
            model_names: List of model keys to test
            save_path: Optional path to save results
            
        Returns:
            AllModelResults
        """
        logger.info(f"Testing {len(model_names)} open models")
        
        all_results = AllModelResults(timestamp=datetime.now().isoformat())
        
        for model_name in model_names:
            if model_name not in SUPPORTED_MODELS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            results = self._test_open_model(model_name)
            all_results.models[model_name] = results
        
        logger.info("\n" + all_results.summary())
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_results.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {save_path}")
        
        return all_results
    
    def run_closed_models(
        self,
        save_path: Optional[str] = None,
    ) -> AllModelResults:
        """
        Run attacks on closed-weight models (API-based).
        
        Requires API keys to be configured.
        
        Args:
            save_path: Optional path to save results
            
        Returns:
            AllModelResults
        """
        logger.info("Testing closed models via API")
        
        all_results = AllModelResults(timestamp=datetime.now().isoformat())
        
        if self.openai_client:
            results = self._test_api_model(self.openai_client, "gpt-4")
            all_results.models["openai"] = results
        
        if self.anthropic_client:
            results = self._test_api_model(self.anthropic_client, "claude-3-5-sonnet")
            all_results.models["anthropic"] = results
        
        logger.info("\n" + all_results.summary())
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_results.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {save_path}")
        
        return all_results
    
    def run_all(
        self,
        open_model_names: List[str],
        save_path: Optional[str] = None,
    ) -> AllModelResults:
        """
        Run attacks on all models (open and closed).
        
        Args:
            open_model_names: List of open model keys to test
            save_path: Optional path to save results
            
        Returns:
            AllModelResults
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Attack Testing on All Models")
        logger.info("=" * 60)
        
        all_results = AllModelResults(timestamp=datetime.now().isoformat())
        
        # Test open models
        for model_name in open_model_names:
            if model_name not in SUPPORTED_MODELS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            results = self._test_open_model(model_name)
            all_results.models[model_name] = results
        
        # Test closed models
        if self.openai_client:
            results = self._test_api_model(self.openai_client, "gpt-4")
            all_results.models["openai"] = results
        
        if self.anthropic_client:
            results = self._test_api_model(self.anthropic_client, "claude-3-5-sonnet")
            all_results.models["anthropic"] = results
        
        logger.info("\n" + all_results.summary())
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_results.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {save_path}")
        
        return all_results
