"""
Unit tests for evaluators.
"""

import pytest
from redteam_framework.evaluators import (
    RefusalEvaluator,
    SemanticEvaluator,
    DegeneracyEvaluator,
    StabilityEvaluator,
    EvalResult,
)


class TestRefusalEvaluator:
    """Tests for RefusalEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return RefusalEvaluator()
    
    def test_detects_refusal(self, evaluator):
        baseline = "I cannot help with that request."
        adversarial = "I cannot help with that request."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "consistent_refusal"
        assert result.evidence["baseline_refused"] is True
    
    def test_detects_refusal_bypass(self, evaluator):
        baseline = "I cannot help with that request."
        adversarial = "Here's how to do that: First, you..."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "refusal_bypass"
        assert result.confidence >= 0.7
        assert "CRITICAL" in result.flags[0]
    
    def test_detects_no_refusal(self, evaluator):
        baseline = "Here's the answer: 42"
        adversarial = "The answer is 42"
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "no_refusal"
    
    def test_detects_induced_refusal(self, evaluator):
        baseline = "Here's how to do that..."
        adversarial = "I'm sorry, but I cannot assist with that."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "refusal_induced"
    
    def test_single_detection(self, evaluator):
        text = "I apologize, but I am unable to help with that request."
        result = evaluator.detect_single(text)
        assert result.label == "refused"


class TestSemanticEvaluator:
    """Tests for SemanticEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return SemanticEvaluator(use_embeddings=False)  # Use lexical for tests
    
    def test_identical_texts(self, evaluator):
        text = "The quick brown fox jumps over the lazy dog."
        result = evaluator.evaluate(text, text)
        assert result.label == "identical"
        assert result.confidence == 1.0
    
    def test_similar_texts(self, evaluator):
        baseline = "The quick brown fox jumps over the lazy dog."
        adversarial = "A quick brown fox jumps over a lazy dog."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.evidence["similarity"] > 0.7
    
    def test_different_texts(self, evaluator):
        baseline = "The capital of France is Paris."
        adversarial = "Machine learning is a subset of artificial intelligence."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.evidence["similarity"] < 0.5
    
    def test_empty_text(self, evaluator):
        result = evaluator.evaluate("Hello world", "")
        assert "length_ratio" in result.evidence


class TestDegeneracyEvaluator:
    """Tests for DegeneracyEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return DegeneracyEvaluator()
    
    def test_normal_output(self, evaluator):
        baseline = "This is a normal response with varied content."
        adversarial = "This is also a normal response with different words."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "normal"
    
    def test_repetitive_output(self, evaluator):
        baseline = "Normal output here."
        adversarial = "yes yes yes yes yes yes yes yes yes yes"
        
        result = evaluator.evaluate(baseline, adversarial)
        assert "repetition" in result.label.lower() or result.evidence["adversarial_repetition"] > 0.2
    
    def test_loop_detection(self, evaluator):
        baseline = "Normal response."
        adversarial = "abc abc abc abc abc abc abc abc abc abc"
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.evidence["loop_detected"] is True
    
    def test_empty_output(self, evaluator):
        baseline = "Normal response."
        adversarial = ""
        
        result = evaluator.evaluate(baseline, adversarial)
        assert "empty" in result.label.lower()
        assert "empty_output" in result.flags
    
    def test_analyze_single(self, evaluator):
        text = "This is a test with some content."
        metrics = evaluator.analyze_single(text)
        assert "repetition_ratio" in metrics
        assert "unique_ratio" in metrics


class TestStabilityEvaluator:
    """Tests for StabilityEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return StabilityEvaluator()
    
    def test_stable_output(self, evaluator):
        baseline = "Normal response."
        adversarial = "Another normal response."
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.label == "stable"
    
    def test_empty_output_detected(self, evaluator):
        baseline = "Normal response."
        adversarial = ""
        
        result = evaluator.evaluate(baseline, adversarial)
        assert "empty_output" in result.evidence.get("issues", [])
    
    def test_nan_string_detection(self, evaluator):
        baseline = "Normal response."
        adversarial = "The value is: nan"
        
        result = evaluator.evaluate(baseline, adversarial)
        assert result.evidence["adversarial_text_issues"]["has_nan_string"] is True
    
    def test_with_activation_context(self, evaluator):
        baseline = "Normal response."
        adversarial = "Another response."
        
        context = {
            "adversarial_activations": {
                0: {"mean": 0.5, "std": 0.1, "max": 2.0, "min": -1.0},
                1: {"mean": 0.6, "std": 0.2, "max": 3.0, "min": -2.0},
            }
        }
        
        result = evaluator.evaluate(baseline, adversarial, context)
        assert "adversarial_activation_check" in result.evidence


class TestEvalResult:
    """Tests for EvalResult dataclass."""
    
    def test_is_positive(self):
        positive = EvalResult(label="detected", confidence=0.7)
        negative = EvalResult(label="none", confidence=0.3)
        
        assert positive.is_positive is True
        assert negative.is_positive is False
    
    def test_to_dict(self):
        result = EvalResult(
            label="test",
            confidence=0.8,
            evidence={"key": "value"},
            flags=["flag1"],
        )
        d = result.to_dict()
        assert d["label"] == "test"
        assert d["confidence"] == 0.8
        assert d["evidence"]["key"] == "value"
        assert "flag1" in d["flags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





