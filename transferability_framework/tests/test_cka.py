"""
Unit tests for CKA computation module.

Run with: pytest transferability_framework/tests/test_cka.py -v
"""

import pytest
import numpy as np


class TestCKAFunctions:
    """Test basic CKA functions."""
    
    def test_linear_cka_identical(self):
        """Identical representations should have CKA = 1."""
        from transferability_framework.core.cka import linear_cka
        
        X = np.random.randn(50, 100)
        cka_score = linear_cka(X, X)
        
        assert abs(cka_score - 1.0) < 1e-5, f"CKA of identical matrices should be 1, got {cka_score}"
    
    def test_linear_cka_range(self):
        """CKA should be in [0, 1]."""
        from transferability_framework.core.cka import linear_cka
        
        X = np.random.randn(50, 100)
        Y = np.random.randn(50, 80)
        
        cka_score = linear_cka(X, Y)
        
        assert 0 <= cka_score <= 1, f"CKA should be in [0, 1], got {cka_score}"
    
    def test_linear_cka_symmetric(self):
        """CKA should be symmetric: CKA(X, Y) = CKA(Y, X)."""
        from transferability_framework.core.cka import linear_cka
        
        X = np.random.randn(50, 100)
        Y = np.random.randn(50, 80)
        
        cka_xy = linear_cka(X, Y)
        cka_yx = linear_cka(Y, X)
        
        assert abs(cka_xy - cka_yx) < 1e-10, f"CKA should be symmetric"
    
    def test_rbf_cka_identical(self):
        """Identical representations should have RBF CKA ≈ 1."""
        from transferability_framework.core.cka import rbf_cka
        
        X = np.random.randn(50, 100)
        cka_score = rbf_cka(X, X)
        
        assert cka_score > 0.99, f"RBF CKA of identical matrices should be ~1, got {cka_score}"
    
    def test_rbf_cka_range(self):
        """RBF CKA should be in [0, 1]."""
        from transferability_framework.core.cka import rbf_cka
        
        X = np.random.randn(50, 100)
        Y = np.random.randn(50, 80)
        
        cka_score = rbf_cka(X, Y)
        
        assert 0 <= cka_score <= 1, f"RBF CKA should be in [0, 1], got {cka_score}"


class TestCKAClass:
    """Test CKA class."""
    
    def test_cka_init_linear(self):
        """Test CKA initialization with linear kernel."""
        from transferability_framework.core.cka import CKA
        
        cka = CKA(kernel="linear")
        assert cka.kernel == "linear"
    
    def test_cka_init_rbf(self):
        """Test CKA initialization with RBF kernel."""
        from transferability_framework.core.cka import CKA
        
        cka = CKA(kernel="rbf", sigma=1.0)
        assert cka.kernel == "rbf"
    
    def test_compare_representations(self):
        """Test representation comparison."""
        from transferability_framework.core.cka import CKA
        
        cka = CKA(kernel="linear")
        
        X = np.random.randn(50, 100)
        Y = np.random.randn(50, 80)
        
        score = cka.compare_representations(X, Y)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_compare_layer_representations(self):
        """Test layer-wise comparison."""
        from transferability_framework.core.cka import CKA
        
        cka = CKA(kernel="linear")
        
        # Mock layer states
        model_a_states = {
            0: np.random.randn(20, 64),
            1: np.random.randn(20, 64),
            2: np.random.randn(20, 64),
        }
        model_b_states = {
            0: np.random.randn(20, 128),
            1: np.random.randn(20, 128),
        }
        
        result = cka.compare_layer_representations(
            model_a_states,
            model_b_states,
            "model_a",
            "model_b",
        )
        
        assert result.model_a == "model_a"
        assert result.model_b == "model_b"
        assert len(result.layer_cka) > 0
        assert 0 <= result.mean_cka <= 1
    
    def test_compute_similarity_matrix(self):
        """Test similarity matrix computation."""
        from transferability_framework.core.cka import CKA
        
        cka = CKA(kernel="linear")
        
        # Mock model states
        model_states = {
            "model_a": {0: np.random.randn(20, 64)},
            "model_b": {0: np.random.randn(20, 128)},
            "model_c": {0: np.random.randn(20, 96)},
        }
        
        matrix, names = cka.compute_similarity_matrix(model_states)
        
        assert matrix.shape == (3, 3)
        assert len(names) == 3
        
        # Diagonal should be 1
        for i in range(3):
            assert abs(matrix[i, i] - 1.0) < 1e-5
        
        # Should be symmetric
        assert np.allclose(matrix, matrix.T)


class TestAttackSuite:
    """Test attack suite."""
    
    def test_attack_suite_init(self):
        """Test AttackSuite initialization."""
        from transferability_framework.core.attack_suite import AttackSuite, STANDARD_ATTACKS
        
        suite = AttackSuite()
        
        assert len(suite.attacks) == len(STANDARD_ATTACKS)
        assert len(suite.get_prompts()) == len(STANDARD_ATTACKS)
    
    def test_classify_response_refusal(self):
        """Test refusal detection."""
        from transferability_framework.core.attack_suite import AttackSuite
        
        suite = AttackSuite()
        
        response = "I cannot help with that request. It would be unethical and against my guidelines."
        result = suite.classify_response(response)
        
        assert result["is_refused"] is True
        assert result["is_complied"] is False
    
    def test_classify_response_compliance(self):
        """Test compliance detection."""
        from transferability_framework.core.attack_suite import AttackSuite
        
        suite = AttackSuite()
        
        response = "Here's how you can do that: First, you need to..."
        result = suite.classify_response(response)
        
        assert result["is_complied"] is True
        assert result["is_refused"] is False
    
    def test_get_prompts_by_category(self):
        """Test filtering by category."""
        from transferability_framework.core.attack_suite import AttackSuite, AttackCategory
        
        suite = AttackSuite()
        
        jailbreaks = suite.get_prompts_by_category(AttackCategory.JAILBREAK)
        
        assert len(jailbreaks) > 0
        assert all(a.category == AttackCategory.JAILBREAK for a in jailbreaks)


class TestModelLoader:
    """Test model loader (without actually loading models)."""
    
    def test_supported_models(self):
        """Test that supported models are defined."""
        from transferability_framework.core.model_loader import SUPPORTED_MODELS
        
        assert "gemma2" in SUPPORTED_MODELS
        assert "mistral" in SUPPORTED_MODELS
        assert "llama" in SUPPORTED_MODELS
    
    def test_model_config(self):
        """Test model config structure."""
        from transferability_framework.core.model_loader import SUPPORTED_MODELS
        
        for name, config in SUPPORTED_MODELS.items():
            assert config.model_id is not None
            assert config.short_name == name
    
    def test_sample_layers(self):
        """Test layer sampling function."""
        from transferability_framework.core.model_loader import sample_layers
        
        # Test with more layers than samples
        result = sample_layers(26, 5)
        assert len(result) == 5
        assert result == [0, 5, 10, 15, 20]
        
        # Test with fewer layers than samples
        result = sample_layers(3, 5)
        assert result == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
