"""
Unit tests for the schema module.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from redteam_framework.core.schema import (
    ModelFamily,
    BlockType,
    DecodingConfig,
    ModelConfig,
    RunConfig,
    BehaviorDelta,
    StateDelta,
    StabilityDelta,
    SampleRecord,
    ExperimentResult,
)


class TestDecodingConfig:
    """Tests for DecodingConfig."""
    
    def test_default_values(self):
        config = DecodingConfig()
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.do_sample is True
        
    def test_greedy_factory(self):
        config = DecodingConfig.greedy(max_new_tokens=128)
        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.max_new_tokens == 128
        
    def test_to_dict(self):
        config = DecodingConfig(temperature=0.5, top_p=0.9)
        d = config.to_dict()
        assert d["temperature"] == 0.5
        assert d["top_p"] == 0.9
        assert "seed" in d


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_from_model_id_gemma(self):
        config = ModelConfig.from_model_id("google/gemma-2-2b-it")
        assert config.model_id == "google/gemma-2-2b-it"
        assert config.family == ModelFamily.GEMMA
        
    def test_from_model_id_llama(self):
        config = ModelConfig.from_model_id("meta-llama/Llama-3-8B")
        assert config.family == ModelFamily.LLAMA
        
    def test_from_model_id_mistral(self):
        config = ModelConfig.from_model_id("mistralai/Mistral-7B-v0.1")
        assert config.family == ModelFamily.MISTRAL
        
    def test_from_model_id_unknown(self):
        config = ModelConfig.from_model_id("some-random/model")
        assert config.family == ModelFamily.OTHER
        
    def test_capability_flags_default(self):
        config = ModelConfig(model_id="test")
        assert config.has_attention_output is True
        assert config.has_kv_cache_access is True
        assert config.has_hidden_states is True
        
    def test_to_dict(self):
        config = ModelConfig.from_model_id("google/gemma-2-2b-it")
        d = config.to_dict()
        assert d["model_id"] == "google/gemma-2-2b-it"
        assert d["family"] == "gemma"  # Enum value


class TestRunConfig:
    """Tests for RunConfig."""
    
    def test_auto_run_id(self):
        config = RunConfig()
        assert config.run_id is not None
        assert len(config.run_id) == 8
        
    def test_auto_timestamp(self):
        config = RunConfig()
        assert config.timestamp is not None
        assert "T" in config.timestamp  # ISO format
        
    def test_to_dict_nested(self):
        config = RunConfig(
            experiment_name="test_exp",
            model=ModelConfig.from_model_id("test/model"),
            decoding=DecodingConfig(temperature=0.5),
        )
        d = config.to_dict()
        assert d["experiment_name"] == "test_exp"
        assert d["model"]["model_id"] == "test/model"
        assert d["decoding"]["temperature"] == 0.5


class TestBehaviorDelta:
    """Tests for BehaviorDelta."""
    
    def test_default_safe_state(self):
        delta = BehaviorDelta()
        assert delta.semantic_similarity == 1.0
        assert delta.semantic_changed is False
        assert delta.refusal_flipped is False
        
    def test_to_dict(self):
        delta = BehaviorDelta(semantic_similarity=0.8, semantic_changed=True)
        d = delta.to_dict()
        assert d["semantic_similarity"] == 0.8
        assert d["semantic_changed"] is True


class TestSampleRecord:
    """Tests for SampleRecord."""
    
    def test_auto_hash(self):
        sample = SampleRecord(output="test output")
        assert sample.output_hash is not None
        assert len(sample.output_hash) == 8
        
    def test_same_output_same_hash(self):
        s1 = SampleRecord(output="identical")
        s2 = SampleRecord(output="identical")
        assert s1.output_hash == s2.output_hash
        
    def test_different_output_different_hash(self):
        s1 = SampleRecord(output="output1")
        s2 = SampleRecord(output="output2")
        assert s1.output_hash != s2.output_hash
        
    def test_to_jsonl(self):
        sample = SampleRecord(
            prompt="test prompt",
            output="test output",
            behavior=BehaviorDelta(semantic_similarity=0.9),
        )
        jsonl = sample.to_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["prompt"] == "test prompt"
        assert parsed["behavior"]["semantic_similarity"] == 0.9


class TestExperimentResult:
    """Tests for ExperimentResult."""
    
    def test_compute_aggregates_empty(self):
        result = ExperimentResult(config=RunConfig())
        result.compute_aggregates()
        assert result.total_samples == 0
        assert result.refusal_rate == 0.0
        
    def test_compute_aggregates_with_samples(self):
        samples = [
            SampleRecord(behavior=BehaviorDelta(adversarial_refused=True, semantic_similarity=0.8)),
            SampleRecord(behavior=BehaviorDelta(adversarial_refused=False, semantic_similarity=0.9)),
            SampleRecord(behavior=BehaviorDelta(adversarial_refused=True, semantic_similarity=1.0)),
        ]
        result = ExperimentResult(config=RunConfig(), samples=samples)
        result.compute_aggregates()
        
        assert result.total_samples == 3
        assert result.refusal_rate == pytest.approx(2/3)
        assert result.mean_semantic_similarity == pytest.approx(0.9)
        
    def test_save_load_jsonl(self):
        config = RunConfig(experiment_name="test")
        samples = [
            SampleRecord(prompt="p1", output="o1"),
            SampleRecord(prompt="p2", output="o2"),
        ]
        result = ExperimentResult(config=config, samples=samples)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            result.save_jsonl(path)
            
            # Verify file exists and has correct structure
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # 1 config + 2 samples
            
            # First line should be config
            first = json.loads(lines[0])
            assert first["_type"] == "config"


class TestBlockType:
    """Tests for BlockType enum."""
    
    def test_enum_values(self):
        assert BlockType.EMBED.value == "embed"
        assert BlockType.ATTN.value == "attn"
        assert BlockType.MLP.value == "mlp"
        assert BlockType.NORM.value == "norm"
        assert BlockType.LM_HEAD.value == "lm_head"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
