"""
Experiment runner for coordinating red team experiments.

Handles output directory creation, result saving, and experiment orchestration.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict
from dataclasses import dataclass

from .schema import RunConfig, SampleRecord, ExperimentResult, ModelConfig, DecodingConfig
from .logging import get_logger, setup_logging, log_context


@dataclass
class OutputPaths:
    """Standardized output directory structure."""
    base_dir: Path
    metrics_file: Path
    artifacts_dir: Path
    plots_dir: Path
    
    @classmethod
    def create(cls, runs_dir: str, model_id: str, experiment_name: str) -> "OutputPaths":
        """Create output directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model_id for filesystem
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        
        base = Path(runs_dir) / f"{timestamp}_{safe_model_id}" / experiment_name
        base.mkdir(parents=True, exist_ok=True)
        
        artifacts = base / "artifacts"
        artifacts.mkdir(exist_ok=True)
        
        plots = base / "plots"
        plots.mkdir(exist_ok=True)
        
        return cls(
            base_dir=base,
            metrics_file=base / "metrics.jsonl",
            artifacts_dir=artifacts,
            plots_dir=plots,
        )


class ExperimentRunner:
    """
    Orchestrates experiment execution with standardized output handling.
    
    Usage:
        runner = ExperimentRunner(
            model_id="google/gemma-2-2b-it",
            experiment_name="decode_fragility",
        )
        
        with runner.run() as ctx:
            for prompt in prompts:
                result = ctx.process(prompt)
                ctx.log_sample(result)
    """
    
    def __init__(
        self,
        model_id: str,
        experiment_name: str,
        runs_dir: str = "./runs",
        model_config: Optional[ModelConfig] = None,
        decoding_config: Optional[DecodingConfig] = None,
        prompt_set_id: str = "",
        tags: Optional[List[str]] = None,
    ):
        self.model_id = model_id
        self.experiment_name = experiment_name
        self.runs_dir = runs_dir
        
        # Create config
        self.config = RunConfig(
            experiment_name=experiment_name,
            model=model_config or ModelConfig.from_model_id(model_id),
            decoding=decoding_config or DecodingConfig(),
            prompt_set_id=prompt_set_id,
            tags=tags or [],
        )
        
        # Setup output paths
        self.paths = OutputPaths.create(runs_dir, model_id, experiment_name)
        
        # Setup logging
        setup_logging(
            log_dir=str(self.paths.base_dir / "logs"),
            run_id=self.config.run_id,
        )
        self.logger = get_logger(
            name=experiment_name,
            run_id=self.config.run_id,
            experiment=experiment_name,
        )
        
        # Results collection
        self.samples: List[SampleRecord] = []
        self._started = False
        
    def start(self):
        """Begin experiment run."""
        if self._started:
            raise RuntimeError("Experiment already started")
        
        self._started = True
        self.logger.log_event(
            "experiment_start",
            model=self.model_id,
            experiment=self.experiment_name,
            run_id=self.config.run_id,
        )
        
        # Save initial config
        config_path = self.paths.base_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        return self
    
    def log_sample(self, sample: SampleRecord):
        """Log a sample result."""
        if not self._started:
            raise RuntimeError("Experiment not started. Call start() first.")
        
        self.samples.append(sample)
        
        # Append to JSONL immediately (crash-safe)
        with open(self.paths.metrics_file, "a") as f:
            f.write(sample.to_jsonl() + "\n")
        
        # Log summary
        self.logger.log_metric(
            "sample_processed",
            len(self.samples),
            prompt_id=sample.prompt_id,
            classification=sample.behavior.classification,
        )
    
    def save_artifact(self, name: str, data: Any, format: str = "json"):
        """Save an artifact (intermediate result, tensor, etc.)."""
        if format == "json":
            path = self.paths.artifacts_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "jsonl":
            path = self.paths.artifacts_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item, default=str) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.log_event("artifact_saved", name=name, path=str(path))
        return path
    
    def finish(self) -> ExperimentResult:
        """Complete experiment and return results."""
        if not self._started:
            raise RuntimeError("Experiment not started")
        
        result = ExperimentResult(config=self.config, samples=self.samples)
        result.compute_aggregates()
        
        # Save final summary
        summary_path = self.paths.base_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.log_event(
            "experiment_complete",
            total_samples=result.total_samples,
            refusal_rate=result.refusal_rate,
            bypass_count=result.refusal_bypass_count,
        )
        
        self._started = False
        return result
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.logger.log_error(
                f"Experiment failed: {exc_val}",
                exception=exc_val,
            )
        
        if self._started:
            try:
                self.finish()
            except Exception as e:
                self.logger.log_error(f"Error during finish: {e}", exception=e)
        
        return False  # Don't suppress exceptions


def run_experiment(
    model_id: str,
    experiment_name: str,
    prompts: List[str],
    process_fn: Callable[[str], SampleRecord],
    **runner_kwargs,
) -> ExperimentResult:
    """
    Convenience function to run an experiment.
    
    Args:
        model_id: Model identifier
        experiment_name: Name of experiment
        prompts: List of prompts to process
        process_fn: Function that takes a prompt and returns a SampleRecord
        **runner_kwargs: Additional arguments for ExperimentRunner
        
    Returns:
        ExperimentResult with all samples
    """
    runner = ExperimentRunner(
        model_id=model_id,
        experiment_name=experiment_name,
        **runner_kwargs,
    )
    
    with runner as ctx:
        for i, prompt in enumerate(prompts):
            with log_context(ctx.logger, "process_prompt", prompt_idx=i):
                sample = process_fn(prompt)
                sample.prompt_id = f"p{i}"
                ctx.log_sample(sample)
    
    return runner.finish()
