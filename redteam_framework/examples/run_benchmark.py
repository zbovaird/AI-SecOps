#!/usr/bin/env python3
"""
Run Cross-Model Benchmark

Compares vulnerability profiles across multiple models using standardized
experiments and generates a comparative scorecard.

Usage:
    # Benchmark two models
    python run_benchmark.py --models google/gemma-2-2b-it mistralai/Mistral-7B-Instruct-v0.1

    # Quick benchmark (reduced parameters)
    python run_benchmark.py --models gpt2 distilgpt2 --quick

    # Full benchmark with all experiments
    python run_benchmark.py --models MODEL1 MODEL2 MODEL3 --full
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-model red team benchmark"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Model IDs to benchmark (HuggingFace hub or local paths)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: reduced parameters, fewer experiments"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Full mode: all experiments with full parameters"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        choices=["all", "fragility", "logit_lens", "drift", "attention", "kv_cache"],
        help="Which experiments to run (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("RED TEAM FRAMEWORK - Cross-Model Benchmark")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Output: {args.output}")
    print(f"Mode: {'quick' if args.quick else 'full' if args.full else 'standard'}")
    print(f"Seed: {args.seed}")
    
    from redteam_framework.benchmark import BenchmarkHarness, BenchmarkConfig
    from redteam_framework.experiments import DecodeGridConfig
    
    # Configure based on mode
    if args.quick:
        config = BenchmarkConfig(
            model_ids=args.models,
            output_dir=args.output,
            seed=args.seed,
            decode_grid=DecodeGridConfig(
                temperatures=[0.0, 1.0],
                top_p_values=[1.0],
                max_new_tokens=50,
            ),
            max_multiturn_turns=4,
            run_attention_routing=False,
            run_kv_cache=False,
        )
    elif args.full:
        config = BenchmarkConfig(
            model_ids=args.models,
            output_dir=args.output,
            seed=args.seed,
            decode_grid=DecodeGridConfig(
                temperatures=[0.0, 0.3, 0.5, 0.7, 1.0],
                top_p_values=[0.7, 0.9, 0.95, 1.0],
                repetition_penalties=[1.0, 1.1],
                max_new_tokens=200,
            ),
            max_multiturn_turns=10,
        )
    else:
        config = BenchmarkConfig(
            model_ids=args.models,
            output_dir=args.output,
            seed=args.seed,
        )
    
    # Filter experiments if specified
    if args.experiments != "all":
        config.run_decode_fragility = args.experiments == "fragility"
        config.run_logit_lens = args.experiments == "logit_lens"
        config.run_multiturn_drift = args.experiments == "drift"
        config.run_attention_routing = args.experiments == "attention"
        config.run_kv_cache = args.experiments == "kv_cache"
    
    # Run benchmark
    harness = BenchmarkHarness(config)
    
    def progress(model_id, current, total):
        print(f"\n{'='*60}")
        print(f"Model {current+1}/{total}: {model_id}")
        print(f"{'='*60}")
    
    result = harness.run(progress_callback=progress)
    
    # Print scorecard
    print("\n")
    print(harness.generate_scorecard(result))
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    if result.vulnerability_ranking:
        print(f"\nMost vulnerable: {result.vulnerability_ranking[0]}")
        if len(result.vulnerability_ranking) > 1:
            print(f"Least vulnerable: {result.vulnerability_ranking[-1]}")


if __name__ == "__main__":
    main()





