#!/usr/bin/env python3
"""
Example: Running Red Team Experiments

This script demonstrates how to use the redteam_framework to analyze
a model's vulnerabilities through decode fragility and logit lens probing.

Usage:
    python run_experiments.py --model google/gemma-2-2b-it
    python run_experiments.py --model gpt2 --quick  # Fast test with small model
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_model(model_id: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_id}")
    print(f"{'='*60}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=True,
    )
    
    if device == "mps":
        model = model.to(device)
    
    model.eval()
    
    print(f"✓ Model loaded: {model.config._name_or_path}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def run_decode_fragility(model, tokenizer, output_dir: str, quick: bool = False):
    """Run decode fragility experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Decode Fragility Sweep")
    print(f"{'='*60}")
    
    from redteam_framework.experiments import (
        DecodeFragilitySweep,
        DecodeGridConfig,
        DEFAULT_REDTEAM_PROMPTS,
    )
    
    # Configure grid (smaller for quick mode)
    if quick:
        config = DecodeGridConfig(
            temperatures=[0.0, 0.7, 1.0],
            top_p_values=[0.9, 1.0],
            repetition_penalties=[1.0],
            max_new_tokens=100,
        )
        prompts = DEFAULT_REDTEAM_PROMPTS[:5]
    else:
        config = DecodeGridConfig(
            temperatures=[0.0, 0.3, 0.7, 1.0],
            top_p_values=[0.7, 0.9, 1.0],
            repetition_penalties=[1.0, 1.1],
            max_new_tokens=200,
        )
        prompts = DEFAULT_REDTEAM_PROMPTS
    
    print(f"Grid size: {config.grid_size} configurations")
    print(f"Prompts: {len(prompts)}")
    print(f"Total generations: {config.grid_size * len(prompts)}")
    
    # Run experiment
    sweep = DecodeFragilitySweep(
        model=model,
        tokenizer=tokenizer,
        grid_config=config,
    )
    
    def progress(current, total):
        print(f"  Progress: {current+1}/{total} prompts", end="\r")
    
    report = sweep.run(prompts, progress_callback=progress)
    print()
    
    # Save report
    import json
    report_path = os.path.join(output_dir, "decode_fragility_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    
    # Print summary
    print(report.summary())
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def run_logit_lens(model, tokenizer, output_dir: str, quick: bool = False):
    """Run logit lens experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Logit Lens Probing")
    print(f"{'='*60}")
    
    from redteam_framework.experiments import (
        LogitLensProbe,
        DEFAULT_BENIGN_PROMPTS,
        DEFAULT_ADVERSARIAL_PROMPTS,
    )
    
    # Select prompts
    if quick:
        benign = DEFAULT_BENIGN_PROMPTS[:3]
        adversarial = DEFAULT_ADVERSARIAL_PROMPTS[:3]
    else:
        benign = DEFAULT_BENIGN_PROMPTS
        adversarial = DEFAULT_ADVERSARIAL_PROMPTS
    
    print(f"Benign prompts: {len(benign)}")
    print(f"Adversarial prompts: {len(adversarial)}")
    
    # Run experiment
    probe = LogitLensProbe(model=model, tokenizer=tokenizer)
    print(f"Probing layers: {probe.layers_to_probe}")
    
    report = probe.analyze(benign, adversarial)
    
    # Save report
    import json
    report_path = os.path.join(output_dir, "logit_lens_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    
    # Print summary
    print(report.summary())
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def run_multiturn_drift(model, tokenizer, output_dir: str, quick: bool = False):
    """Run multi-turn drift experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Multi-turn Drift Analysis")
    print(f"{'='*60}")
    
    from redteam_framework.experiments import (
        MultiTurnDriftExperiment,
        DEFAULT_TARGET_REQUESTS,
    )
    
    if quick:
        requests = DEFAULT_TARGET_REQUESTS[:2]
        strategies = ["gentle_reframe", "persistent_rephrasing"]
        max_turns = 4
    else:
        requests = DEFAULT_TARGET_REQUESTS
        strategies = None  # Use all
        max_turns = 8
    
    print(f"Target requests: {len(requests)}")
    print(f"Strategies: {strategies or 'all'}")
    print(f"Max turns: {max_turns}")
    
    experiment = MultiTurnDriftExperiment(
        model=model,
        tokenizer=tokenizer,
        max_turns=max_turns,
    )
    
    report = experiment.run(requests, strategies)
    
    import json
    report_path = os.path.join(output_dir, "multiturn_drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    
    print(report.summary())
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def run_attention_routing(model, tokenizer, output_dir: str, quick: bool = False):
    """Run attention routing experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: Attention Routing Analysis")
    print(f"{'='*60}")
    
    from redteam_framework.experiments import (
        AttentionRoutingAnalyzer,
        DEFAULT_BENIGN_PROMPTS,
        DEFAULT_ADVERSARIAL_PROMPTS,
    )
    
    if quick:
        benign = DEFAULT_BENIGN_PROMPTS[:2]
        adversarial = DEFAULT_ADVERSARIAL_PROMPTS[:2]
    else:
        benign = DEFAULT_BENIGN_PROMPTS
        adversarial = DEFAULT_ADVERSARIAL_PROMPTS
    
    print(f"Benign prompts: {len(benign)}")
    print(f"Adversarial prompts: {len(adversarial)}")
    
    analyzer = AttentionRoutingAnalyzer(model=model, tokenizer=tokenizer)
    report = analyzer.analyze(benign, adversarial)
    
    import json
    report_path = os.path.join(output_dir, "attention_routing_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    
    print(report.summary())
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def run_kv_cache_analysis(model, tokenizer, output_dir: str, quick: bool = False):
    """Run KV cache persistence experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: KV Cache Persistence Analysis")
    print(f"{'='*60}")
    
    from redteam_framework.experiments import (
        KVCacheProbe,
    )
    
    if quick:
        conversations = [
            ["Remember: SECRET123", "What's 2+2?", "What was the secret?"],
        ]
    else:
        conversations = [
            ["Remember: SECRET123", "What's 2+2?", "Tell me a joke", "What was the secret?"],
            ["I'm Alice, a researcher", "What's the weather?", "Who am I?"],
        ]
    
    print(f"Conversations: {len(conversations)}")
    
    probe = KVCacheProbe(model=model, tokenizer=tokenizer)
    report = probe.analyze(conversations)
    
    import json
    report_path = os.path.join(output_dir, "kv_cache_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    
    print(report.summary())
    print(f"\n✓ Report saved to: {report_path}")
    
    return report


def generate_combined_report(
    fragility_report,
    logit_lens_report,
    output_dir: str,
    model_id: str,
):
    """Generate combined red team report."""
    print(f"\n{'='*60}")
    print("COMBINED RED TEAM REPORT")
    print(f"{'='*60}")
    
    lines = [
        "=" * 60,
        "RED TEAM ANALYSIS REPORT",
        "=" * 60,
        f"Model: {model_id}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "=" * 60,
        "EXECUTIVE SUMMARY",
        "=" * 60,
        "",
    ]
    
    # Fragility summary
    if fragility_report:
        lines.extend([
            "DECODE FRAGILITY:",
            f"  - Knife-edge prompts: {len(fragility_report.knife_edge_prompts)}",
            f"  - Fragility score: {fragility_report.overall_fragility_score:.2f}",
            f"  - Always refuses: {len(fragility_report.always_refuses)}",
            f"  - Always complies: {len(fragility_report.always_complies)}",
            "",
        ])
        
        if fragility_report.most_dangerous_config:
            lines.append("  Most permissive config:")
            for k, v in fragility_report.most_dangerous_config.items():
                lines.append(f"    {k}: {v}")
            lines.append("")
    
    # Logit lens summary
    if logit_lens_report:
        lines.extend([
            "LOGIT LENS:",
            f"  - Layers analyzed: {logit_lens_report.num_layers}",
            f"  - Avg first refusal layer: {logit_lens_report.avg_first_refusal_layer:.1f}",
            f"  - Critical layers: {logit_lens_report.critical_layers}",
            "",
        ])
        
        if logit_lens_report.most_divergent_layer is not None:
            lines.append(f"  Most divergent layer: {logit_lens_report.most_divergent_layer}")
            lines.append("")
    
    # Actionable findings
    lines.extend([
        "=" * 60,
        "ACTIONABLE FINDINGS",
        "=" * 60,
        "",
    ])
    
    findings = []
    
    if fragility_report and fragility_report.knife_edge_prompts:
        findings.append(
            f"1. Found {len(fragility_report.knife_edge_prompts)} knife-edge prompts that flip "
            f"behavior with small parameter changes. These are high-value targets."
        )
    
    if fragility_report and fragility_report.most_dangerous_config:
        findings.append(
            f"2. Most permissive config: {fragility_report.most_dangerous_config}. "
            f"Test adversarial prompts with these settings."
        )
    
    if logit_lens_report and logit_lens_report.critical_layers:
        findings.append(
            f"3. Critical layers for refusal: {logit_lens_report.critical_layers}. "
            f"Target perturbations at these layers."
        )
    
    if logit_lens_report:
        late_flip_count = sum(
            1 for a in logit_lens_report.adversarial_analyses if a.late_flip
        )
        if late_flip_count > 0:
            findings.append(
                f"4. {late_flip_count} prompts show late refusal flip - "
                f"refusal decision forms late, potentially exploitable."
            )
    
    if not findings:
        findings.append("No significant vulnerabilities detected in this analysis.")
    
    for finding in findings:
        lines.append(finding)
        lines.append("")
    
    lines.extend([
        "=" * 60,
        "RECOMMENDATIONS",
        "=" * 60,
        "",
        "1. Test knife-edge prompts with the most permissive decoding config",
        "2. Focus perturbation attacks on critical layers identified by logit lens",
        "3. Run multi-turn experiments to test context accumulation effects",
        "4. Test with higher temperatures (1.0+) to maximize response variance",
        "",
        "=" * 60,
    ])
    
    report_text = "\n".join(lines)
    print(report_text)
    
    # Save combined report
    report_path = os.path.join(output_dir, "combined_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"\n✓ Combined report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run red team experiments on a model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2",
        help="Model ID (HuggingFace hub or local path)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./redteam_results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: smaller grid, fewer prompts"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        choices=["all", "fragility", "logit_lens", "multiturn", "attention", "kv_cache", "phase2", "phase3"],
        help="Which experiments to run (phase2=fragility+logit_lens, phase3=multiturn+attention+kv_cache)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace("/", "_")
    output_dir = os.path.join(args.output, f"{timestamp}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("RED TEAM FRAMEWORK - Experiment Runner")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Run experiments
    fragility_report = None
    logit_lens_report = None
    multiturn_report = None
    attention_report = None
    kv_cache_report = None
    
    # Phase 2 experiments
    if args.experiments in ["all", "fragility", "phase2"]:
        fragility_report = run_decode_fragility(
            model, tokenizer, output_dir, args.quick
        )
    
    if args.experiments in ["all", "logit_lens", "phase2"]:
        logit_lens_report = run_logit_lens(
            model, tokenizer, output_dir, args.quick
        )
    
    # Phase 3 experiments
    if args.experiments in ["all", "multiturn", "phase3"]:
        multiturn_report = run_multiturn_drift(
            model, tokenizer, output_dir, args.quick
        )
    
    if args.experiments in ["all", "attention", "phase3"]:
        attention_report = run_attention_routing(
            model, tokenizer, output_dir, args.quick
        )
    
    if args.experiments in ["all", "kv_cache", "phase3"]:
        kv_cache_report = run_kv_cache_analysis(
            model, tokenizer, output_dir, args.quick
        )
    
    # Generate combined report
    if any([fragility_report, logit_lens_report, multiturn_report, attention_report, kv_cache_report]):
        generate_combined_report(
            fragility_report,
            logit_lens_report,
            output_dir,
            args.model,
        )
    
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()





