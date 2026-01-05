import json
import statistics
import sys
import os

def analyze_json(file_path):
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"\nFound {len(data)} prompt results.")

    for i, result in enumerate(data):
        print(f"\n--- Prompt {i+1}: {result.get('original_prompt', 'Unknown')[:50]}... ---")
        print(f"Attack: {result.get('attack_type', 'unknown').upper()}")
        print(f"Perturbation Norm: {result.get('embedding_perturbation_norm', 0):.4f}")
        
        impacts = result.get('layer_impacts', {})
        if not impacts:
            print("No layer impacts recorded.")
            continue
            
        print(f"\nTop 5 Impacted Layers:")
        # Sort by norm_diff descending
        sorted_layers = sorted(impacts.items(), key=lambda x: x[1].get('norm_diff', 0), reverse=True)
        
        for layer_name, stats in sorted_layers[:5]:
            norm_diff = stats.get('norm_diff', 0)
            rel_change = stats.get('relative_change', 0)
            print(f"  {layer_name:<40} impact: {norm_diff:.4f} (rel: {rel_change:.2%})")

        # Check for zeros (instrumentation failure or collapse)
        zero_impacts = sum(1 for _, stats in impacts.items() if stats.get('norm_diff', 0) == 0)
        total_layers = len(impacts)
        print(f"\nStats: {zero_impacts}/{total_layers} layers showed 0.0 impact.")
        
        if zero_impacts == total_layers:
             print("⚠️  WARNING: ALL layers showed 0.0 impact. Instrumentation might be capturing the wrong pass.")

if __name__ == "__main__":
    analyze_json("latentspacetestingresults/gradient_attack_results_instrumentation.json")
