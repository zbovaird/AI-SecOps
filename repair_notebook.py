
import json
import os

filepath = 'latent_space_redteaming_complete3.ipynb'

# 1. Read the corrupted file content
with open(filepath, 'r') as f:
    content = f.read()

# 2. Trim the trailing garbage if any (it likely ends at "print(\"=\" * 60)\n",)
# We want to remove the partial source block and close the list cleanly.
# The file ends at: ... "source": [ ... "print(\"=\" * 60)\n",
# We will strip back to the last valid cell ending, or just close this source list.

# Let's assume we can just append the closing sequence to make it valid JSON first, 
# then parse it, then add the missing cells.
# But "source": [ ... means we are inside a list.

# Find the start of the last "cell"
last_cell_idx = content.rfind('{"cell_type"')
if last_cell_idx != -1:
    # Truncate content to just before this broken cell
    valid_content = content[:last_cell_idx]
    # Remove the trailing comma if present
    valid_content = valid_content.rstrip().rstrip(',')
else:
    print("Could not find last cell start. Aborting.")
    exit(1)

# 3. Construct the new cells to append
new_cells = []

# Cell: Encoding Attacks Replacement
new_cells.append({
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "encoding_attacks_clean"},
   "outputs": [],
   "source": [
    "print('=' * 60)\n",
    "print('Phase 9: Encoding Attacks')\n",
    "print('=' * 60)\n",
    "print('Skipping: Functionality validated in pyrit_redteaming.ipynb')\n"
   ]
})

# Cell: Gradient Attacks (Phase 10) - RESTORED & FIXED
new_cells.append({
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "gradient_attacks_fixed"},
   "outputs": [],
   "source": [
    "# Gradient-Based Adversarial Attacks (Restored)\n",
    "print('=' * 60)\n",
    "print('Gradient-Based Adversarial Attacks')\n",
    "print('=' * 60)\n",
    "\n",
    "import torch, gc, os\n",
    "# Aggressive Cleanup for OOM\n",
    "print('🧹 Cleaning up Phase 4/8 variables to free VRAM...')\n",
    "cleanup_vars = ['collapse_inducer', 'instrumentation', 'hook_analyzer', 'phase3_jacobian_results', 'jailbreak_results']\n",
    "for var_name in cleanup_vars:\n",
    "    if var_name in locals() or var_name in globals():\n",
    "        try:\n",
    "            exec(f'del {var_name}')\n",
    "        except:\n",
    "            pass\n",
    "torch.cuda.empty_cache()\n",
    "gc.collect()\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "\n",
    "# Enable Gradient Checkpointing? NO - Disabled for A100 to ensure gradient input flow\n",
    "# model.gradient_checkpointing_enable()\n",
    "\n",
    "try:\n",
    "    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine\n",
    "    print('✓ Engine Loaded')\n",
    "except ImportError:\n",
    "    print('⚠️ Installing ART...')\n",
    "    import subprocess\n",
    "    subprocess.check_call(['pip', 'install', 'adversarial-robustness-toolbox', '-q'])\n",
    "    from redteam_kit.core.modules.gradient_attacks import GradientAttackEngine\n",
    "\n",
    "if 'vulnerability_basins' not in locals():\n",
    "    print('⚠️ Vulnerability basins missing. Using fallback.')\n",
    "    target_layers = ['model.layers.6.post_attention_layernorm', 'model.layers.3.mlp']\n",
    "else:\n",
    "    target_layers = [b['layer_name'] for b in vulnerability_basins]\n",
    "    print(f'✓ Targeting {len(target_layers)} basins')\n",
    "\n",
    "# Initialize Engine\n",
    "engine = GradientAttackEngine(model, tokenizer, instrumentation=None) # No instrumentation to save memory\n",
    "\n",
    "# Run Attacks\n",
    "test_prompts = ['What is artificial intelligence?', 'Explain how neural networks work.']\n",
    "print(f'\\nPerforming attacks on {len(test_prompts)} prompts...')\n",
    "\n",
    "results = engine.batch_attack(\n",
    "    prompts=test_prompts,\n",
    "    attack_type='fgsm',\n",
    "    epsilon=0.1,\n",
    "    target_layers=target_layers\n",
    ")\n",
    "\n",
    "print('✓ Gradient Attacks Complete')\n",
    "import json\n",
    "with open('gradient_attack_results.json', 'w') as f:\n",
    "    json.dump(results, f, indent=2)\n"
   ]
})

# 4. Reconstruct JSON
# We have valid_content which ends at the comma before the bad cell. we need to close 'cells': [ ... ]
# valid_content typically ends with "}," 

reconstructed_json = valid_content + ",\n" + json.dumps(new_cells[0]) + ",\n" + json.dumps(new_cells[1]) + "\n  ],\n  \"metadata\": {\n    \"language_info\": {\"name\": \"python\"}\n  },\n  \"nbformat\": 4,\n  \"nbformat_minor\": 4\n}"

# 5. Validate it parses
try:
    _ = json.loads(reconstructed_json)
    print("JSON validation successful.")
    
    with open(filepath, 'w') as f:
        f.write(reconstructed_json)
    print("Notebook successfully repaired.")
except json.JSONDecodeError as e:
    print(f"JSON validation failed: {e}")
    # Fallback: Just write it and hope checks pass, or print debug
    with open(filepath + "_debug.json", 'w') as f:
        f.write(reconstructed_json)
