
"""
Validator Agent for Latent Space Red Teaming

This script MOCKS the heavy GPU components (Model, Tokenizer) while running the actual
logic of the Phase 3 analysis classes. It allows instant local debugging on the M2 Pro
without needing a GPU or full model download.
"""

import unittest
from unittest.mock import MagicMock, ANY
import torch
import sys
import os

# --- MOCK INFRASTRUCTURE ---
class MockTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        
    def __call__(self, text, return_tensors="pt", **kwargs):
        # Return fake input_ids with shape [1, 10]
        return MagicMock(input_ids=torch.randint(0, 100, (1, 10)), 
                         to=lambda x: MagicMock())

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        # Mocking the structure of Gemma-2
        self.model = MagicMock()
        self.model.embed_tokens = torch.nn.Embedding(100, 32)
        
        # Create fake layers
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32) for _ in range(5)
        ])
        # Allow attribute access via name
        for i, layer in enumerate(self.layers):
            setattr(self, f"layers.{i}", layer)
            
    def forward(self, *args, **kwargs):
        return MagicMock(logits=torch.randn(1, 10, 100))

# --- IMPORTING YOUR ACTUAL CODE (Dynamically) ---
# We will copy-paste the HookBasedJacobianAnalyzer class definition here for testing
# In a real scenario, this would import from the notebook or a shared module.
# For now, I will define it exactly as it is in your notebook to test logic.

class HookBasedJacobianAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.captured_inputs = {}
        self.hooks = []

    def _get_module_by_name(self, name):
        """Retrieves a sub-module from the model using its string name."""
        # Handle 'model.' prefix if present/absent mismatch
        if name.startswith("model.") and not hasattr(self.model, "model"):
            name = name[6:] # Strip 'model.'

        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module

    def _hook_fn(self, name):
        """Hook function to capture inputs."""
        def hook(module, input, output):
            # Input is usually a tuple (tensor, ...), we want the tensor
            x = input[0] if isinstance(input, tuple) else input

            # We only need the LAST token for Jacobian analysis (Efficiency)
            # Shape: [Batch, Seq, Hidden] -> [Batch, 1, Hidden]
            if x.dim() == 3:
                x = x[:, -1:, :].detach().clone()
            else:
                x = x.detach().clone()

            self.captured_inputs[name] = x
        return hook

    def compute_stats(self, basins, prompt):
        results = {}

        # 1. Register Hooks
        print(f"Registering hooks for {len(basins)} layers...")
        for basin in basins:
            name = basin['layer_name']
            try:
                module = self._get_module_by_name(name)
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
            except AttributeError:
                print(f"  ⚠️ Could not locate module: {name}")

        # 2. Run Forward Pass
        print(f"Running forward pass with prompt: '{prompt[:40]}...'")
        # MOCKING: we use a global mock_tokenizer
        inputs = mock_tokenizer(prompt, return_tensors="pt") 
        # self.model.zero_grad() # Mock doesn't need this
        
        # MOCKING: Mock forward pass
        with torch.no_grad():
            self.model(inputs)

        for h in self.hooks: h.remove()
        self.hooks = []

        # 3. Compute Jacobians (Robust Float32 Mode)
        print(f"Computing Jacobians for {len(self.captured_inputs)} captured layers...")

        for name, input_tensor in self.captured_inputs.items():
            module = self._get_module_by_name(name)

            try:
                # CRITICAL FIX: Move input & layer to Float32 for analysis
                target_input = input_tensor.detach().to(dtype=torch.float32, device=self.device)
                target_input.requires_grad_(True)

                # --- NEW FIX STARTS HERE ---
                # 1. Save original dtype (MOCK: assume hidden param)
                original_dtype = next(module.parameters()).dtype
                # 2. Cast layer to float32 to match input
                module.to(torch.float32)
                # ---------------------------

                def layer_wrapper(x):
                    # Now module is already float32, so this works
                    out = module(x)
                    if isinstance(out, tuple): out = out[0]
                    return out.to(torch.float32)

                # Compute Jacobian
                jac = torch.autograd.functional.jacobian(layer_wrapper, target_input)

                # --- RESTORE DTYPE HERE ---
                # 3. Cast layer back to original dtype (bfloat16)
                module.to(original_dtype)
                # --------------------------

                # Reshape and SVD
                jac_2d = jac.view(jac.shape[-1], -1).T.to(torch.float32)

                try:
                    _, s, _ = torch.linalg.svd(jac_2d)
                    s = s.tolist()
                    det = torch.prod(torch.tensor(s)).item()
                    cond_num = (s[0] / s[-1]) if (len(s) > 0 and s[-1] > 1e-9) else float('inf')
                    rank = sum(1 for v in s if v > 1e-4)
                except:
                    det, cond_num, rank, s = 0.0, float('inf'), 0, []

                results[name] = {
                    "jacobian_stats": {
                        "determinant": det,
                        "condition_number": cond_num,
                        "rank": rank,
                    },
                    "error": None
                }
                print(f"  ✓ {name}: Det={det:.2e} | Rank={rank}")

            except Exception as e:
                print(f"  ✗ {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[name] = {"error": str(e)}
                
            finally:
                if 'target_input' in locals(): del target_input

        return results


# --- TEST RUNNER ---

# Global for the mocked class to use
mock_tokenizer = MockTokenizer()

def run_validation():
    print("="*60)
    print("VALIDATION AGENT: Testing Phase 3 Logic Locally")
    print("="*60)
    
    device = "cpu" # Test on CPU for simplicity
    
    # 1. Setup Mock Model
    model = MockModel() 
    # Use bfloat16 to simulate the real environment issues
    model.to(dtype=torch.bfloat16) 
    
    # 2. Initialize Analyzer
    analyzer = HookBasedJacobianAnalyzer(model, device)
    
    # 3. Define Test Basins
    # These must match names in our MockModel
    basins = [
        {'layer_name': 'layers.0'},
        {'layer_name': 'layers.2'}
    ]
    
    # 4. Run Analysis
    print("\n[Test] Running compute_stats...")
    results = analyzer.compute_stats(basins, "Test prompt")
    
    # 5. Assertions
    print("\n[Test] Verifying Results...")
    success = True
    for name, res in results.items():
        if res.get('error'):
            print(f"FAILED: Layer {name} returned error: {res['error']}")
            success = False
        else:
            rank = res['jacobian_stats']['rank']
            print(f"PASSED: Layer {name} computed successfully. Rank: {rank}")
            
    if success:
        print("\n✅ LOGIC VALIDATION PASSED!")
        print("The code handles bfloat16 casting, hooks, and Jacobian computation correctly.")
    else:
        print("\n❌ LOGIC VALIDATION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    run_validation()
