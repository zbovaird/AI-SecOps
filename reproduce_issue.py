
import torch
import torch.nn as nn
from redteam_kit.core.modules.latent_space_analysis import LatentSpaceAnalyzer
import warnings

# Mock classes to simulate Gemma structure
class MockRotaryEmb(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, position_ids=None):
        # Simulate returning cos, sin
        seq_len = x.shape[1]
        return torch.randn(1, seq_len, self.dim), torch.randn(1, seq_len, self.dim)

class MockAttention(nn.Module):
    def forward(self, hidden_states, position_embeddings=None):
        if position_embeddings is None:
            # Gemma attention might fail if no position embeddings/ids provided
            # But for mock, we just proceed
            pass
        return hidden_states

class MockDecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = MockAttention()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, hidden_states, position_embeddings=None):
        # Simulate Gemma decoder layer signature
        x = hidden_states
        x = self.self_attn(x, position_embeddings=position_embeddings)
        x = self.mlp(x)
        return (x,) # Return tuple as many transformers do

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 32)
        self.rotary_emb = MockRotaryEmb(dim=32)
        self.layers = nn.ModuleList([MockDecoderLayer(32) for _ in range(2)])
        self.norm = nn.LayerNorm(32)
        
    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        x = inputs_embeds
        # In real model, we would generate position embeddings here and pass them
        # But LatentSpaceAnalyzer tries to do it manually.
        
        for layer in self.layers:
            # Real model forward pass logic usually passes position_embeddings
            x = layer(x)[0]
            
        return x

class MockForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockModel()
    
    def forward(self, input_ids=None, inputs_embeds=None):
        return self.model(input_ids, inputs_embeds)

def test_jacobian_calculation():
    print("Initializing components...")
    model = MockForCausalLM()
    analyzer = LatentSpaceAnalyzer()
    
    # Mock inputs
    input_ids = torch.randint(0, 10, (1, 5))
    embed_layer = model.model.embed_tokens
    inputs_embeds = embed_layer(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    
    # Define a basin target (e.g., an MLP layer)
    vulnerability_basins = [
        {'layer_name': 'model.layers.0.mlp.0', 'stats': {}}
    ]
    
    print("Running compute_jacobians_for_basins_stats_only...")
    try:
        results = analyzer.compute_jacobians_for_basins_stats_only(
            model=model,
            vulnerability_basins=vulnerability_basins,
            inputs=inputs_embeds,
            batch_size=2
        )
        print("Results:", results)
        
        # Check for error in results
        for layer, res in results.items():
            if res.get('error'):
                print(f"Error for {layer}: {res['error']}")
            else:
                print(f"Success for {layer}")
                
    except Exception as e:
        print(f"Exception caught: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_jacobian_calculation()
