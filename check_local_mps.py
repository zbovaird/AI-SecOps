import torch

def check_mps():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return False
    
    try:
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(f"Success: MPS available and working. Device type: {x.device.type}")
        return True
    except Exception as e:
        print(f"MPS available but error creating tensor: {e}")
        return False

if __name__ == "__main__":
    check_mps()
