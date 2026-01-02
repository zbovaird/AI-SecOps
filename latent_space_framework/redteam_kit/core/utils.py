"""
Utility functions for redteam_kit
Handles common operations like type conversion, serialization, etc.
"""

import torch
import numpy as np
from typing import Any, Union, Dict, List


def ensure_numpy_compatible(tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert tensor to float32 if it's bfloat16 or other unsupported types for NumPy.
    
    Args:
        tensor: Input tensor or array
        
    Returns:
        Tensor compatible with NumPy operations
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.float()
        # Also handle other potentially problematic types
        if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
            try:
                # Try to convert to float32
                return tensor.float()
            except:
                pass
    return tensor


def convert_to_native(obj: Any) -> Any:
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        # Convert tensor to list
        if obj.dtype == torch.bfloat16:
            obj = obj.float()
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


def normalize_activation_shape(activation: torch.Tensor, method: str = 'last_token') -> torch.Tensor:
    """
    Normalize activation to consistent shape for comparison.
    
    Args:
        activation: Input activation tensor
        method: Normalization method ('last_token', 'mean', 'first_token')
        
    Returns:
        Normalized activation tensor
    """
    if len(activation.shape) == 3:  # (batch, seq_len, hidden)
        if method == 'last_token':
            return activation[:, -1, :]
        elif method == 'mean':
            return activation.mean(dim=1)
        elif method == 'first_token':
            return activation[:, 0, :]
    elif len(activation.shape) == 2:  # (batch, hidden)
        if activation.shape[0] > 1:
            return activation[-1] if method == 'last_token' else activation[0]
        else:
            return activation[0]
    return activation


def safe_detach_clone(tensor: Any) -> Union[torch.Tensor, None]:
    """
    Safely detach and clone a tensor, handling None and non-tensor inputs.
    
    Args:
        tensor: Input that may be a tensor, None, or tuple
        
    Returns:
        Detached and cloned tensor, or None if input was None
    """
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        if len(tensor) > 0:
            tensor = tensor[0]
        else:
            return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().clone()
    return tensor
