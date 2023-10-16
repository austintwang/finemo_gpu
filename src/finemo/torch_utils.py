import torch
import warnings

def compile_if_possible(**kwargs):
    try:
        torch.compile(**kwargs)
    except Exception as e:
        warnings.warn(f"PyTorch optimization failed. Falling back to non-compiled code. Original error message:\n{e}")
        torch.compile(**dict(kwargs, disable=True))