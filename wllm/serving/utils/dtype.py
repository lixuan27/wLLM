import torch

def parse_dtype_getattr(s: str) -> torch.dtype:
    s = s.strip().lower()
    dtype = getattr(torch, s, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype string: {s!r}")
    return dtype
    