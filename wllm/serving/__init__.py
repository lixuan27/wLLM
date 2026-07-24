__version__ = "0.1.0"

import os
import torch

# On ROCm, HuggingFace's flash_attention_2 path runs on flash-attention's Triton
# AMD backend; select it before anything imports flash_attn.
if getattr(torch.version, "hip", None) is not None:
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
