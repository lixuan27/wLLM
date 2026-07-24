import torch
from wllm.serving.layers.attention_backends.base_attention_backend import BaseAttentionBackend
try:
  from flash_attn.cute.interface import flash_attn_func
except:
  flash_attn_func = None

class FA4Backend(BaseAttentionBackend):
    def __init__(self):
       super().__init__()
       if flash_attn_func is None:
          raise

    def run(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs):
            # some versions return (out, lse), others a bare tensor
            out = flash_attn_func(q, k, v, causal=False)
            return out[0] if isinstance(out, tuple) else out
