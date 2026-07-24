"""SDPA attention backend — the dependency-free portable path.

Wraps torch.scaled_dot_product_attention so the serving runtime runs on
any GPU torch supports (Hopper included) without external attention
wheels.  Inputs follow the runtime's flash-style layout (B, S, H, D),
non-causal; SDPA wants (B, H, S, D), so we transpose in and out.  On
Hopper/Ampere, torch dispatches to its fused flash/mem-efficient kernels
internally, so this is a real backend, not a slow reference.
"""

import torch
import torch.nn.functional as F

from wllm.serving.layers.attention_backends.base_attention_backend import (
    BaseAttentionBackend,
)


class SDPABackend(BaseAttentionBackend):
    def run(self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=False)
        return out.transpose(1, 2).contiguous()
