# SPDX-License-Identifier: Apache-2.0
"""AITER flash-attention backend.

Uses AITER's dense ``flash_attn_func``; inputs and output are ``[B, S, H, D]``
(GQA/MQA native).
"""

import torch
from aiter import flash_attn_func

from wllm.serving.layers.attention_backends.base_attention_backend import BaseAttentionBackend


class AiterFlashAttentionBackend(BaseAttentionBackend):

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def run(self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs) -> torch.Tensor:
        return flash_attn_func(q, k, v, causal=self.causal)
