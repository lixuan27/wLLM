import torch
from wllm.serving.layers.attention_backends.base_attention_backend import BaseAttentionBackend
try:
  import flashinfer
except:
    flashinfer = None


class FlashInferBackend(BaseAttentionBackend):

    def __init__(self):
        super().__init__()
        if flashinfer is None:
            raise
    
    def run(self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        **kwargs):

        b, sq, _, d = q.shape
        b, skv,_,d = k.shape
        q = q.transpose(0, 1).reshape(sq, -1, d)
        k = k.transpose(0, 1).reshape(skv, -1, d)
        v = v.transpose(0, 1).reshape(skv, -1, d)
        o = flashinfer.single_prefill_with_kv_cache(
            q, k, v, kv_layout="NHD", use_fp16_qk_reduction=True,
            backend="auto"
        )
        o = o.reshape(sq, b, -1, d).transpose(0, 1).contiguous()
        return o



