# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from typing import Optional, Tuple
import torch
from wllm.serving.layers.custom_op import CustomOp
from wllm.serving.runner.forward_batch import ForwardBatchInfo
import wllm.kernels_t

@CustomOp.register("camera_rope")
class CameraRope(CustomOp):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward_native(self, viewmats: torch.Tensor, Ks: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return _prepare_apply_fns_all_dim(self.head_dim, viewmats, Ks)
    def forward_cuda(self, viewmats: torch.Tensor, Ks: torch.Tensor) -> torch.Tensor:
        
        return self.forward_native(viewmats, Ks)
    

def _prepare_apply_fns_all_dim(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    return P_T.contiguous(), P_inv.contiguous(), P.contiguous()



def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    out = out.to(dtype=transforms.dtype)
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


@CustomOp.register("apply_rope_prope_qkv")
class ApplyRopePRopeQKV(CustomOp):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
    
    def forward_native(self, 
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    rotary_emb: torch.Tensor,
    P_q: torch.Tensor, 
    P_kv: torch.Tensor,
    forward_batch_info: ForwardBatchInfo) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        raise NotImplementedError
    
    def forward_cuda(self,
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    rotary_emb: torch.Tensor,
    P_q: torch.Tensor, 
    P_kv: torch.Tensor,
    forward_batch_info: ForwardBatchInfo
    ) -> torch.Tensor:
        
        return wllm.kernels_t.rope_ops.apply_rope_prope_qkv(
            query, key, value, rotary_emb,
            P_q, P_kv,
            forward_batch_info.kv_memory.get_rope_k_ptr(self.layer_id),
            forward_batch_info.kv_memory.get_rope_v_ptr(self.layer_id),
            forward_batch_info.kv_memory.get_prope_k_ptr(self.layer_id),
            forward_batch_info.kv_memory.get_prope_v_ptr(self.layer_id),
            forward_batch_info.cache_start
        )


@CustomOp.register("apply_rope_qkv")
class ApplyRopeQKV(CustomOp):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> torch.Tensor:
        return wllm.kernels_t.rope_ops.apply_rope_qkv(
            query,
            key,
            value,
            rotary_emb,
            forward_batch_info.kv_memory.get_k_ptr(self.layer_id),
            forward_batch_info.kv_memory.get_v_ptr(self.layer_id),
            forward_batch_info.cache_start,
        )


@CustomOp.register("apply_prope_o")
class ApplyPRopeO(CustomOp):
    def __init__(self, in_place: bool = True) -> None:
        super().__init__()
        self.in_place = in_place
        assert self.in_place

    def forward_native(self, 
    o_combine: torch.Tensor, 
    P_o: torch.Tensor) -> ForwardBatchInfo:
        """PyTorch-native implementation equivalent to forward()."""
        raise NotImplementedError
    
    def forward_cuda(self,
    o_combine: torch.Tensor, 
    P_o: torch.Tensor,
    ) -> torch.Tensor:
        
        assert o_combine.shape[0] == 2
        assert P_o.shape[0] == 1

        if self.in_place:
            wllm.kernels_t.rope_ops.apply_prope_o(o_combine[1], P_o)
        else:
            raise NotADirectoryError