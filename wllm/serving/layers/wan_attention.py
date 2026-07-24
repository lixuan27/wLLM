# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from wllm.serving.runner.forward_batch import ForwardBatchInfo
from wllm.serving.distributed.communication_op import (
    sequence_model_parallel_all_to_all_4D,
)
from wllm.serving.distributed.parallel_state import get_sp_world_size
from wllm.serving.distributed.utils import padded_head_count_for_sp
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.layernorm import FP32LayerNorm, RMSNorm
from wllm.serving.layers.mlp import FeedForward
from wllm.serving.layers.camera_rope import ApplyRopePRopeQKV, ApplyRopeQKV, ApplyPRopeO


def _pad_head_dim(t: torch.Tensor, old_h: int, new_h: int) -> torch.Tensor:
    """Pad dim 2 of [B, S, H, D] from H=old_h to H=new_h with zeros."""
    if new_h == old_h:
        return t
    pad_shape = (t.shape[0], t.shape[1], new_h - old_h, t.shape[3])
    pad = t.new_zeros(pad_shape)
    return torch.cat([t, pad], dim=2)


class WanPropeSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_id: int,
        scale_qk: float = None,
        bias: bool = False,
        out_dim: int = None,
        out_bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale_qk = scale_qk if (scale_qk is not None) else (self.head_dim**-0.5)
        self.bias = bias
        self.out_dim = out_dim if (out_dim is not None) else hidden_size
        self.out_bias = out_bias
        self.norm_q = RMSNorm(self.head_dim * self.num_attention_heads, eps=eps)
        self.norm_k = RMSNorm(self.head_dim * self.num_key_value_heads, eps=eps)
        self.to_q = ReplicatedLinear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=bias)
        self.to_k = ReplicatedLinear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=bias)
        self.to_v = ReplicatedLinear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=bias)
        self.to_out = ReplicatedLinear(self.head_dim * self.num_attention_heads, self.out_dim, bias=out_bias)
        self.to_out_prope = ReplicatedLinear(self.head_dim * self.num_attention_heads, self.out_dim, bias=out_bias)
        self.apply_rope_prope_qkv = ApplyRopePRopeQKV(self.layer_id)
        self.apply_prope_o = ApplyPRopeO(in_place=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor = None,
        P_q: torch.Tensor = None,
        P_kv: torch.Tensor = None,
        P_o: torch.Tensor = None,
        forward_batch_info: ForwardBatchInfo = None,
    ):
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.unflatten(2, (-1, self.head_dim))
        key = key.unflatten(2, (-1, self.head_dim))
        value = value.unflatten(2, (-1, self.head_dim))

        qkv = torch.cat([query, key, value], dim=0)
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
        query, key, value = qkv.chunk(3, dim=0)

        query_all = self.apply_rope_prope_qkv(query, key, value, rotary_emb, P_q, P_kv, forward_batch_info)

        key_all, value_all = forward_batch_info.kv_memory.get_kv(self.layer_id, forward_batch_info.cache_end)
        hidden_states_all = forward_batch_info.attention_backend.run(query_all, key_all, value_all)
        hidden_states_all = sequence_model_parallel_all_to_all_4D(hidden_states_all, scatter_dim=1, gather_dim=2)
        self.apply_prope_o(hidden_states_all, P_o)
        hidden_states = self.merge_states(hidden_states_all)
        return hidden_states

    @torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    def merge_states(self, hidden_states_all: torch.Tensor):
        hidden_states_all = hidden_states_all.transpose(0, 1).reshape(1, hidden_states_all.shape[1], -1)
        hidden_states = F.linear(hidden_states_all, self.o_fuse_w, self.o_fuse_b)
        return hidden_states


class WanRopeSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_id: int,
        scale_qk: float = None,
        bias: bool = False,
        out_dim: int = None,
        out_bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale_qk = scale_qk if (scale_qk is not None) else (self.head_dim**-0.5)
        self.bias = bias
        self.out_dim = out_dim if (out_dim is not None) else hidden_size
        self.out_bias = out_bias

        self.norm_q = RMSNorm(self.head_dim * self.num_attention_heads, eps=eps)
        self.norm_k = RMSNorm(self.head_dim * self.num_key_value_heads, eps=eps)
        self.to_q = ReplicatedLinear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=bias,
        )
        self.to_k = ReplicatedLinear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=bias,
        )
        self.to_v = ReplicatedLinear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=bias,
        )
        self.to_out = ReplicatedLinear(
            self.head_dim * self.num_attention_heads,
            self.out_dim,
            bias=out_bias,
        )
        self.apply_rope_qkv = ApplyRopeQKV(self.layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor = None,
        forward_batch_info: ForwardBatchInfo = None,
    ):
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)
        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.unflatten(2, (-1, self.head_dim))
        key = key.unflatten(2, (-1, self.head_dim))
        value = value.unflatten(2, (-1, self.head_dim))

        # When the global head count doesn't divide the SP world size, pad
        # the head dim with zeros so the existing equal-shard all-to-all and
        # the BH=2 rope kernel both stay valid. The padded heads contribute
        # nothing to the real heads' attention output (their K, V are zero,
        # so they only show up in the slots we slice off below).
        sp_world_size = get_sp_world_size()
        H_total = query.shape[2]
        H_padded = padded_head_count_for_sp(H_total, sp_world_size)
        if H_padded != H_total:
            query = _pad_head_dim(query, H_total, H_padded)
            key = _pad_head_dim(key, H_total, H_padded)
            value = _pad_head_dim(value, H_total, H_padded)

        qkv = torch.cat([query, key, value], dim=0)
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
        query, key, value = qkv.chunk(3, dim=0)

        query_all = self.apply_rope_qkv(query, key, value, rotary_emb, forward_batch_info)

        key_all, value_all = forward_batch_info.kv_memory.get_kv(self.layer_id, forward_batch_info.cache_end)

        hidden_states = forward_batch_info.attention_backend.run(query_all, key_all, value_all)
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)

        if H_padded != H_total:
            hidden_states = hidden_states[:, :, :H_total, :]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states


class WanT2VCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_id: int,
        scale_qk: float = None,
        bias: bool = False,
        out_dim: int = None,
        out_bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale_qk = scale_qk if (scale_qk is not None) else (self.head_dim**-0.5)
        self.bias = bias
        self.out_dim = out_dim if (out_dim is not None) else hidden_size
        self.out_bias = out_bias
        self.norm_q = RMSNorm(self.head_dim * self.num_attention_heads, eps=eps)
        self.norm_k = RMSNorm(self.head_dim * self.num_key_value_heads, eps=eps)
        self.to_q = ReplicatedLinear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=bias)
        self.to_k = ReplicatedLinear(self.cross_attention_hidden_size, self.head_dim * self.num_key_value_heads, bias=bias)
        self.to_v = ReplicatedLinear(self.cross_attention_hidden_size, self.head_dim * self.num_key_value_heads, bias=bias)
        self.to_out = ReplicatedLinear(self.head_dim * self.num_attention_heads, self.out_dim, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch_info: ForwardBatchInfo = None,
    ) -> torch.Tensor:
        query, _ = self.to_q(hidden_states)
        query = self.norm_q(query)
        query = query.unflatten(2, (-1, self.head_dim))
        key, value = forward_batch_info.kv_memory.get_encoder_kv(self.layer_id)
        hidden_states = forward_batch_info.attention_backend.run(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states, _ = self.to_out(hidden_states)
        return hidden_states

    def get_encoder_kv(self, encoder_hidden_states: torch.Tensor):
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)
        key = self.norm_k(key)
        key = key.unflatten(2, (-1, self.head_dim))
        value = value.unflatten(2, (-1, self.head_dim))
        return key, value


class WanImageEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()
        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states
