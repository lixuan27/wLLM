# Copyright 2025 Krea. All rights reserved.

from typing import Dict, Iterable, Union
import math

import torch
import torch.nn as nn

from wllm.serving.configs.models.dits.krea_realtime import KreaRealtimeConfig
from wllm.serving.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_shard,
)
from wllm.serving.layers.activation import get_act_fn
from wllm.serving.layers.convolution import ReplicatedConv3D
from wllm.serving.layers.layernorm import (
    FP32LayerNorm,
    FP32LayerNormScaleShift,
    ScaleResidual,
)
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.mlp import FeedForward
from wllm.serving.layers.visual_embedding import (
    PixArtAlphaTextProjection,
    Timesteps,
    TimestepEmbedder,
)
from wllm.serving.layers.wan_attention import WanRopeSelfAttention, WanT2VCrossAttention
from wllm.serving.models.dit.base import BaseDiT
from wllm.serving.models.loader.weight_utils import (
    default_weight_loader,
    map_param_name,
    maybe_remap_kv_scale_name,
)
from wllm.serving.runner.forward_batch import ForwardBatchInfo


class KreaRealtimeTimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedder(
            dim,
            frequency_embedding_size=time_freq_dim,
        )
        self.act_fn = get_act_fn("silu")
        self.time_proj = ReplicatedLinear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim,
            dim,
            act_fn="gelu_pytorch_tanh",
        )

    def forward(self, timestep: torch.Tensor):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep)

        timestep_proj, _ = self.time_proj(self.act_fn(temb))
        return temb, timestep_proj


class KreaRealtimeTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        layer_id: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.norm1 = FP32LayerNormScaleShift(
            dim, eps, elementwise_affine=False, bias=False
        )
        self.attn1 = WanRopeSelfAttention(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dim // num_heads,
            layer_id=layer_id,
            bias=True,
            out_dim=dim,
            out_bias=True,
            eps=eps,
        )
        self.scale_residual = ScaleResidual()

        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.attn2 = WanT2VCrossAttention(
            hidden_size=dim,
            cross_attention_hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dim // num_heads,
            layer_id=layer_id,
            bias=True,
            out_dim=dim,
            out_bias=True,
            eps=eps,
        )

        self.norm3 = FP32LayerNormScaleShift(
            dim, eps, elementwise_affine=False, bias=False
        )
        self.ffn = FeedForward(
            dim, inner_dim=ffn_dim, activation_fn="gelu_pytorch_tanh"
        )

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states, scale_msa, shift_msa)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            forward_batch_info=forward_batch_info,
        )
        hidden_states = self.scale_residual(hidden_states, attn_output, gate_msa)

        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            forward_batch_info=forward_batch_info,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states, c_scale_msa, c_shift_msa)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.scale_residual(hidden_states, ff_output, c_gate_msa)
        return hidden_states

    def optimize(self, **kwargs):
        self.ffn = torch.compile(
            self.ffn, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn1.to_q = torch.compile(
            self.attn1.to_q, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn1.to_k = torch.compile(
            self.attn1.to_k, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn1.to_v = torch.compile(
            self.attn1.to_v, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn1.to_out = torch.compile(
            self.attn1.to_out, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn2.to_q = torch.compile(
            self.attn2.to_q, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        self.attn2.to_out = torch.compile(
            self.attn2.to_out, mode="max-autotune-no-cudagraphs", dynamic=False
        )


class KreaRealtimeTransformer3DModel(BaseDiT):
    _default_config = KreaRealtimeConfig()

    def __init__(self, config: KreaRealtimeConfig) -> None:
        super().__init__(config=config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        out_channels = config.out_channels or config.in_channels

        self.patch_size = config.patch_size
        self.attention_head_dim = config.attention_head_dim
        self.num_attention_heads = config.num_attention_heads
        self.inner_dim = inner_dim

        self.patch_embedding = ReplicatedConv3D(
            config.in_channels,
            inner_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.condition_embedder = KreaRealtimeTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=config.text_dim,
        )

        self.blocks = nn.ModuleList(
            [
                KreaRealtimeTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    layer_id,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                )
                for layer_id in range(config.num_layers)
            ]
        )

        self.norm_out = FP32LayerNormScaleShift(
            inner_dim, config.eps, elementwise_affine=False, bias=False
        )
        self.proj_out = ReplicatedLinear(
            inner_dim, out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = sequence_model_parallel_shard(hidden_states, dim=2)
        timestep = sequence_model_parallel_shard(timestep, dim=0)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj = self.condition_embedder(timestep)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                timestep_proj,
                rotary_emb,
                forward_batch_info,
            )

        if forward_batch_info.is_cache:
            return None

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, scale, shift)
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
        hidden_states = hidden_states.reshape(
            1,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def fill_encoder_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ):
        hidden = self.condition_embedder.text_embedder(encoder_hidden_states)

        for idx, block in enumerate(self.blocks):
            key, value = block.attn2.get_encoder_kv(hidden)
            forward_batch_info.kv_memory.set_encoder_kv(idx, key, value)

    def optimize(self, **kwargs):
        for module in self.blocks:
            module.optimize(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for raw_name, loaded_weight in weights:
            name = map_param_name(raw_name, self.config.arch_config.param_names_mapping)
            if "rotary_emb.inv_freq" in name or name == "freqs":
                continue
            if "scale" in name:
                kv_scale_name = maybe_remap_kv_scale_name(name, params_dict)
                if kv_scale_name is None:
                    continue
                name = kv_scale_name

            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


EntryClass = KreaRealtimeTransformer3DModel
