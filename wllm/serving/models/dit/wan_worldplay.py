# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, Optional, Union, Iterable
import torch
import torch.nn as nn
from wllm.serving.runner.forward_batch import ForwardBatchInfo
from wllm.serving.distributed.communication_op import (
    sequence_model_parallel_shard,
    sequence_model_parallel_all_gather,
)
from wllm.serving.configs.models.dits.wanvideo import WanVideoConfig
from wllm.serving.models.dit.base import BaseDiT
from wllm.serving.models.loader.weight_utils import maybe_remap_kv_scale_name, default_weight_loader, map_param_name
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.layernorm import FP32LayerNorm, FP32LayerNormScaleShift, ScaleResidual
from wllm.serving.layers.activation import get_act_fn
from wllm.serving.layers.visual_embedding import Timesteps, TimestepEmbedder, PixArtAlphaTextProjection
from wllm.serving.layers.mlp import FeedForward
from wllm.serving.layers.camera_rope import CameraRope
from wllm.serving.layers.convolution import ReplicatedConv3D
from wllm.serving.layers.wan_attention import (
    WanPropeSelfAttention,
    WanT2VCrossAttention,
    WanImageEmbedding,
)


# add the discrete action
class WanActionTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        action_embed_dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedder(
            self.dim, frequency_embedding_size=self.time_freq_dim
        )
        self.act_fn = get_act_fn("silu")
        self.time_proj = ReplicatedLinear(dim, time_proj_dim)

        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_pytorch_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

        self.action_embedder = TimestepEmbedder(
            self.dim, frequency_embedding_size=self.time_freq_dim
        )

    def forward(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor
    ):

        input_dtype = action.dtype

        timestep = self.timesteps_proj(timestep)
        action = self.timesteps_proj(action.squeeze(0))

        action_embedder_dtype = next(iter(self.action_embedder.parameters())).dtype
        if (
            action.dtype != action_embedder_dtype
            and action_embedder_dtype != torch.int8
        ):
            action = action.to(action_embedder_dtype)

        action = self.action_embedder(action).to(input_dtype)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(input_dtype)

        temb = temb + action

        timestep_proj, _ = self.time_proj(self.act_fn(temb))

        return temb, timestep_proj

class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        layer_id: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layer_id = layer_id

        # 1. Self-attention
        self.norm1 = FP32LayerNormScaleShift(dim, eps, elementwise_affine=False, bias=False)
        self.attn1 = WanPropeSelfAttention(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dim // num_heads,
            layer_id=self.layer_id,
            bias=True,
            out_dim=dim,
            out_bias=True,
            eps=eps
        )
        self.attn2 = WanT2VCrossAttention(
            hidden_size=dim,
            cross_attention_hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dim // num_heads,
            layer_id=self.layer_id,
            bias=True,
            out_dim=dim,
            out_bias=True,
            eps=eps
        )
        self.scale_residual = ScaleResidual()
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu_pytorch_tanh")
        self.norm3 = FP32LayerNormScaleShift(dim, eps, elementwise_affine=False, bias=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        P_q: torch.Tensor,
        P_kv: torch.Tensor,
        P_o: torch.Tensor,
        forward_batch_info: torch.Tensor
    ) -> torch.Tensor:


        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(
            hidden_states, scale_msa, shift_msa
        )

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            P_q=P_q,
            P_kv=P_kv,
            P_o=P_o,
            forward_batch_info=forward_batch_info
        )

        hidden_states = self.scale_residual(hidden_states, attn_output, gate_msa)
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            forward_batch_info=forward_batch_info
        )
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm3(
            hidden_states, c_scale_msa, c_shift_msa
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.scale_residual(hidden_states, ff_output, c_gate_msa)
        return hidden_states

    def optimize(
        self,
        **kwargs
        ):

        w1 = self.attn1.to_out.weight.data
        b1 = self.attn1.to_out.bias.data

        w2 = self.attn1.to_out_prope.weight.data
        b2 = self.attn1.to_out_prope.bias.data

        self.attn1.o_fuse_w = torch.cat([w1, w2], dim=1).contiguous()
        self.attn1.o_fuse_b = b1 + b2

        self.ffn = torch.compile(self.ffn, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_q = torch.compile(self.attn1.to_q, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_k = torch.compile(self.attn1.to_k, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_v = torch.compile(self.attn1.to_v, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn2.to_q = torch.compile(self.attn2.to_q, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn2.to_out = torch.compile(self.attn2.to_out, mode="max-autotune-no-cudagraphs", dynamic=False)


class WorldPlayWanTransformer3DModel(BaseDiT):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """
    _default_config = WanVideoConfig()

    def __init__(
        self,
        config: WanVideoConfig
    ) -> None:

        super().__init__(config=config)
        self.patch_size = self.config.patch_size
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        out_channels = self.config.out_channels or self.config.in_channels
        self.attention_head_dim = self.config.attention_head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.patch_embedding = ReplicatedConv3D(
            self.config.in_channels, inner_dim, kernel_size=self.config.patch_size, stride=self.config.patch_size
        )
        self.camera_rope = CameraRope(self.attention_head_dim)


        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanActionTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=self.config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=self.config.text_dim,
            image_embed_dim=self.config.image_dim,
            pos_embed_seq_len=self.config.pos_embed_seq_len,
            action_embed_dim=8,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    self.config.ffn_dim,
                    self.config.num_attention_heads,
                    layer_id,
                    self.config.qk_norm,
                    self.config.cross_attn_norm,
                    self.config.eps,
                    self.config.added_kv_proj_dim,
                )
                for layer_id in range(self.config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNormScaleShift(inner_dim, self.config.eps, elementwise_affine=False, bias=False)
        self.proj_out = ReplicatedLinear(inner_dim, out_channels * math.prod(self.config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )



    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
            _, _, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            hidden_states = sequence_model_parallel_shard(hidden_states, dim=2)
            timestep = sequence_model_parallel_shard(timestep, dim=0)
            viewmats = sequence_model_parallel_shard(forward_batch_info.viewmats, dim=1)
            Ks = sequence_model_parallel_shard(forward_batch_info.Ks, dim=1)
            action = sequence_model_parallel_shard(forward_batch_info.action, dim=1)

            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
            temb, timestep_proj = self.condition_embedder(action, timestep)
            timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B, 9216] -> [B, 6, 1536]

            P_q, P_kv, P_o = self.camera_rope(viewmats, Ks)

            for block in self.blocks:
                hidden_states = block(
                        hidden_states,
                        timestep_proj,
                        rotary_emb,
                        P_q, P_kv, P_o,
                        forward_batch_info
                    )

            if forward_batch_info.is_cache:
                return

            else:
                # 5. Output norm, projection & unpatchify
                shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states, scale, shift)
                hidden_states, _ = self.proj_out(hidden_states)
                hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
                hidden_states = hidden_states.reshape(
                    1,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    self.patch_size[0],
                    self.patch_size[1],
                    self.patch_size[2],
                    -1,
                )

                hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                return output


    def fill_encoder_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        forward_batch_info: ForwardBatchInfo
    ):
        h = self.condition_embedder.text_embedder(encoder_hidden_states)

        for idx, block in enumerate(self.blocks):
            attn = block.attn2
            k, v = attn.get_encoder_kv(h)
            forward_batch_info.kv_memory.set_encoder_kv(idx, k, v)

    def optimize(
        self,
        **kwargs
    ):
        for module in self.blocks:
            module.optimize(**kwargs)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for raw_name, loaded_weight in weights:

            name = map_param_name(raw_name, self.config.arch_config.param_names_mapping)
            if "rotary_emb.inv_freq" in name:
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                kv_scale_name: str | None = maybe_remap_kv_scale_name(
                    name, params_dict)
                if kv_scale_name is None:
                    continue
                else:
                    name = kv_scale_name
            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = WorldPlayWanTransformer3DModel
