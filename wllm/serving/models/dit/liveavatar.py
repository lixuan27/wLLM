# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import Dict, Iterable, Union

import torch
import torch.nn as nn

from wllm.serving.runner.forward_batch import ForwardBatchInfo
from wllm.serving.distributed.communication_op import (
    sequence_model_parallel_shard,
    sequence_model_parallel_all_gather,
)
from wllm.serving.configs.models.dits.liveavatar import LiveAvatarConfig
from wllm.serving.models.dit.base import BaseDiT
from wllm.serving.models.loader.weight_utils import (
    maybe_remap_kv_scale_name,
    default_weight_loader,
    map_param_name,
)
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.layernorm import (
    FP32LayerNorm,
    FP32LayerNormScaleShift,
    ScaleResidual,
)
from wllm.serving.layers.activation import get_act_fn
from wllm.serving.layers.visual_embedding import (
    Timesteps,
    TimestepEmbedder,
    PixArtAlphaTextProjection,
)
from wllm.serving.layers.mlp import FeedForward
from wllm.serving.layers.convolution import ReplicatedConv3D
from wllm.serving.layers.wan_attention import (
    WanRopeSelfAttention,
    WanT2VCrossAttention,
)
from wllm.serving.layers.audio import (
    CausalAudioEncoder,
    AudioInjector,
)

class LiveAvatarTimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        zero_timestep: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.time_freq_dim = time_freq_dim
        self.zero_timestep = zero_timestep

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

    def forward(self, timestep: torch.Tensor):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep)

        timestep_proj, _ = self.time_proj(self.act_fn(temb))
        return temb, timestep_proj

class LiveAvatarTransformerBlock(nn.Module):
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

        self.norm1 = FP32LayerNormScaleShift(dim, eps, elementwise_affine=False, bias=False)
        self.attn1 = WanRopeSelfAttention(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=dim // num_heads,
            layer_id=self.layer_id,
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
            layer_id=self.layer_id,
            bias=True,
            out_dim=dim,
            out_bias=True,
            eps=eps,
        )

        self.norm3 = FP32LayerNormScaleShift(dim, eps, elementwise_affine=False, bias=False)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu_pytorch_tanh")

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
        self.ffn = torch.compile(self.ffn, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_q = torch.compile(self.attn1.to_q, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_k = torch.compile(self.attn1.to_k, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_v = torch.compile(self.attn1.to_v, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn1.to_out = torch.compile(self.attn1.to_out, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn2.to_q = torch.compile(self.attn2.to_q, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.attn2.to_out = torch.compile(self.attn2.to_out, mode="max-autotune-no-cudagraphs", dynamic=False)


class LiveAvatarTransformer3DModel(BaseDiT):
    _default_config = LiveAvatarConfig()

    def __init__(self, config: LiveAvatarConfig) -> None:
        super().__init__(config=config)
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        out_channels = self.config.out_channels or self.config.in_channels
        self.attention_head_dim = self.config.attention_head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.inner_dim = inner_dim
        self.patch_size = self.config.patch_size

        self.patch_embedding = ReplicatedConv3D(
            self.config.in_channels,
            inner_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )

        if self.config.arch_config.cond_dim > 0:
            self.cond_encoder = ReplicatedConv3D(
                self.config.arch_config.cond_dim,
                inner_dim,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            )

        if self.config.arch_config.enable_framepack:
            self.frame_pack_proj = ReplicatedConv3D(
                self.config.in_channels, inner_dim,
                kernel_size=(1, 2, 2), stride=(1, 2, 2),
            )
            self.frame_pack_proj_2x = ReplicatedConv3D(
                self.config.in_channels, inner_dim,
                kernel_size=(2, 4, 4), stride=(2, 4, 4),
            )
            self.frame_pack_proj_4x = ReplicatedConv3D(
                self.config.in_channels, inner_dim,
                kernel_size=(4, 8, 8), stride=(4, 8, 8),
            )

        self.condition_embedder = LiveAvatarTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=self.config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=self.config.text_dim,
            zero_timestep=self.config.arch_config.zero_timestep,
        )

        self.audio_encoder = CausalAudioEncoder(
            audio_dim=self.config.arch_config.audio_dim,
            hidden_dim=inner_dim,
            num_layers=25,
            num_tokens=self.config.arch_config.num_audio_token,
            need_global=self.config.arch_config.enable_adain,
        )
        self.audio_injector = AudioInjector(
            dim=inner_dim,
            num_heads=self.config.num_attention_heads,
            inject_layers=self.config.arch_config.audio_inject_layers,
            num_transformer_layers=self.config.num_layers,
            enable_adain=self.config.arch_config.enable_adain,
        )

        self.trainable_cond_mask = nn.Embedding(3, inner_dim)

        self.blocks = nn.ModuleList(
            [
                LiveAvatarTransformerBlock(
                    inner_dim,
                    self.config.ffn_dim,
                    self.config.num_attention_heads,
                    layer_id,
                    self.config.qk_norm,
                    self.config.cross_attn_norm,
                    self.config.eps,
                )
                for layer_id in range(self.config.num_layers)
            ]
        )

        self.norm_out = FP32LayerNormScaleShift(
            inner_dim, self.config.eps, elementwise_affine=False, bias=False,
        )
        self.proj_out = ReplicatedLinear(
            inner_dim, out_channels * math.prod(self.config.patch_size),
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        buckets = list(getattr(self.config.arch_config, "zip_frame_buckets", [1, 2, 16]))
        self._motion_frames_default = int(sum(buckets))
        motion_frames_cfg = int(getattr(self.config.arch_config, "motion_frames", 0) or 0)
        if motion_frames_cfg > 0:
            self._motion_frames_default = motion_frames_cfg
        self._motion_latent_frames_default = int((self._motion_frames_default + 3) // 4)
        self._zip_frame_buckets = buckets
        self._wan_freqs = self._build_wan_freqs(
            head_dim=self.attention_head_dim,
            max_seq_len=self.config.rope_max_seq_len,
        )

    @staticmethod
    def _normalize_motion_latents(
        motion_latents: torch.Tensor,
        ref_latents: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, ref_h, ref_w = ref_latents.shape
        bsz, channels, t_m, h_m, w_m = motion_latents.shape

        if h_m == ref_h and w_m == ref_w:
            return motion_latents

        # Recover [B,C,1,T*H,W] -> [B,C,T,H,W] when ref spatial shape is known
        if t_m == 1 and w_m == ref_w and ref_h > 0 and h_m % ref_h == 0:
            recovered_t = h_m // ref_h
            if recovered_t > 1:
                return motion_latents.reshape(
                    bsz,
                    channels,
                    recovered_t,
                    ref_h,
                    ref_w,
                ).contiguous()

        raise ValueError(
            "motion_latents spatial mismatch against reference latents: "
            f"motion_shape={tuple(motion_latents.shape)}, ref_shape={tuple(ref_latents.shape)}"
        )

    @staticmethod
    def _rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
        assert dim % 2 == 0
        pos = torch.arange(max_seq_len, dtype=torch.float64)
        freq_scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        freqs = torch.outer(pos, 1.0 / (theta ** freq_scale))
        return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)

    def _build_wan_freqs(self, head_dim: int, max_seq_len: int) -> torch.Tensor:
        return torch.cat(
            [
                self._rope_params(max_seq_len, head_dim - 4 * (head_dim // 6)),
                self._rope_params(max_seq_len, 2 * (head_dim // 6)),
                self._rope_params(max_seq_len, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

    @staticmethod
    def _sample_grid_indices(start: int, span: int, target_span: int, max_index: int, device: torch.device):
        if span <= 0:
            return torch.empty((0,), dtype=torch.long, device=device), False

        if start >= 0:
            idx = torch.linspace(
                float(start),
                float(start + target_span - 1),
                steps=span,
                device=device,
            ).to(torch.long)
            return idx.clamp_(0, max_index), False

        idx = torch.linspace(
            float(-start),
            float(-target_span - start + 1),
            steps=span,
            device=device,
        ).to(torch.long)
        return idx.clamp_(0, max_index), True

    def _grid_rope_chunk(
        self,
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        target: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        f_o, h_o, w_o = start
        f, h, w = end
        t_f, t_h, t_w = target

        seq_f = int(f - f_o)
        seq_h = int(h - h_o)
        seq_w = int(w - w_o)
        if seq_f <= 0 or seq_h <= 0 or seq_w <= 0:
            return torch.empty((0, self.attention_head_dim // 2), dtype=torch.complex64, device=device)

        half = self.attention_head_dim // 2
        split_sizes = [half - 2 * (half // 3), half // 3, half // 3]
        freqs_f, freqs_h, freqs_w = self._wan_freqs.to(device).split(split_sizes, dim=1)
        max_idx = freqs_f.shape[0] - 1

        f_idx, use_conj = self._sample_grid_indices(f_o, seq_f, int(t_f), max_idx, device)
        h_idx, _ = self._sample_grid_indices(h_o, seq_h, int(t_h), max_idx, device)
        w_idx, _ = self._sample_grid_indices(w_o, seq_w, int(t_w), max_idx, device)

        f_part = freqs_f[f_idx]
        if use_conj:
            f_part = f_part.conj()

        f_part = f_part.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1)
        h_part = freqs_h[h_idx].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1)
        w_part = freqs_w[w_idx].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1)

        return torch.cat([f_part, h_part, w_part], dim=-1).reshape(seq_f * seq_h * seq_w, -1)

    def _build_condition_rotary(
        self,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, h_lat, w_lat = ref_latents.shape
        _, p_h, p_w = self.patch_size
        h = h_lat // p_h
        w = w_lat // p_w

        chunks = [
            self._grid_rope_chunk(
                start=(30, 0, 0),
                end=(31, h, w),
                target=(1, h, w),
                device=ref_latents.device,
            )
        ]

        if self.config.arch_config.enable_framepack and len(self._zip_frame_buckets) >= 3:
            b1, b2, b3 = [int(v) for v in self._zip_frame_buckets[:3]]
            chunks.append(
                self._grid_rope_chunk(
                    start=(-b1, 0, 0),
                    end=(0, h, w),
                    target=(b1, h, w),
                    device=ref_latents.device,
                )
            )
            chunks.append(
                self._grid_rope_chunk(
                    start=(-(b1 + b2), 0, 0),
                    end=(-(b1 + b2) + (b2 // 2), h // 2, w // 2),
                    target=(b2, h, w),
                    device=ref_latents.device,
                )
            )
            chunks.append(
                self._grid_rope_chunk(
                    start=(-(b1 + b2 + b3), 0, 0),
                    end=(-(b1 + b2 + b3) + (b3 // 4), h // 4, w // 4),
                    target=(b3, h, w),
                    device=ref_latents.device,
                )
            )

        rope_complex = torch.cat(chunks, dim=0)
        return torch.cat([rope_complex.real, rope_complex.imag], dim=-1).to(dtype=torch.float32)

    def _build_motion_framepack_tokens(self, motion_latents: torch.Tensor) -> torch.Tensor:
        if not self.config.arch_config.enable_framepack:
            return torch.empty(
                motion_latents.shape[0],
                0,
                self.inner_dim,
                device=motion_latents.device,
                dtype=motion_latents.dtype,
            )

        b1, b2, b3 = self._zip_frame_buckets
        total = b1 + b2 + b3
        bsz, channels, t, h_lat, w_lat = motion_latents.shape
        padded = torch.zeros(
            bsz,
            channels,
            total,
            h_lat,
            w_lat,
            device=motion_latents.device,
            dtype=motion_latents.dtype,
        )
        overlap = min(total, int(t))
        if overlap > 0:
            padded[:, :, -overlap:] = motion_latents[:, :, -overlap:]

        clean_4x, clean_2x, clean_post = padded.split([b3, b2, b1], dim=2)
        tok_post = self.frame_pack_proj(clean_post).flatten(2).transpose(1, 2)
        tok_2x = self.frame_pack_proj_2x(clean_2x).flatten(2).transpose(1, 2)
        tok_4x = self.frame_pack_proj_4x(clean_4x).flatten(2).transpose(1, 2)
        return torch.cat([tok_post, tok_2x, tok_4x], dim=1)

    def _build_condition_tokens(
        self,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        ref_latents = ref_latents[:, :, :1].contiguous()
        ref_tokens = self.patch_embedding(ref_latents).flatten(2).transpose(1, 2)
        ref_token_count = int(ref_tokens.shape[1])
        motion_tokens = self._build_motion_framepack_tokens(motion_latents)
        motion_token_count = int(motion_tokens.shape[1])
        cond_tokens = torch.cat([ref_tokens, motion_tokens], dim=1)
        cond_rotary = self._build_condition_rotary(ref_latents, motion_latents)
        return cond_tokens, cond_rotary, ref_token_count, motion_token_count

    def _prefill_condition_cache(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> None:
        ref_latents = forward_batch_info.ref_latents
        if ref_latents is None:
            ref_latents = hidden_states[:, :, :1].detach().clone()
        ref_latents = ref_latents[:, :, :1].contiguous()

        motion_latents = forward_batch_info.motion_latents
        if motion_latents is None:
            motion_len = max(1, self._motion_latent_frames_default)
            motion_latents = ref_latents.repeat(1, 1, motion_len, 1, 1)
        
        motion_latents = self._normalize_motion_latents(motion_latents, ref_latents)

        cond_tokens, cond_rotary, ref_token_count, motion_token_count = self._build_condition_tokens(
            ref_latents,
            motion_latents,
        )

        assert cond_rotary.shape[0] == cond_tokens.shape[1]

        cond_mask = torch.ones(
            cond_tokens.shape[0],
            cond_tokens.shape[1],
            dtype=torch.long,
            device=cond_tokens.device,
        )
        if cond_tokens.shape[1] > ref_token_count:
            cond_mask[:, ref_token_count:] = 2
        cond_tokens = cond_tokens + self.trainable_cond_mask(cond_mask).to(cond_tokens.dtype)

        expected_tokens = int(forward_batch_info.cache_end - forward_batch_info.cache_start)
        assert cond_tokens.shape[1] == expected_tokens

        zero_t = torch.zeros(
            (cond_tokens.shape[0],),
            device=cond_tokens.device,
            dtype=timestep.dtype,
        )
        _, timestep_proj = self.condition_embedder(zero_t)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        hidden = cond_tokens
        for block in self.blocks:
            hidden = block(
                hidden,
                timestep_proj,
                cond_rotary,
                forward_batch_info,
            )

    def _inject_audio(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        num_noisy_tokens: int,
        forward_batch_info: ForwardBatchInfo,
    ) -> torch.Tensor:
        if block_idx not in self.audio_injector.injected_block_id:
            return hidden_states

        attn_id = self.audio_injector.injected_block_id[block_idx]
        audio_emb = self._audio_local_emb   # (B, F, N, C)
        num_frames = audio_emb.shape[1]
        if num_frames <= 0:
            return hidden_states
        assert (num_noisy_tokens % num_frames) == 0

        vis = hidden_states[:, :num_noisy_tokens].clone()
        hw = num_noisy_tokens // num_frames
        vis = vis.reshape(vis.shape[0] * num_frames, hw, vis.shape[-1])

        if self.audio_injector.enable_adain:
            audio_global = self._audio_global_emb.to(hidden_states.dtype)
            audio_global_flat = audio_global.reshape(
                audio_global.shape[0] * num_frames, 1, audio_global.shape[-1]
            )
            vis = self.audio_injector.injector_adain_layers[attn_id](
                vis, temb=audio_global_flat[:, 0]
            )
        else:
            vis = self.audio_injector.injector_pre_norm_feat[attn_id](vis)

        vis = vis.to(hidden_states.dtype)
        audio_flat = audio_emb.reshape(
            audio_emb.shape[0] * num_frames, audio_emb.shape[2], audio_emb.shape[3]
        )
        audio_flat = audio_flat.to(vis.dtype)
        residual = self.audio_injector.injector[attn_id](
            x=vis,
            context=audio_flat,
            forward_batch_info=forward_batch_info,
        )
        residual = residual.reshape(
            hidden_states.shape[0], num_frames * hw, hidden_states.shape[-1]
        )
        residual = residual.to(hidden_states.dtype)
        hidden_states[:, :num_noisy_tokens] = (
            hidden_states[:, :num_noisy_tokens] + residual
        )
        return hidden_states

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        rotary_emb: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if forward_batch_info.prefill_cond:
            self._audio_local_emb = None
            self._audio_global_emb = None
            self._prefill_condition_cache(hidden_states, timestep, forward_batch_info)
            return

        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = sequence_model_parallel_shard(hidden_states, dim=2)
        timestep = sequence_model_parallel_shard(timestep, dim=0)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        num_noisy_tokens = hidden_states.shape[1]

        if forward_batch_info.cond_latents is not None:
            cond = sequence_model_parallel_shard(forward_batch_info.cond_latents, dim=2)
            cond = self.cond_encoder(cond)
            cond = cond.flatten(2).transpose(1, 2)
            hidden_states = hidden_states + cond

        mask_input = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], dtype=torch.long, device=hidden_states.device,
        )
        hidden_states = hidden_states + self.trainable_cond_mask(mask_input).to(hidden_states.dtype)

        if forward_batch_info.audio_input is not None:
            raw_prefix = int(forward_batch_info.motion_frames_raw or 0)
            lat_prefix = int(forward_batch_info.motion_frames_latent or 0)
            if raw_prefix <= 0:
                raw_prefix = self._motion_frames_default
            if lat_prefix <= 0:
                lat_prefix = self._motion_latent_frames_default

            audio_input = forward_batch_info.audio_input
            audio_input = torch.cat(
                [audio_input[..., 0:1].repeat(1, 1, 1, raw_prefix), audio_input],
                dim=-1,
            ).to(dtype=hidden_states.dtype)

            audio_res = self.audio_encoder(audio_input)
            if self.config.arch_config.enable_adain:
                audio_global, audio_local = audio_res
                self._audio_global_emb = audio_global[:, lat_prefix:].clone().to(hidden_states.dtype)
                self._audio_local_emb = audio_local[:, lat_prefix:].to(hidden_states.dtype)
            else:
                self._audio_local_emb = audio_res[:, lat_prefix:].to(hidden_states.dtype)
                self._audio_global_emb = None
        else:
            self._audio_local_emb = None
            self._audio_global_emb = None

        temb, timestep_proj = self.condition_embedder(timestep)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                timestep_proj,
                rotary_emb,
                forward_batch_info,
            )
            if self._audio_local_emb is not None:
                hidden_states = self._inject_audio(
                    idx,
                    hidden_states,
                    num_noisy_tokens,
                    forward_batch_info,
                )

        if forward_batch_info.is_cache:
            return

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
        forward_batch_info: ForwardBatchInfo,
    ):
        h = self.condition_embedder.text_embedder(encoder_hidden_states)

        for idx, block in enumerate(self.blocks):
            attn = block.attn2
            k, v = attn.get_encoder_kv(h)
            forward_batch_info.kv_memory.set_encoder_kv(idx, k, v)

    def optimize(self, **kwargs):
        for module in self.blocks:
            module.optimize(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for raw_name, loaded_weight in weights:

            name = map_param_name(raw_name, self.config.arch_config.param_names_mapping)
            if "rotary_emb.inv_freq" in name:
                continue
            if "freqs" == name:
                continue
            if "scale" in name:
                kv_scale_name: str | None = maybe_remap_kv_scale_name(
                    name, params_dict
                )
                if kv_scale_name is None:
                    continue
                else:
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


EntryClass = LiveAvatarTransformer3DModel
