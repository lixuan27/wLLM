from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.runner.forward_batch import ForwardBatchInfo

class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        **conv_kwargs,
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self._time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, **conv_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

class AudioTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        need_global: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.need_global = need_global

        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3)
        self.norm1_local = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)

        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3)
            self.norm1_global = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)

        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def _down_sample(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.conv3(x)
        x = x.transpose(1, 2)
        x = self.norm3(x)
        x = self.act(x)
        return x

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x_ct = x.transpose(1, 2)  # (B, C, T)

        local = self.conv1_local(x_ct)  # (B, N*C_h, T)
        local = local.view(b * self.num_heads, -1, local.shape[2])  # (B*N, C_h, T)
        local = local.transpose(1, 2)
        local = self.norm1_local(local)
        local = self.act(local)
        local = local.transpose(1, 2)
        local = self._down_sample(local)  # (B*N, T', C_out)
        local = local.view(b, self.num_heads, local.shape[1], local.shape[2])
        local = local.permute(0, 2, 1, 3)  # (B, T', N, C_out)
        padding = self.padding_tokens.expand(b, local.shape[1], -1, -1)
        local = torch.cat([local, padding], dim=2)  # (B, T', N+1, C_out)

        if not self.need_global:
            return local

        glob = self.conv1_global(x_ct)  # (B, C_h, T)
        glob = glob.transpose(1, 2)
        glob = self.norm1_global(glob)
        glob = self.act(glob)
        glob = glob.transpose(1, 2)
        glob = self._down_sample(glob)  # (B, T', C_out)
        glob = self.final_linear(glob)
        glob = glob.view(b, 1, glob.shape[1], glob.shape[2])
        glob = glob.permute(0, 2, 1, 3)  # (B, T', 1, C_out)

        return glob, local


class CausalAudioEncoder(nn.Module):
    def __init__(
        self,
        audio_dim: int = 1024,
        hidden_dim: int = 5120,
        num_layers: int = 25,
        num_tokens: int = 4,
        need_global: bool = True,
    ):
        super().__init__()
        self.encoder = AudioTemporalEncoder(
            in_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_heads=num_tokens,
            need_global=need_global,
        )
        self.weights = nn.Parameter(torch.ones(1, num_layers, 1, 1) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: ``(B, num_layers, C_audio, T_audio)``
        """
        with torch.amp.autocast("cuda", dtype=torch.float32):
            w = self.act(self.weights)
            w_sum = w.sum(dim=1, keepdim=True)
            weighted = (features * w / w_sum).sum(dim=1)  # (B, C_audio, T)
            weighted = weighted.permute(0, 2, 1)  # (B, T, C_audio)
            return self.encoder(weighted)


class AudioCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)

        if qk_norm:
            from wllm.serving.layers.layernorm import RMSNorm
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: visual hidden states ``(B, S, C)``
            context: audio embeddings ``(B, S_a, C)``
        Returns:
            Residual contribution ``(B, S, C)``
        """
        context = context.to(x.dtype)

        q, _ = self.to_q(x)
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.unflatten(2, (self.num_heads, self.head_dim))
        k = k.unflatten(2, (self.num_heads, self.head_dim))
        v = v.unflatten(2, (self.num_heads, self.head_dim))

        out = forward_batch_info.attention_backend.run(q, k, v)
        out = out.flatten(2, 3)
        out = out.type_as(q)
        out, _ = self.to_out(out)
        return out


class AudioInjector(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        inject_layers: Sequence[int],
        num_transformer_layers: int,
        enable_adain: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.inject_layers = list(inject_layers)

        self.injected_block_id: dict[int, int] = {}
        idx = 0
        for layer_id in self.inject_layers:
            if layer_id < num_transformer_layers:
                self.injected_block_id[layer_id] = idx
                idx += 1
        num_injectors = idx

        self.injector = nn.ModuleList([
            AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True, eps=eps)
            for _ in range(num_injectors)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            for _ in range(num_injectors)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            for _ in range(num_injectors)
        ])

        self.enable_adain = enable_adain
        if enable_adain:
            from diffusers.models.attention import AdaLayerNorm
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(output_dim=dim * 2, embedding_dim=dim, chunk_dim=1)
                for _ in range(num_injectors)
            ])
