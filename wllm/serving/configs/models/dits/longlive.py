# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from wllm.serving.configs.models.dits.base import DiTArchConfig, DiTConfig
from wllm.serving.configs.models.dits.krea_realtime import KreaRealtimeArchConfig


@dataclass
class LongLiveArchConfig(KreaRealtimeArchConfig):
    # Wan2.2-TI2V-5B dimensions; everything else (param mapping, qk_norm,
    # cross_attn_norm, patch_size) is inherited from KreaRealtimeArchConfig.
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30


@dataclass
class LongLiveConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LongLiveArchConfig)

    prefix: str = "LongLive"
