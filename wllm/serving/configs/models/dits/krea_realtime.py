# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from wllm.serving.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(name: str, _) -> bool:
    return "blocks" in name and str.isdigit(name.split(".")[-1])


@dataclass
class KreaRealtimeArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(?:model\.)?patch_embedding\.(.*)$": r"patch_embedding.\1",
            r"^(?:model\.)?text_embedding\.0\.(.*)$":
            r"condition_embedder.text_embedder.linear_1.\1",
            r"^(?:model\.)?text_embedding\.2\.(.*)$":
            r"condition_embedder.text_embedder.linear_2.\1",
            r"^(?:model\.)?time_embedding\.0\.(.*)$":
            r"condition_embedder.time_embedder.linear_1.\1",
            r"^(?:model\.)?time_embedding\.2\.(.*)$":
            r"condition_embedder.time_embedder.linear_2.\1",
            r"^(?:model\.)?time_projection\.1\.(.*)$":
            r"condition_embedder.time_proj.\1",
            r"^(?:model\.)?head\.modulation$": r"scale_shift_table",
            r"^(?:model\.)?head\.head\.(.*)$": r"proj_out.\1",
            r"^(?:model\.)?blocks\.(\d+)\.modulation$":
            r"blocks.\1.scale_shift_table",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.q\.(.*)$":
            r"blocks.\1.attn1.to_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.k\.(.*)$":
            r"blocks.\1.attn1.to_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.v\.(.*)$":
            r"blocks.\1.attn1.to_v.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.attn1.to_out.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn1.norm_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn1.norm_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.q\.(.*)$":
            r"blocks.\1.attn2.to_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.k\.(.*)$":
            r"blocks.\1.attn2.to_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.v\.(.*)$":
            r"blocks.\1.attn2.to_v.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn2.norm_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn2.norm_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.norm3\.(.*)$":
            r"blocks.\1.norm2.\2",
            r"^(?:model\.)?blocks\.(\d+)\.ffn\.0\.(.*)$":
            r"blocks.\1.ffn.net.0.\2",
            r"^(?:model\.)?blocks\.(\d+)\.ffn\.2\.(.*)$":
            r"blocks.\1.ffn.net.3.\2",
        }
    )

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.q\.(.*)$":
            r"blocks.\1.attn1.to_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.k\.(.*)$":
            r"blocks.\1.attn1.to_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.v\.(.*)$":
            r"blocks.\1.attn1.to_v.\2",
            r"^(?:model\.)?blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.attn1.to_out.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.q\.(.*)$":
            r"blocks.\1.attn2.to_q.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.k\.(.*)$":
            r"blocks.\1.attn2.to_k.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.v\.(.*)$":
            r"blocks.\1.attn2.to_v.\2",
            r"^(?:model\.)?blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^(?:model\.)?blocks\.(\d+)\.ffn\.0\.(.*)$":
            r"blocks.\1.ffn.net.0.\2",
            r"^(?:model\.)?blocks\.(\d+)\.ffn\.2\.(.*)$":
            r"blocks.\1.ffn.net.3.\2",
        }
    )

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    local_attn_size: int = -1
    sink_size: int = 0
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class KreaRealtimeConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=KreaRealtimeArchConfig)

    prefix: str = "KreaRealtime"
