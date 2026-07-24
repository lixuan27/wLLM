# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from wllm.serving.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class WanVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"blocks.\1.attn1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.to_out_prope\.0\.(.*)$":
            r"blocks.\1.attn1.to_out_prope.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$":
            r"blocks.\1.ffn.net.0.\2",
            r"^blocks\.(\d+)\.ffn\.net\.1\.(.*)$":
            r"blocks.\1.ffn.net.2.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$":
            r"blocks.\1.ffn.net.3.\2",
        })

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Some LoRA adapters use the original official layer names instead of hf layer names,
    # so apply this before the param_names_mapping
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.attn1.to_out.0.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.0.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
        })

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len = 512
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
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    # Wan MoE
    boundary_ratio: float | None = None

    # Causal Wan
    local_attn_size: int = -1  # Window size for temporal local attention (-1 indicates global attention)
    sink_size: int = 0  # Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanVideoArchConfig)

    prefix: str = "Wan"
