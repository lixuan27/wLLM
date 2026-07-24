# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from wllm.serving.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LiveAvatarArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # --- base embeddings ---
            r"^casual_audio_encoder\.(.*)$":
            r"audio_encoder.\1",
            r"^frame_packer\.proj\.(.*)$":
            r"frame_pack_proj.\1",
            r"^frame_packer\.proj_2x\.(.*)$":
            r"frame_pack_proj_2x.\1",
            r"^frame_packer\.proj_4x\.(.*)$":
            r"frame_pack_proj_4x.\1",
            # --- self-attention ---
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$":
            r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$":
            r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$":
            r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.attn1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn1.norm_k.\2",
            # --- cross-attention (text) ---
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$":
            r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$":
            r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$":
            r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$":
            r"blocks.\1.norm2.\2",
            # --- modulation ---
            r"^blocks\.(\d+)\.modulation$":
            r"blocks.\1.scale_shift_table",
            # --- FFN ---
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$":
            r"blocks.\1.ffn.net.0.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$":
            r"blocks.\1.ffn.net.3.\2",
            # --- head ---
            r"^head\.modulation$":
            r"scale_shift_table",
            r"^head\.head\.(.*)$":
            r"proj_out.\1",
            # --- text embedding (sequential) ---
            r"^text_embedding\.0\.(.*)$":
            r"condition_embedder.text_embedder.linear_1.\1",
            r"^text_embedding\.2\.(.*)$":
            r"condition_embedder.text_embedder.linear_2.\1",
            # --- time embedding (sequential) ---
            r"^time_embedding\.0\.(.*)$":
            r"condition_embedder.time_embedder.linear_1.\1",
            r"^time_embedding\.2\.(.*)$":
            r"condition_embedder.time_embedder.linear_2.\1",
            # --- time projection ---
            r"^time_projection\.1\.(.*)$":
            r"condition_embedder.time_proj.\1",
            # --- HF-style .0. in to_out ---
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"blocks.\1.attn1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.to_out_prope\.0\.(.*)$":
            r"blocks.\1.attn1.to_out_prope.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            # --- FFN (HF diffusers convention) ---
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$":
            r"blocks.\1.ffn.net.0.\2",
            r"^blocks\.(\d+)\.ffn\.net\.1\.(.*)$":
            r"blocks.\1.ffn.net.2.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$":
            r"blocks.\1.ffn.net.3.\2",
            # --- audio injector ---
            r"^audio_injector\.injector\.(\d+)\.q\.(.*)$":
            r"audio_injector.injector.\1.to_q.\2",
            r"^audio_injector\.injector\.(\d+)\.k\.(.*)$":
            r"audio_injector.injector.\1.to_k.\2",
            r"^audio_injector\.injector\.(\d+)\.v\.(.*)$":
            r"audio_injector.injector.\1.to_v.\2",
            r"^audio_injector\.injector\.(\d+)\.o\.(.*)$":
            r"audio_injector.injector.\1.to_out.\2",
        })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.net.0.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.net.3.\2",
            # Audio injector: q/k/v/o → to_q/to_k/to_v/to_out
            r"^audio_injector\.injector\.(\d+)\.q\.(.*)$":
            r"audio_injector.injector.\1.to_q.\2",
            r"^audio_injector\.injector\.(\d+)\.k\.(.*)$":
            r"audio_injector.injector.\1.to_k.\2",
            r"^audio_injector\.injector\.(\d+)\.v\.(.*)$":
            r"audio_injector.injector.\1.to_v.\2",
            r"^audio_injector\.injector\.(\d+)\.o\.(.*)$":
            r"audio_injector.injector.\1.to_out.\2",
        })

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
    image_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    boundary_ratio: float | None = None

    # --- S2V audio conditioning ---
    audio_dim: int = 1024          # Wav2Vec2 hidden dimension
    num_audio_token: int = 4       # tokens per frame from audio encoder
    enable_adain: bool = True
    adain_mode: str = "attn_norm"
    audio_inject_layers: list[int] = field(
        default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]
    )

    # --- Pose conditioning ---
    cond_dim: int = 16

    # --- Motion (framepack) ---
    enable_framepack: bool = True
    framepack_drop_mode: str = "padd"
    zip_frame_buckets: list[int] = field(default_factory=lambda: [1, 2, 16])

    # --- Dual-timestep ---
    zero_timestep: bool = True
    zero_init: bool = True

    # --- legacy Wan2.2-S2V config keys retained for compatibility ---
    model_type: str = "s2v"
    enable_motioner: bool = False
    add_last_motion: bool = True
    enable_tsm: bool = False
    trainable_token_pos_emb: bool = False
    motion_token_num: int = 1024
    # Raw motion context frames used by reference LiveAvatar TPP runtime.
    motion_frames: int = 73

    # Causal parameters
    local_attn_size: int = -1
    sink_size: int = 0
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels

        if self.enable_framepack:
            if len(self.zip_frame_buckets) < 3:
                raise ValueError(
                    "LiveAvatar framepack requires three zip_frame_buckets values"
                )
            if any(int(v) <= 0 for v in self.zip_frame_buckets[:3]):
                raise ValueError(
                    f"Invalid zip_frame_buckets={self.zip_frame_buckets}; first three entries must be > 0"
                )

        if int(self.motion_frames) <= 0:
            raise ValueError(f"motion_frames must be > 0, got {self.motion_frames}")


@dataclass
class LiveAvatarConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LiveAvatarArchConfig)

    prefix: str = "LiveAvatar"
