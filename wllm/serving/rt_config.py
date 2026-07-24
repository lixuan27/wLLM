from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import yaml

from diffusers.configuration_utils import ConfigMixin
from wllm.serving.models.registry import ModelRegistry


@dataclass
class DiTConfig:
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    out_channels: int
    rope_max_seq_len: int
    text_dim: int
    patch_size: List[int]
    theta: float = 10000.0
    in_channels: Optional[int] = None


@dataclass
class VAEConfig:
    scale_factor_spatial: int
    scale_factor_temporal: int
    out_channels: int
    z_dim: int
    patch_size: int
    latents_mean: List[float]
    latents_std: List[float]


@dataclass
class RTConfig:
    # -------- basic runtime --------
    model_name: Optional[str] = None
    seed: int = 42
    prompt: str = ""
    image_path: Optional[str] = None

    # -------- model paths --------
    tokenizer_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    text_encoder_name: Optional[str] = None
    transformer_path: Optional[str] = None
    transformer_path_high: Optional[str] = None
    transformer_path_low: Optional[str] = None
    vae_name: Optional[str] = None
    vae_path: Optional[str] = None
    wav2vec2_path: Optional[str] = None

    # -------- dual-model / CFG settings --------
    boundary_ratio: Optional[float] = None
    negative_prompt: str = ""
    
    # -------- generation settings --------
    max_sequence_length: int = 512
    max_num_frames: int = 285
    height: int = 704
    width: int = 1280
    num_videos_per_prompt: int = 1
    num_inference_steps: int = 4
    first_chunk_size: int = 4
    chunk_size: int = 4
    context_window_size: int = 16
    continuous_decoding: bool = False
    reactive_decoding: bool = False
    stabilization_level: int = 15
    guidance_scale: float = 1.0
    guidance_scale_2: Optional[float] = None

    # -------- derived / optional overrides --------
    max_num_actions: Optional[int] = None
    audio_frame_samples: int = 960
    audio_max_chunks: int = 4096
    audio_sample_rate: int = 48000
    latent_height: Optional[int] = None
    latent_width: Optional[int] = None
    latent_spatial: Optional[int] = None
    kv_height: Optional[int] = None
    kv_width: Optional[int] = None
    kv_spatial: Optional[int] = None

    # -------- condition-cache / motion-prefix settings --------
    kv_cond_tokens: Optional[int] = None
    motion_token_buckets: Optional[List[int]] = None
    motion_prefix_frames: Optional[int] = None
    motion_prefix_latent_frames: Optional[int] = None

    # -------- optional runtime controls --------
    attention_backend: str = "auto"
    kv_memory: str = "preallocated"
    use_cuda_graph: bool = False
    capture_shapes: Optional[List[Tuple[int, int, bool]]] = None

    # -------- experiment metadata --------
    video_buffer_name: Optional[str] = None
    ctrl_buffer_name: Optional[str] = None
    action_buffer_name: Optional[str] = None
    audio_buffer_name: Optional[str] = None

    # -------- video-input pipeline settings (Krea v2v) --------
    video_input_buffer_name: Optional[str] = None
    video_input_max_frames: int = 256
    video_fps: int = 30

    # -------- Krea-Realtime denoising / context settings --------
    denoising_strength: float = 1.0
    timestep_shift: float = 5.0
    keep_first_frame: bool = False

    # -------- LongLive sliding-window KV cache + multi-shot settings --------
    sink_size: int = 0
    multi_shot_rope_offset: int = 0
    scene_cut_prefix: str = "The scene transitions. "

    # -------- SAM 3 (background-only style transfer) settings --------
    sam_text_prompt: str = "person"
    sam_min_score: float = 0.5
    sam_mask_threshold: float = 0.5
    sam_dilate_pixels: int = 0
    sam_disable: bool = False

    # -------- audio output + pipeline settings (S2V / TTS) --------
    audio_output_buffer_name: Optional[str] = None
    signal_buffer_name: Optional[str] = None
    tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_voice: str = "vivian"
    tts_language: str = "English"
    tts_chunk_size: int = 7680
    tts_prompt_token_count: int = 2048
    tts_stage_configs_path: Optional[str] = None
    tts_gpu_index: Optional[int] = None
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_gpu_index: Optional[int] = None
    llm_gpu_memory_utilization: float = 0.3
    llm_max_model_len: int = 8192
    llm_max_completion_tokens: int = 80
    llm_temperature: float = 0.8
    llm_top_p: float = 0.95
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    asr_gpu_index: Optional[int] = None
    audio_stretch_ratio: float = 1.0

    # -------- LoRA settings --------
    lora_path: Optional[str] = None
    lora_rank: int = 128
    lora_alpha: float = 64.0
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    # -------- device settings --------
    device: str = "cuda"
    dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"

    # -------- distributed settings --------
    tp_size: int = 1
    sp_size: int = 1
    pp_size: int = 1

    # -------- nested model configs (auto-loaded) --------
    dit_config: Optional[DiTConfig] = None
    vae_config: Optional[VAEConfig] = None

    # -------- behavior knobs --------
    # If True: always recompute derived fields (max_num_actions/latent_*/kv_*)
    # If False: only fill them when they are None (YAML can override)
    recompute_derived: bool = True

    def __post_init__(self) -> None:
        """
        Execution flow:
        1) Load DiT/VAE configs from config.json if paths are provided.
        2) Validate key dimensions.
        3) Compute derived fields (optionally respecting YAML overrides).
        """

        # 1) Load sub-configs
        if self.transformer_path is not None and self.dit_config is None:
            raw = self._load_json_config(self.transformer_path)
            self.dit_config = self._build_dit_config(raw)

        if self.vae_path is not None and self.vae_config is None:
            raw = self._load_json_config(self.vae_path)
            # older VAE configs lack the scale_factor_* keys; derive them
            sf_sp = raw.get("scale_factor_spatial")
            if sf_sp is None and raw.get("dim_mult"):
                sf_sp = 2 ** (len(raw["dim_mult"]) - 1) * (raw.get("patch_size") or 1)
            sf_tp = raw.get("scale_factor_temporal")
            if sf_tp is None and raw.get("temperal_downsample") is not None:
                sf_tp = 2 ** sum(bool(x) for x in raw["temperal_downsample"])
            self.vae_config = VAEConfig(
                scale_factor_spatial=sf_sp,
                scale_factor_temporal=sf_tp,
                out_channels=raw.get("out_channels"),
                z_dim=raw.get("z_dim"),
                patch_size=raw.get("patch_size"),
                latents_mean=raw.get("latents_mean"),
                latents_std=raw.get("latents_std")
            )

        # 2) Validate requirements before computing derived values
        if self.vae_config is None:
            raise ValueError(
                "vae_config is None. Provide vae_path (with config.json) or pass vae_config explicitly."
            )
        if self.dit_config is None:
            raise ValueError(
                "dit_config is None. Provide transformer_path (with config.json) or pass dit_config explicitly."
            )

        self._validate_dims()

        # 3) Compute derived values
        self._compute_derived()

        # Convert capture_shapes to List[Tuple[int, int, bool]]
        if self.capture_shapes is not None:
            self.capture_shapes = [
                (int(a), int(b), bool(c))
                for a, b, c in self.capture_shapes
            ]

    def _build_dit_config(self, raw: Dict[str, Any]) -> DiTConfig:
        model_cls, _ = ModelRegistry.resolve_model_cls(self.model_name)
        arch_defaults = model_cls._default_config.arch_config

        def pick_first(raw, keys, default=None):
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            return default

        num_layers = raw.get("num_layers", arch_defaults.num_layers)
        num_attention_heads = pick_first(raw, ["num_attention_heads", "num_heads"], arch_defaults.num_attention_heads)
        head_dim = raw.get("attention_head_dim", arch_defaults.attention_head_dim)
    
        in_channels = pick_first(raw, ["in_channels", "in_dim"], arch_defaults.in_channels)
        out_channels = pick_first(raw, ["out_channels", "out_dim"], arch_defaults.out_channels)
        rope_max_seq_len = raw.get("rope_max_seq_len", arch_defaults.rope_max_seq_len)
        text_dim = raw.get("text_dim", arch_defaults.text_dim)
        patch_size = raw.get("patch_size", arch_defaults.patch_size)

        if self.motion_token_buckets is None:
            self.motion_token_buckets = raw.get("zip_frame_buckets", getattr(arch_defaults, "zip_frame_buckets", None))

        if self.motion_prefix_frames is None:
            self.motion_prefix_frames = raw.get("motion_frames", getattr(arch_defaults, "motion_frames", None))

        return DiTConfig(
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=head_dim,
            out_channels=out_channels,
            in_channels=in_channels,
            rope_max_seq_len=rope_max_seq_len,
            text_dim=text_dim,
            patch_size=patch_size,
        )
    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @staticmethod
    def _load_json_config(path_or_dir: str) -> Dict[str, Any]:
        """
        Accepts either:
          - a directory containing config.json
          - a direct path to config.json
        """
        p = str(path_or_dir)

        # If user passed ".../config.json"
        if p.endswith(".json"):
            json_path = p
        else:
            json_path = os.path.join(p, "config.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config file not found: {json_path}")

        return ConfigMixin._dict_from_json_file(json_path)

    def _validate_dims(self) -> None:
        """
        Validate divisibility and patch_size indexing assumptions.
        """

        # VAE scaling must produce integer latent dims
        sf_sp = self.vae_config.scale_factor_spatial
        if sf_sp is None or sf_sp <= 0:
            raise ValueError(f"Invalid vae_config.scale_factor_spatial: {sf_sp}")

        if self.height % sf_sp != 0:
            raise ValueError(
                f"height ({self.height}) must be divisible by vae scale_factor_spatial ({sf_sp})"
            )
        if self.width % sf_sp != 0:
            raise ValueError(
                f"width ({self.width}) must be divisible by vae scale_factor_spatial ({sf_sp})"
            )

        # patch_size must have at least 3 elements since you use [1] and [2]
        ps = list(self.dit_config.patch_size)
        if len(ps) < 3:
            raise ValueError(
                f"dit_config.patch_size must be a list with length >= 3, got: {ps}"
            )

        if self.dit_config.out_channels != self.vae_config.z_dim:
            raise ValueError(
                f"Incompatible DiT/VAE latent channels: dit_config.out_channels={self.dit_config.out_channels}, vae_config.z_dim={self.vae_config.z_dim}"
            )

        # After latent dims computed, kv dims must also be integer
        latent_h = self.height // sf_sp
        latent_w = self.width // sf_sp

        ph = ps[1]
        pw = ps[2]
        if ph is None or ph <= 0 or pw is None or pw <= 0:
            raise ValueError(f"Invalid patch sizes: patch_h={ph}, patch_w={pw}, patch_size={ps}")

        if latent_h % ph != 0:
            raise ValueError(
                f"latent_height ({latent_h}) must be divisible by patch_size[1] ({ph})"
            )
        if latent_w % pw != 0:
            raise ValueError(
                f"latent_width ({latent_w}) must be divisible by patch_size[2] ({pw})"
            )

        # Temporal scale should be valid if we compute max_num_actions
        sf_t = self.vae_config.scale_factor_temporal
        if sf_t is None or sf_t <= 0:
            raise ValueError(f"Invalid vae_config.scale_factor_temporal: {sf_t}")

    def _maybe_set(self, field_name: str, value: int) -> None:
        """
        If recompute_derived is True, always set.
        Otherwise, only set if current value is None.
        """
        if self.recompute_derived or getattr(self, field_name) is None:
            setattr(self, field_name, value)

    def _compute_derived(self) -> None:
        """
        Compute:
          max_num_actions
          latent_height/latent_width/latent_spatial
          kv_height/kv_width/kv_spatial
        """
        sf_sp = self.vae_config.scale_factor_spatial
        sf_t = self.vae_config.scale_factor_temporal
        ps = self.dit_config.patch_size

        max_num_actions = (self.max_num_frames - 1) // sf_t + 1
        latent_h = self.height // sf_sp
        latent_w = self.width // sf_sp
        kv_h = latent_h // ps[1]
        kv_w = latent_w // ps[2]
        latent_spatial = latent_h * latent_w
        kv_spatial = kv_h * kv_w

        self._maybe_set("max_num_actions", max_num_actions)
        self._maybe_set("latent_height", latent_h)
        self._maybe_set("latent_width", latent_w)
        self._maybe_set("kv_height", kv_h)
        self._maybe_set("kv_width", kv_w)
        self._maybe_set("latent_spatial", latent_spatial)
        self._maybe_set("kv_spatial", kv_spatial)

        if self.motion_token_buckets:
            b1, b2, b3 = int(self.motion_token_buckets[0]), int(self.motion_token_buckets[1]), int(self.motion_token_buckets[2])
            if self.motion_prefix_frames is None:
                self._maybe_set("motion_prefix_frames", int(b1 + b2 + b3))
            if self.motion_prefix_latent_frames is None:
                self._maybe_set("motion_prefix_latent_frames", (self.motion_prefix_frames + sf_t - 1) // sf_t)
            if self.kv_cond_tokens is None:
                ref_tokens = kv_h * kv_w
                motion_tokens = b1 * kv_h * kv_w + (b2 // 2) * (kv_h // 2) * (kv_w // 2) + (b3 // 4) * (kv_h // 4) * (kv_w // 4)
                self._maybe_set("kv_cond_tokens", int(ref_tokens + motion_tokens))

    # ============================================================
    # YAML loading utilities
    # ============================================================

    @classmethod
    def from_yaml(
        cls,
        yaml_input: Union[str, os.PathLike],
        *,
        is_path: bool = True,
    ) -> "RTConfig":
        """
        Create RTConfig from a YAML file or YAML string.
        """
        if is_path:
            with open(yaml_input, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(yaml_input))

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping (dict), got: {type(data)}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RTConfig":
        """
        Create RTConfig from dictionary.

        Unknown fields in YAML will be ignored.
        """
        allowed_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        return cls(**filtered_data)
