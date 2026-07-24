from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import yaml

from wllm.serving.rt_config import RTConfig


@dataclass
class LiveAvatarReferenceConfig:
    """User-facing config for the simple sequential LiveAvatar backend."""

    model_name: Optional[str] = None
    text_encoder_name: Optional[str] = None
    vae_name: Optional[str] = None

    seed: int = 42
    prompt: str = ""
    negative_prompt: str = ""
    image_path: Optional[str] = None

    tokenizer_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    transformer_path: Optional[str] = None
    vae_path: Optional[str] = None
    wav2vec2_path: Optional[str] = None

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

    audio_frame_samples: int = 960
    audio_max_chunks: int = 4096
    audio_sample_rate: int = 48000
    audio_stretch_ratio: float = 1.0

    motion_prefix_frames: Optional[int] = None
    motion_prefix_latent_frames: Optional[int] = None

    attention_backend: str = "auto"
    kv_memory: str = "preallocated"
    use_cuda_graph: bool = False
    capture_shapes: Optional[List[Tuple[int, int, bool]]] = None

    video_buffer_name: Optional[str] = None
    ctrl_buffer_name: Optional[str] = None
    action_buffer_name: Optional[str] = None
    audio_buffer_name: Optional[str] = None
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

    lora_path: Optional[str] = None
    lora_rank: int = 128
    lora_alpha: float = 64.0
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    device: str = "cuda"
    dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"

    def to_runtime_config(self) -> RTConfig:
        return RTConfig.from_dict(asdict(self))

    @classmethod
    def from_yaml(
        cls,
        yaml_input: Union[str, os.PathLike],
        *,
        is_path: bool = True,
    ) -> "LiveAvatarReferenceConfig":
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
    def from_dict(cls, data: Dict[str, Any]) -> "LiveAvatarReferenceConfig":
        allowed_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        return cls(**filtered_data)
