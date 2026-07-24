from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from wllm.serving.rt_config import RTConfig


@dataclass
class LongLiveReferenceConfig:
    # -------- model selection --------
    model_name: Optional[str] = None
    text_encoder_name: Optional[str] = None
    vae_name: Optional[str] = None

    # -------- model paths --------
    tokenizer_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    transformer_path: Optional[str] = None
    vae_path: Optional[str] = None

    # -------- generation knobs --------
    seed: int = 42
    prompt: str = ""
    negative_prompt: str = ""
    max_sequence_length: int = 512
    max_num_frames: int = 1537
    height: int = 704
    width: int = 1280
    num_videos_per_prompt: int = 1
    num_inference_steps: int = 4
    first_chunk_size: int = 8
    chunk_size: int = 8
    context_window_size: int = 24
    guidance_scale: float = 1.0
    timestep_shift: float = 5.0
    stabilization_level: int = 15

    # -------- LongLive sliding-window KV cache + multi-shot --------
    sink_size: int = 8
    multi_shot_rope_offset: int = 8
    scene_cut_prefix: str = "The scene transitions. "
    kv_cond_tokens: int = 0

    # -------- runtime controls --------
    attention_backend: str = "auto"
    kv_memory: str = "preallocated"
    use_cuda_graph: bool = False
    capture_shapes: Optional[List[Tuple[int, int, bool]]] = None

    # -------- IPC buffers --------
    video_buffer_name: Optional[str] = None
    ctrl_buffer_name: Optional[str] = None
    audio_buffer_name: Optional[str] = None
    signal_buffer_name: Optional[str] = None
    audio_frame_samples: int = 320
    audio_sample_rate: int = 16000
    audio_max_chunks: int = 8192

    # -------- ASR (prompt updates from the user's microphone) --------
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B"

    # -------- LoRA (LongLive 2.0 ships BF16 only -- no LoRA) --------
    lora_path: Optional[str] = None
    lora_rank: int = 128
    lora_alpha: float = 64.0
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    # -------- device --------
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
    ) -> "LongLiveReferenceConfig":
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
    def from_dict(cls, data: Dict[str, Any]) -> "LongLiveReferenceConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
