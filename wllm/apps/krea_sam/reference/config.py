from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from wllm.serving.rt_config import RTConfig


@dataclass
class KreaSAMReferenceConfig:
    """User-facing config for the simple sequential Krea-Realtime + SAM 3 backend."""

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
    max_num_frames: int = 4096
    height: int = 480
    width: int = 832
    num_videos_per_prompt: int = 1
    num_inference_steps: int = 4
    first_chunk_size: int = 3
    chunk_size: int = 3
    context_window_size: int = 3
    kv_cond_tokens: int = 0
    continuous_decoding: bool = True
    reactive_decoding: bool = False
    stabilization_level: int = 15
    guidance_scale: float = 1.0

    # -------- Krea-Realtime denoising / context --------
    denoising_strength: float = 0.8
    timestep_shift: float = 5.0
    keep_first_frame: bool = True

    # -------- runtime controls --------
    attention_backend: str = "auto"
    kv_memory: str = "preallocated"
    use_cuda_graph: bool = False
    capture_shapes: Optional[List[Tuple[int, int, bool]]] = None

    # -------- IPC buffers --------
    video_buffer_name: Optional[str] = None
    video_input_buffer_name: Optional[str] = None
    ctrl_buffer_name: Optional[str] = None
    video_input_max_frames: int = 512
    video_fps: int = 30

    # -------- SAM 3 (background-only style transfer) --------
    sam_text_prompt: str = "person"
    sam_min_score: float = 0.5
    sam_mask_threshold: float = 0.5
    sam_dilate_pixels: int = 3
    sam_disable: bool = False

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
    ) -> "KreaSAMReferenceConfig":
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
    def from_dict(cls, data: Dict[str, Any]) -> "KreaSAMReferenceConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
