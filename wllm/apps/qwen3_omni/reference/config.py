from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class Qwen3OmniSamplingConfig:
    """Sampling overrides for the three Qwen3-Omni components."""

    thinker_max_tokens: int = 2048
    thinker_temperature: float = 0.4
    thinker_top_p: float = 0.9
    thinker_top_k: int = 1
    thinker_repetition_penalty: float = 1.05

    talker_max_tokens: int = 4096
    talker_temperature: float = 0.9
    talker_top_k: int = 50
    talker_top_p: float = 1.0
    talker_repetition_penalty: float = 1.05
    talker_max_seq_len: int = 8192

    code2wav_max_tokens: int = 65536
    code2wav_temperature: float = 0.0
    code2wav_top_p: float = 1.0
    code2wav_repetition_penalty: float = 1.1


@dataclass
class Qwen3OmniThinkerConfig:
    model_path: str
    stage_configs_path: str
    gpu_index: Optional[int] = None
    extra_engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Qwen3OmniTalkerConfig:
    model_path: str
    gpu_index: Optional[int] = None


@dataclass
class Qwen3OmniCode2WavConfig:
    model_path: str
    stage_configs_path: str
    gpu_index: Optional[int] = None
    extra_engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Qwen3OmniReferenceConfig:
    """User-facing config for the simple sequential Qwen3-Omni backend."""

    seed: int = 42

    # -------- IPC buffers --------
    ctrl_buffer_name: str = "qwen3_omni_ctrl"
    text_input_buffer_name: str = "qwen3_omni_text_input"
    audio_output_buffer_name: str = "qwen3_omni_audio_output"
    audio_meta_buffer_name: str = "qwen3_omni_audio_meta"

    # -------- audio buffer sizing --------
    audio_sample_rate: int = 24000
    audio_frame_samples: int = 960
    audio_max_chunks: int = 8192
    text_max_pending: int = 16

    # -------- conditioning --------
    speaker: str = "chelsie"
    language: str = "en"
    system_prompt: str = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    # -------- per-component config --------
    thinker: Qwen3OmniThinkerConfig = None  # filled in __post_init__
    talker: Qwen3OmniTalkerConfig = None
    code2wav: Qwen3OmniCode2WavConfig = None
    sampling: Qwen3OmniSamplingConfig = field(default_factory=Qwen3OmniSamplingConfig)

    # -------- vocoder upsample factor --------
    # total_upsample for Qwen3-Omni = prod([8,5,4,3,2,2]) = 1920
    # (codec frame rate = 24000/1920 = 12.5 Hz). Used as a sanity-check
    # on Code2Wav output length.
    audio_samples_per_codec_frame: int = 1920

    @classmethod
    def from_yaml(
        cls,
        yaml_input: Union[str, os.PathLike],
        *,
        is_path: bool = True,
    ) -> "Qwen3OmniReferenceConfig":
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
    def from_dict(cls, data: Dict[str, Any]) -> "Qwen3OmniReferenceConfig":
        thinker_data = data.get("thinker") or {}
        talker_data = data.get("talker") or {}
        code2wav_data = data.get("code2wav") or {}
        sampling_data = data.get("sampling") or {}

        if "model_path" not in thinker_data:
            raise ValueError("config.thinker.model_path is required")
        if "stage_configs_path" not in thinker_data:
            raise ValueError("config.thinker.stage_configs_path is required")
        if "model_path" not in talker_data:
            raise ValueError("config.talker.model_path is required")
        if "model_path" not in code2wav_data:
            raise ValueError("config.code2wav.model_path is required")
        if "stage_configs_path" not in code2wav_data:
            raise ValueError("config.code2wav.stage_configs_path is required")

        allowed_top = {
            f.name for f in cls.__dataclass_fields__.values()
        } - {"thinker", "talker", "code2wav", "sampling"}
        top = {k: v for k, v in data.items() if k in allowed_top}

        return cls(
            **top,
            thinker=Qwen3OmniThinkerConfig(
                model_path=str(thinker_data["model_path"]),
                stage_configs_path=str(thinker_data["stage_configs_path"]),
                gpu_index=thinker_data.get("gpu_index"),
                extra_engine_kwargs=thinker_data.get("extra_engine_kwargs") or {},
            ),
            talker=Qwen3OmniTalkerConfig(
                model_path=str(talker_data["model_path"]),
                gpu_index=talker_data.get("gpu_index"),
            ),
            code2wav=Qwen3OmniCode2WavConfig(
                model_path=str(code2wav_data["model_path"]),
                stage_configs_path=str(code2wav_data["stage_configs_path"]),
                gpu_index=code2wav_data.get("gpu_index"),
                extra_engine_kwargs=code2wav_data.get("extra_engine_kwargs") or {},
            ),
            sampling=Qwen3OmniSamplingConfig(**{
                k: v for k, v in sampling_data.items()
                if k in Qwen3OmniSamplingConfig.__dataclass_fields__
            }),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
