"""Config for the streaming/pipelined Qwen3-Omni worker.

Extends the reference config's fields with the agent's optimization knobs:
pipeline scheduling mode, per-stage device placement + tensor parallelism,
and the code2wav streaming chunk schedule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class StreamingConfig:
    seed: int = 42

    # -------- IPC buffers (adapter contract) --------
    ctrl_buffer_name: str = "qwen3_omni_ctrl"
    text_input_buffer_name: str = "qwen3_omni_text_input"
    audio_output_buffer_name: str = "qwen3_omni_audio_output"
    audio_meta_buffer_name: str = "qwen3_omni_audio_meta"

    # -------- audio buffer sizing --------
    audio_sample_rate: int = 24000
    audio_frame_samples: int = 960
    audio_max_chunks: int = 8192
    text_max_pending: int = 16
    audio_samples_per_codec_frame: int = 1920

    # -------- conditioning --------
    speaker: str = "chelsie"
    language: str = "en"
    system_prompt: str = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech.")

    # -------- model paths / stage configs --------
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    thinker_stage_configs_path: str = "wllm/configs/qwen3_omni/thinker_only.yaml"
    code2wav_stage_configs_path: str = "wllm/configs/qwen3_omni/code2wav_only.yaml"

    # -------- OPTIMIZATION KNOBS --------
    # pipeline scheduling: how much the 3 stages overlap.
    #   "sequential"          : reference-like (thinker->talker->c2w, whole).
    #   "stream_talker_c2w"   : talker->c2w chunked/overlapped only.
    #   "stream_thinker_talker": thinker->talker overlapped, c2w whole.
    #   "full_stream"         : all three overlapped.
    pipeline_mode: str = "full_stream"

    # device placement (physical GPU indices) + tensor parallelism
    thinker_gpu: int = 0
    thinker_tp: int = 1
    talker_gpu: int = 1
    talker_tp: int = 1
    c2w_gpu: int = 1
    c2w_tp: int = 1

    # code2wav streaming chunk schedule (frames). Native Qwen3-Omni uses
    # 25 + 25 left context. first_chunk_frames lets the FIRST chunk be
    # smaller for lower first-audio latency.
    codec_chunk_frames: int = 25
    codec_left_context_frames: int = 25
    first_chunk_frames: int = 25

    # sampling
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

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "StreamingConfig":
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        fields = {f_.name for f_ in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in data.items() if k in fields}
        extra = {k: v for k, v in data.items() if k not in fields}
        cfg = cls(**known)
        cfg.extra = extra
        return cfg
