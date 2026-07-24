"""Streaming-worker config: the reference config plus streaming knobs.

The base config (Qwen3OmniReferenceConfig) is reused for buffers,
sampling, model paths, and device placement. Extra top-level YAML keys
under ``streaming:`` parameterize the schedule so one worker backs the
stream_c2w / stream_thinker_talker / stream_full variants (each a
distinct config + launch + benchmark — never sharing a measurement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import yaml

from wllm.apps.qwen3_omni.reference.config import Qwen3OmniReferenceConfig


@dataclass
class StreamingParams:
    # Pipeline schedule
    stream_thinker_talker: bool = False   # prime talker after 1st thinker token, push tokens as they stream
    stream_c2w: bool = True               # emit audio incrementally from Code2Wav

    # Code2Wav emission policy (only used when stream_c2w)
    c2w_emit_interval_frames: int = 25    # vocode + emit after this many NEW codec frames
    c2w_first_emit_frames: int = 25       # frames to accumulate before the FIRST emit
    c2w_lookahead_frames: int = 0         # hold back this many newest frames (future context) before emitting
    c2w_context_frames: int = -1          # left context: -1 => growing prefix (decode [0:end]); else decode [end-context:end]
    c2w_threaded: bool = False            # run Code2Wav vocoding on a dedicated thread+loop so it never stalls talker stepping

    # Device exposure overrides (for tensor parallelism / placement variants).
    # CSV physical-GPU strings exposed to each engine's subprocess (e.g.
    # "1,2" for thinker TP=2 on physical GPUs 1,2). None => str(gpu_index).
    thinker_visible_devices: Optional[str] = None
    thinker_stage_configs_path: Optional[str] = None
    c2w_visible_devices: Optional[str] = None
    c2w_stage_configs_path: Optional[str] = None

    # Talker build knobs. enforce_eager disables the talker's torch.compile
    # + CUDA graphs (needed when co-hosting a TP AsyncOmni thinker, whose
    # compile-cache device ids otherwise collide with the in-process talker).
    # talker_tp_size>1 selects the SPMD tensor-parallel talker (talker_tp/).
    talker_enforce_eager: bool = False
    talker_tp_size: int = 1
    talker_tp_devices: Optional[str] = None      # CSV physical GPUs for the talker TP ranks


@dataclass
class StreamingConfig:
    base: Qwen3OmniReferenceConfig
    streaming: StreamingParams

    @classmethod
    def from_yaml(cls, path: str) -> "StreamingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        base = Qwen3OmniReferenceConfig.from_dict(data)
        sdata = data.get("streaming") or {}
        params = StreamingParams(**{
            k: v for k, v in sdata.items()
            if k in StreamingParams.__dataclass_fields__
        })
        return cls(base=base, streaming=params)
