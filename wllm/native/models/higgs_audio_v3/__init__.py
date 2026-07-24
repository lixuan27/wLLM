"""Higgs Audio v3 TTS (discrete multi-codebook path) on RTX SM120.

The acoustic backbone is a standard dense Qwen3-4B; audio is produced by a
fused multi-codebook embedding/head over an 8-codebook neural codec at 25 Hz.
The hand-written forward path lives in
``wllm_native.frontends.torch.higgs_audio_v3_rtx``.
"""

from wllm.native.models.higgs_audio_v3.pipeline_rtx import (
    HiggsAudioV3Dims,
    HiggsAudioV3Pipeline,
)

__all__ = ["HiggsAudioV3Dims", "HiggsAudioV3Pipeline"]
