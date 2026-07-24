
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from wllm.serving.runner.dit_runner import DiTRunner
    from wllm.serving.memory.base_cache import BaseKVCache
    from wllm.serving.layers.attention_backends.base_attention_backend import BaseAttentionBackend

@dataclass
class ForwardBatchInfo:

    cache_start: int = 0
    cache_end: int = 0
    rope_start: int = 0
    rope_end: int = 0
    kv_memory: BaseKVCache = None
    attention_backend: BaseAttentionBackend = None

     #Self-Forcing / CausVid
    is_cache: bool = False

    #Action Models
    viewmats: torch.Tensor = None
    Ks: torch.Tensor = None
    action: torch.Tensor = None

    # I2V condition (mask + VAE-encoded image), shape [B, C_cond, F, H, W]
    i2v_condition: torch.Tensor = None

    # S2V audio conditioning, shape [B, num_wav2vec_layers, C_audio, T_audio]
    audio_input: torch.Tensor = None

    # Pose/condition latents, shape [B, C_cond, F, H, W]
    cond_latents: torch.Tensor = None

    # condition-cache controls
    prefill_cond: bool = False
    cond_prefix_tokens: int = 0

    # reference / motion latents
    ref_latents: torch.Tensor = None
    motion_latents: torch.Tensor = None

    # Audio prefix alignment metadata for S2V
    motion_frames_raw: int = 0
    motion_frames_latent: int = 0

    @classmethod
    def init_new_dit(
        cls,
        cache_start: int,
        cache_end: int,
        rope_start: int,
        rope_end: int,
        model_runner: DiTRunner,
        **kwargs
    ) -> "ForwardBatchInfo":

        ret = cls(
            cache_start=int(cache_start),
            cache_end=int(cache_end),
            rope_start=int(rope_start),
            rope_end=int(rope_end),
            kv_memory=model_runner.kv_memory,
            attention_backend=model_runner.attention_backend,
        )
        ret.is_cache = kwargs.get("is_cache")
        Ks = kwargs.get("Ks")
        viewmats = kwargs.get("viewmats")
        action = kwargs.get("action")
        ret.Ks = Ks.to(model_runner.dtype) if Ks is not None else None
        ret.viewmats = viewmats.to(model_runner.dtype) if viewmats is not None else None
        ret.action = action.to(model_runner.dtype) if action is not None else None
        i2v_condition = kwargs.get("i2v_condition")
        ret.i2v_condition = i2v_condition.to(model_runner.dtype) if i2v_condition is not None else None
        audio_input = kwargs.get("audio_input")
        ret.audio_input = audio_input.to(model_runner.dtype) if audio_input is not None else None
        cond_latents = kwargs.get("cond_latents")
        ret.cond_latents = cond_latents.to(model_runner.dtype) if cond_latents is not None else None
        ret.prefill_cond = bool(kwargs.get("prefill_cond", False))
        ret.cond_prefix_tokens = int(kwargs.get("cond_prefix_tokens", 0) or 0)
        ref_latents = kwargs.get("ref_latents")
        ret.ref_latents = ref_latents.to(model_runner.dtype) if ref_latents is not None else None
        motion_latents = kwargs.get("motion_latents")
        ret.motion_latents = motion_latents.to(model_runner.dtype) if motion_latents is not None else None
        ret.motion_frames_raw = int(kwargs.get("motion_frames_raw", 0) or 0)
        ret.motion_frames_latent = int(kwargs.get("motion_frames_latent", 0) or 0)
        return ret
