"""Variant `async_asr`: run VAD + ASR on a background thread (and, via
`LL_ASR_DEVICE`, a separate GPU) so the video-generation loop never stalls for
transcription. The reference drains audio, runs ASR, and only *then* generates —
so each prompt update blocks generation for the ASR duration (a hitch in the
frame stream). The IR worker graph shows the black-box ASR stage is independent
of the exposed video-gen stage until the prompt handoff, so ASR can overlap
generation.

Attacks smoothness (no gen stall during ASR) and narration latency (the prompt
applies the instant ASR finishes rather than behind a chunk). The video-gen math
is identical to the reference, so output is bit-faithful. See `async_asr_mixin`.
"""

from __future__ import annotations

from wllm.apps.longlive.backend.rocm.async_asr_mixin import AsyncASRMixin
from wllm.apps.longlive.backend.rocm.baseline_single.worker import Worker as BaselineWorker


class Worker(AsyncASRMixin, BaselineWorker):
    pass
