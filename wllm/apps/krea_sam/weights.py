"""Checkpoint manifest for Krea-Realtime + SAM3.

SAM3 itself resolves by HuggingFace repo id into the HF cache on first
launch and needs no entry here.
"""

from wllm.serving.weights.components import (
    Component,
    WAN_TEXT_ENCODER,
    WAN_TOKENIZER,
    WAN_VAE_21,
)

KREA_TRANSFORMER = Component(
    target="krea-realtime-video",
    repo="krea/krea-realtime-video",
    patterns=("transformer/*",),
    note="Krea-Realtime 14B causal DiT (the repo's duplicate single-file "
         "checkpoint and demo videos are skipped)",
)

COMPONENTS = [WAN_TOKENIZER, WAN_TEXT_ENCODER, WAN_VAE_21, KREA_TRANSFORMER]
