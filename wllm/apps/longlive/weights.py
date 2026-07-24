"""Checkpoint manifest for LongLive.

The official release (Efficient-Large-Model/LongLive-2.0-5B) is a bf16
training pickle; its ``generator`` subtree — the same one the official
inference code loads — is converted once to a plain safetensors on
download.
"""

from wllm.serving.weights.components import (
    Component,
    WAN_TEXT_ENCODER,
    WAN_TOKENIZER,
    WAN_VAE_22,
    generator_pt_convert,
    wan_5b_ar_config,
)

LONGLIVE_DIT = Component(
    target="longlive-2.0-5b",
    repo="Efficient-Large-Model/LongLive-2.0-5B",
    patterns=("model_bf16.pt",),
    convert=generator_pt_convert(
        "model_bf16.pt",
        "",
        wan_5b_ar_config("LongLiveTransformer3DModel"),
    ),
    note="LongLive-2.0-5B DiT (converted from the official training pickle)",
)

COMPONENTS = [WAN_TOKENIZER, WAN_TEXT_ENCODER, WAN_VAE_22, LONGLIVE_DIT]
