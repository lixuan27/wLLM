"""Checkpoint manifest for WorldPlay.

The DiT's official release (tencent/HY-WorldPlay) is a 42 GB fp32
training pickle whose ``generator`` entry is the distilled autoregressive
model; the same subtree their own inference code loads is converted once
to a 10 GB bf16 safetensors on download.
"""

from wllm.serving.weights.components import (
    Component,
    WAN_TEXT_ENCODER,
    WAN_TOKENIZER,
    WAN_VAE_22,
    generator_pt_convert,
    wan_5b_ar_config,
)

WORLDPLAY_DIT = Component(
    target="worldplay-5b",
    repo="tencent/HY-WorldPlay",
    patterns=("wan_distilled_model/model.pt",),
    convert=generator_pt_convert(
        "wan_distilled_model/model.pt",
        "bfloat16",
        wan_5b_ar_config("WanTransformer3DModel"),
    ),
    note="WorldPlay-5B AR DiT (converted from the official 42 GB training pickle)",
)

COMPONENTS = [WAN_TOKENIZER, WAN_TEXT_ENCODER, WAN_VAE_22, WORLDPLAY_DIT]
