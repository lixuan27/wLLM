"""Checkpoint manifest for LiveAvatar."""

from wllm.serving.weights.components import (
    Component,
    WAN_TEXT_ENCODER,
    WAN_TOKENIZER,
    WAN_VAE_21,
)

S2V_BASE = Component(
    target="wan2.2-s2v-14b",
    repo="Wan-AI/Wan2.2-S2V-14B",
    patterns=(
        "config.json",
        "diffusion_pytorch_model*",
        "wav2vec2-large-xlsr-53-english/*",
    ),
    note="Wan2.2-S2V-14B base DiT + wav2vec2 (native layout; the T5/VAE "
         ".pth files are skipped, the shared diffusers components are used instead)",
)

LIVEAVATAR_LORA = Component(
    target="live-avatar",
    repo="Quark-Vision/Live-Avatar",
    patterns=("liveavatar.safetensors",),
    note="LiveAvatar LoRA",
)

COMPONENTS = [WAN_TOKENIZER, WAN_TEXT_ENCODER, WAN_VAE_21, S2V_BASE, LIVEAVATAR_LORA]
