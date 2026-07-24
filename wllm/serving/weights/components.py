"""Checkpoint component descriptors.

A component is one directory under ``checkpoints/`` plus the recipe to
produce it from an official source: a HuggingFace repo, the file patterns
to fetch, and (for the two models whose official release is a training
pickle) a conversion step. Components shared by more than one app live
here; app-specific components live in the app's own
``wllm/apps/<app>/weights.py`` manifest and get promoted here once a
second app shares them. The download engine dedups by ``target``, so two
manifests naming the same target must carry identical descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Component:
    target: str
    """Directory under checkpoints/ this component installs into."""

    repo: str
    """Official HuggingFace repo id."""

    patterns: tuple
    """allow_patterns passed to snapshot_download."""

    rename_from: Optional[str] = None
    """Subdirectory of the snapshot to move to target (None = whole snapshot)."""

    convert: Optional[tuple] = None
    """Conversion step as a hashable spec:
    ("generator_pt", <source file in snapshot>, <cast dtype or "">,
     <config.json content as a sorted (key, value) tuple>)."""

    note: str = ""


def _cfg_items(d: dict) -> tuple:
    import json

    return tuple(sorted((k, json.dumps(v)) for k, v in d.items()))


def wan_5b_ar_config(class_name: str) -> dict:
    """Transformer config for the Wan2.2-5B-based autoregressive DiTs
    (WorldPlay-5B and LongLive-2.0-5B share the architecture). Field values
    from the official tencent/HY-WorldPlay wan_transformer/config.json; the
    loader ignores the _class_name/_diffusers_version metadata."""
    return {
        "_class_name": class_name,
        "_diffusers_version": "0.35.0",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 48,
        "num_attention_heads": 24,
        "num_layers": 30,
        "out_channels": 48,
        "patch_size": [1, 2, 2],
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    }


def generator_pt_convert(source_file: str, cast: str, config: dict) -> tuple:
    return ("generator_pt", source_file, cast, _cfg_items(config))


# ---------------------------------------------------------------------------
# Components shared across apps (the Wan family).
# ---------------------------------------------------------------------------

WAN_TOKENIZER = Component(
    target="wan/tokenizer",
    repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    patterns=("tokenizer/*",),
    rename_from="tokenizer",
    note="UMT5 tokenizer, shared by every Wan-based app",
)

WAN_TEXT_ENCODER = Component(
    target="wan/text_encoder",
    repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    patterns=("text_encoder/*",),
    rename_from="text_encoder",
    note="UMT5-XXL text encoder, shared by every Wan-based app",
)

WAN_VAE_21 = Component(
    target="wan/vae-2.1",
    repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    patterns=("vae/*",),
    rename_from="vae",
    note="Wan2.1 VAE (fp32; the loader casts at load), used by the 14B-based apps",
)

WAN_VAE_22 = Component(
    target="wan/vae-2.2",
    repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    patterns=("vae/*",),
    rename_from="vae",
    note="Wan2.2 VAE (fp32; the loader casts at load), used by the 5B-based apps",
)
