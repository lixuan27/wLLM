"""Checkpoint conversion for models whose official release is a
training-state pickle.

Both converted models (WorldPlay-5B and LongLive-2.0-5B) ship as
``{"generator": {"model.<name>": tensor, ...}, ...}`` pickles; their own
reference code selects the ``generator`` subtree at load. The conversion
does the same selection once, strips the ``model.`` prefix, optionally
casts (WorldPlay is fp32; the artifact everything runs on is bf16), and
writes a plain diffusers-style safetensors + config.json.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

_SUBTREE_KEYS = ("generator", "model", "state_dict")
_PREFIX = "model."


def generator_pt_to_safetensors(
    pt_path: str, out_dir: str, cast: str | None, config: dict
) -> None:
    state = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(state, dict) and not all(torch.is_tensor(v) for v in state.values()):
        for key in _SUBTREE_KEYS:
            if key in state:
                state = state[key]
                break
        else:
            raise ValueError(
                f"{pt_path}: no tensor subtree under any of {_SUBTREE_KEYS}"
            )

    dtype = getattr(torch, cast) if cast else None
    out = {}
    for name, tensor in state.items():
        if name.startswith(_PREFIX):
            name = name[len(_PREFIX):]
        tensor = tensor.contiguous()
        out[name] = tensor.to(dtype) if dtype is not None else tensor.clone()

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
