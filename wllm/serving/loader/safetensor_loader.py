import os
import json
import glob
import inspect
from typing import cast
import torch
import torch.nn as nn
from collections.abc import Generator, Iterable

from safetensors.torch import load_file
from wllm.serving.logger import init_logger
from wllm.serving.hf_transformer_utils import get_diffusers_config
from wllm.serving.models.dit.base import BaseDiT
from wllm.serving.models.registry import ModelRegistry
from wllm.serving.models.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from wllm.serving.models.loader.component_loader import DitLoader

logger = init_logger(__name__)

def _filter_init_kwargs(model_class, init_kwargs: dict):
    """
    Filter out unsupported keyword arguments for model_class.__init__.

    If model_class.__init__ has **kwargs, no filtering is needed.
    """
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters

    # If **kwargs exists, accept all keys
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return init_kwargs, []

    valid = set(params.keys())
    valid.discard("self")

    filtered = {}
    ignored = []
    for k, v in init_kwargs.items():
        if k in valid:
            filtered[k] = v
        else:
            ignored.append(k)
    return filtered, ignored


def _load_state_dict_auto(model_dir: str, index_filename: str | None = None, weights_filename: str | None = None):
    """
    Load state_dict from:
      - sharded safetensors (index.json present), OR
      - a single .safetensors file (index.json absent)

    Returns:
        state_dict (dict[str, Tensor])
    """
    # 1) Try index.json (sharded)
    if index_filename is None:
        # common names in diffusers/transformers
        candidates = [
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors.index.json",
            "pytorch_model.safetensors.index.json",
        ]
        for c in candidates:
            p = os.path.join(model_dir, c)
            if os.path.exists(p):
                index_filename = c
                break

    if index_filename is not None:
        index_path = os.path.join(model_dir, index_filename)
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", None)
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid index file (missing weight_map): {index_path}")

        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            logger.info(f"Loading shard: {shard_file}")
            shard_sd = load_file(shard_path, device="cpu")
            state_dict.update(shard_sd)
        return state_dict

    # 2) Fallback: single .safetensors
    if weights_filename is not None:
        weights_path = os.path.join(model_dir, weights_filename)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights file not found: {weights_path}")
        logger.info(f"Loading weights: {weights_filename}")
        return load_file(weights_path, device="cpu")

    # Auto-detect a single .safetensors (ignore sharded parts if any)
    safes = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    if not safes:
        raise FileNotFoundError(f"No .safetensors found in: {model_dir}")

    # Prefer "main" filenames if multiple exist
    preferred_names = {"diffusion_pytorch_model.safetensors", "model.safetensors", "pytorch_model.safetensors"}
    preferred = [p for p in safes if os.path.basename(p) in preferred_names]
    if preferred:
        weights_path = preferred[0]
    else:
        # If only shards exist without index, pick the first file (still might work, but usually incomplete)
        # Better heuristic: prefer non-shard name
        non_shards = [p for p in safes if "-of-" not in os.path.basename(p)]
        weights_path = non_shards[0] if non_shards else safes[0]

    logger.info(f"Loading weights: {os.path.basename(weights_path)}")
    return load_file(weights_path, device="cpu")


def load_safetensors_model_from_config(
    model_class,
    model_dir: str,
    *,
    config_filename: str = "config.json",
    index_filename: str | None = None,
    weights_filename: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    strict: bool = True,
    config_transform=None,
):
    """
    Unified loader:
      - Initialize model from config.json (with automatic init kwargs filtering)
      - Load weights from sharded safetensors (index.json) OR single .safetensors
      - Move model to (device, dtype)

    Args:
        model_class: nn.Module class (not an instance)
        model_dir: directory containing config + weights
        config_filename: config json name
        index_filename: optional explicit index json name
        weights_filename: optional explicit single safetensors name
        device/dtype: target placement
        strict: load_state_dict strict
        config_transform: optional callable(dict)->dict to adapt config to init kwargs

    Returns:
        model (nn.Module)
    """

    # --- load config ---
    config_path = os.path.join(model_dir, config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    init_kwargs = config_transform(cfg) if config_transform is not None else cfg
    if not isinstance(init_kwargs, dict):
        raise TypeError("config_transform must return a dict of init kwargs.")

    init_kwargs, ignored = _filter_init_kwargs(model_class, init_kwargs)
    if ignored:
        logger.warning(f"Ignored unsupported init args: {ignored}")

    # --- init model on CPU first ---
    model = model_class(**init_kwargs)

    # --- load weights (cpu) ---
    state_dict = _load_state_dict_auto(
        model_dir=model_dir,
        index_filename=index_filename,
        weights_filename=weights_filename,
    )

    # --- load into model ---
    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing, unexpected = load_result
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:20]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]} ...")

    # --- move to target ---
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


ignore_diffuser_config_fields = [
    "_name_or_path",
    "transformers_version",
    "_transformers_version",
    "_diffusers_version",
    "model_type",
    "tokenizer_class",
    "torch_dtype",
    "_class_name"
]


def load_from_safetensors(
    model_name: str,
    model_dir: str,
    *,
    config_filename: str = "config.json",
    index_filename: str | None = None,
    weights_filename: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    strict: bool = True,
    config_transform=None,
):  
    
    raw_diffuser_config = get_diffusers_config(model_dir)
    for field in ignore_diffuser_config_fields:
        raw_diffuser_config.pop(field, None)
    
    model_cls, _ = ModelRegistry.resolve_model_cls(model_name)
    model_config = model_cls._default_config
    model_config.update_model_arch(raw_diffuser_config)
    model = model_cls(model_config)
    state_dict = _load_state_dict_auto(
        model_dir=model_dir,
        index_filename=index_filename,
        weights_filename=weights_filename,
    )
    print(f"Loaded state_dict with {len(state_dict)} keys from {model_dir}")

    # --- load into model ---
    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing, unexpected = load_result
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:20]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]} ...")

    # --- move to target ---
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
