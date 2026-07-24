import dataclasses
import glob
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from copy import deepcopy
from typing import Any, cast

from huggingface_hub.dataclasses import strict
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from safetensors.torch import load_file as safetensors_load_file
from torch.distributed import init_device_mesh

from wllm.serving.distributed import get_local_torch_device

from wllm.serving.layers.quantization import get_quantization_config
from wllm.serving.logger import init_logger
from wllm.serving.models.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
    map_param_name,
)
from wllm.serving.hf_transformer_utils import get_diffusers_config
from wllm.serving.models.registry import ModelRegistry
from wllm.serving.utils.utils import PRECISION_TO_TYPE
from wllm.serving.models.loader.utils import set_default_torch_dtype

logger = init_logger(__name__)

ignore_diffuser_config_fields = [
    "_name_or_path",
    "__name__",
    "transformers_version",
    "_transformers_version",
    "_diffusers_version",
    "model_type",
    "tokenizer_class",
    "torch_dtype",
    "_class_name"
]

class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    def __init__(self, device=None) -> None:
        self.device = device

    @abstractmethod
    def load(self, model_path: str,         
        model_name: str,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16):
        """
        Load the component based on the model path, architecture, and inference args.

        Args:
            model_path: Path to the component model
            fastvideo_args: FastVideoArgs

        Returns:
            The loaded component
        """
        raise NotImplementedError

    @classmethod
    def for_module_type(
        cls, module_type: str, transformers_or_diffusers: str
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific module type.

        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers

        Returns:
            A component loader for the specified module type
        """
        # Map of module types to their loader classes and expected library
        module_loaders = {
            "transformer": (DitLoader, "diffusers"),
        }

        # For unknown module types, use a generic loader
        logger.warning(
            "No specific loader found for module type: %s. Using generic loader.",
            module_type,
        )
        return GenericComponentLoader(transformers_or_diffusers)


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library

    def load(self, model_path: str):
        """Load a generic component based on the model path, and inference args."""
        logger.warning(
            "Using generic loader for %s with library %s",
            model_path,
            self.library,
        )

        if self.library == "transformers":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path
            )
            logger.info(
                "Loaded generic transformers model: %s",
                model.__class__.__name__,
            )
            return model
        elif self.library == "diffusers":
            logger.warning(
                "Generic loading for diffusers components is not fully implemented"
            )

            model_config = get_diffusers_config(model=model_path)
            logger.info("Diffusers Model config: %s", model_config)
            # This is a placeholder - in a real implementation, you'd need to handle this properly
            return None
        else:
            raise ValueError(f"Unsupported library: {self.library}")
        

class DitLoader(ComponentLoader):
    """Loader for DiT models."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def _prepare_weights(
        self,
        model_name_or_path: str,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files
            )

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files, to_cpu=False
            )
        else:
            weights_iterator = pt_weights_iterator(
                hf_weights_files, to_cpu=False
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return (
            (source.prefix + name, tensor)
            for (name, tensor) in weights_iterator
        )

    def _get_all_weights(
        self,
        model: nn.Module,
        model_path: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DitLoader.Source(
            model_path,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(
                model, "allow_patterns_overrides", None
            ),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DitLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    @staticmethod
    def _merge_lora(
        model: nn.Module,
        model_config,
        lora_path: str,
        lora_rank: int,
        lora_alpha: float,
    ) -> None:
        lora_state = safetensors_load_file(lora_path)
        logger.info("Loaded %d LoRA tensors from %s", len(lora_state), lora_path)

        lora_mapping = getattr(model_config.arch_config, "lora_param_names_mapping", {})
        params_dict = dict(model.named_parameters())
        scale = lora_alpha / lora_rank
        merged_count = 0

        # Group by base module: strip .lora_A.*.weight / .lora_B.*.weight
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}
        for key, tensor in lora_state.items():
            if ".lora_A." in key:
                base = key.split(".lora_A.")[0]
                lora_a[base] = tensor
            elif ".lora_B." in key:
                base = key.split(".lora_B.")[0]
                lora_b[base] = tensor

        for base_name in lora_a:
            if base_name not in lora_b:
                logger.warning("LoRA B missing for %s, skipping", base_name)
                continue

            A = lora_a[base_name]  # (r, in_features)
            B = lora_b[base_name]  # (out_features, r)

            # Remap Wan-style name → wLLM name
            param_name = map_param_name(base_name + ".weight", lora_mapping)
            if param_name not in params_dict:
                # Try without the mapping
                fallback = base_name + ".weight"
                if fallback in params_dict:
                    param_name = fallback
                else:
                    logger.warning(
                        "LoRA target %s (mapped: %s) not found in model, skipping",
                        base_name, param_name,
                    )
                    continue

            param = params_dict[param_name]
            delta = (B.to(param.dtype).to(param.device)
                     @ A.to(param.dtype).to(param.device)) * scale
            with torch.no_grad():
                param.add_(delta)
            merged_count += 1

        logger.info("Merged %d LoRA pairs (scale=%.4f)", merged_count, scale)

    @staticmethod
    def _coerce_legacy_arch_keys(
        raw: dict[str, Any], arch_fields: set[str]
    ) -> tuple[dict[str, Any], dict[str, str]]:

        normalized = dict(raw)
        renamed: dict[str, str] = {}

        alias_pairs = {
            "num_attention_heads": "num_heads",
            "in_channels": "in_dim",
            "out_channels": "out_dim",
        }

        for canonical, legacy in alias_pairs.items():
            if (
                canonical in arch_fields
                and canonical not in normalized
                and legacy in normalized
            ):
                normalized[canonical] = normalized[legacy]
                renamed[legacy] = canonical

        if "attention_head_dim" in arch_fields and "attention_head_dim" not in normalized:
            dim = normalized.get("dim")
            num_heads = normalized.get(
                "num_attention_heads", normalized.get("num_heads")
            )
            if dim is not None and num_heads is not None:
                dim_i = int(dim)
                heads_i = int(num_heads)
                assert dim_i % heads_i == 0
                normalized["attention_head_dim"] = dim_i // heads_i
                renamed["dim+num_heads"] = "attention_head_dim"

        return normalized, renamed

    def load(self,
        model_name: str,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        lora_path: str | None = None,
        lora_rank: int = 128,
        lora_alpha: float = 64.0):
        """Load the text encoders based on the model path, and inference args."""
  
        raw_diffuser_config = get_diffusers_config(model_dir)

        with set_default_torch_dtype(dtype):
            with torch.device(device):
                model_cls, _ = ModelRegistry.resolve_model_cls(model_name)
                model_config = deepcopy(model_cls._default_config)
                arch_fields = set(model_config.arch_config.__dataclass_fields__.keys())

                raw_diffuser_config, coerced_aliases = self._coerce_legacy_arch_keys(
                    raw_diffuser_config, arch_fields
                )
                if coerced_aliases:
                    logger.info(
                        "Translated legacy transformer config keys for %s: %s",
                        model_name,
                        coerced_aliases,
                    )

                for field in ignore_diffuser_config_fields:
                    raw_diffuser_config.pop(field, None)

                filtered_arch_config = {
                    key: value
                    for key, value in raw_diffuser_config.items()
                    if key in arch_fields
                }
                ignored_arch_keys = sorted(
                    key for key in raw_diffuser_config.keys() if key not in arch_fields
                )
                if ignored_arch_keys:
                    logger.debug(
                        "Ignoring unsupported DiT config fields for %s: %s",
                        model_name,
                        ignored_arch_keys,
                    )
                model_config.update_model_arch(filtered_arch_config)
                model = model_cls(model_config)

        weights_to_load = {name for name, _ in model.named_parameters()}

        loaded_weights: set[str] = model.load_weights(
            self._get_all_weights(
                model, model_dir
            )
        )

        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights
            - self.counter_before_loading_weights,
        )

        weights_not_loaded = weights_to_load - loaded_weights
        if weights_not_loaded and model_config.quant_config is None:
            raise ValueError(
                "Following weights were not initialized from "
                f"checkpoint: {weights_not_loaded}"
            )

        if lora_path is not None:
            self._merge_lora(model, model_config, lora_path, lora_rank, lora_alpha)

        return model.eval()
