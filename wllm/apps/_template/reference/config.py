from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import yaml

from wllm.serving.rt_config import RTConfig


@dataclass
class AppReferenceConfig:  # TODO rename to <App>ReferenceConfig
    """User-facing config for the sequential <app> reference backend.

    List every field the app's config.yaml can carry; unknown yaml keys are
    dropped. ``to_runtime_config`` hands the shared runtime an RTConfig.
    """

    # TODO model paths / generation settings for your app, e.g.:
    # transformer_path: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # IPC buffers
    ctrl_buffer_name: Optional[str] = None
    # TODO one name field per input/output buffer

    def to_runtime_config(self) -> RTConfig:
        return RTConfig.from_dict(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_input: Union[str, os.PathLike], *, is_path: bool = True) -> "AppReferenceConfig":
        if is_path:
            with open(yaml_input, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(yaml_input))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping (dict), got: {type(data)}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppReferenceConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})
