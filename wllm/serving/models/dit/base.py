# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, Optional
from wllm.serving.runner.forward_batch import ForwardBatchInfo
from wllm.serving.configs.models.dits.base import DiTConfig
import torch
from torch import nn


# TODO

class BaseDiT(nn.Module, ABC):

    _default_config: DiTConfig
    
    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_default_config"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self,  config: DiTConfig, **kwargs) -> None:
        self.config = config
        super().__init__()
       

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                timestep: torch.LongTensor,
                rotary_emb: torch.Tensor,
                forward_batch_info: ForwardBatchInfo,
                **kwargs
                ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def fill_encoder_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        forward_batch_info: ForwardBatchInfo,
        **kwargs
    ) -> None:
        pass

    def __post_init__(self) -> None:
        required_attrs = [
            "hidden_size", "num_attention_heads", "num_channels_latents"
        ]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )
