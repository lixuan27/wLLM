# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import field

import torch
from torch import nn

from wllm.serving.configs.models.encoders import (BaseEncoderOutput,
ImageEncoderConfig,
TextEncoderConfig)

class TextEncoder(nn.Module, ABC):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [])
    _stacked_params_mapping: list[tuple[str, str,
                                        str]] = field(default_factory=list)
    _default_config: TextEncoderConfig
    
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._fsdp_shard_conditions = config._fsdp_shard_conditions
        self._stacked_params_mapping = config.arch_config.stacked_params_mapping

    @abstractmethod
    def forward(self,
                input_ids: torch.Tensor | None,
                position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                output_hidden_states: bool | None = None,
                **kwargs) -> BaseEncoderOutput:
        pass

    
class ImageEncoder(nn.Module, ABC):
    
    _default_config: ImageEncoderConfig
    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor,
                **kwargs) -> BaseEncoderOutput:
        pass
