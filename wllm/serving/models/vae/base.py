# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, Optional
from wllm.serving.runner.forward_batch import ForwardBatchInfo
from wllm.serving.configs.models.vaes.base import VAEConfig
import torch
from torch import nn


# TODO

class BaseVAE(nn.Module, ABC):

    _default_config: VAEConfig
    
    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_default_config"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseVAE must define '{attr}' class variable"
                )

    def __init__(self,  config: VAEConfig, **kwargs) -> None:
        self.config = config
        super().__init__()
       

    @abstractmethod
    def encode(
        self, x: torch.Tensor, return_dict: bool
    ):
        pass

    @abstractmethod
    def decode(
        self, x: torch.Tensor, return_dict: bool
    ):
        pass
    