import torch
from wllm.serving.rt_config import RTConfig
from abc import ABC, abstractmethod


class BaseKVCache(ABC):

    def __init__(self, cfg: RTConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
    @abstractmethod
    def alloc(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_kv(
        self, layer_id: int, offset: int
    ):  
        raise NotImplementedError

    @abstractmethod
    def set_encoder_kv(self,
        layer_id: int,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor
    ):
        
        raise NotImplementedError

    @abstractmethod
    def get_encoder_kv(
        self, layer_id: int,
    ):
        raise NotImplementedError

    
    
