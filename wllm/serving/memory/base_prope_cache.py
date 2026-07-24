import torch
from wllm.serving.rt_config import RTConfig
from abc import abstractmethod
from wllm.serving.memory.base_cache import BaseKVCache

class BasePropeKVCache(BaseKVCache):

    def __init__(self, cfg: RTConfig, deivce: torch.device):
        super().__init__(cfg, deivce)
         
    @abstractmethod
    def get_rope_k_ptr(self, layer_id: int):

        raise NotImplementedError
    
    @abstractmethod
    def get_rope_v_ptr(self, layer_id: int):

        raise NotImplementedError
    
    @abstractmethod
    def get_prope_k_ptr(self, layer_id: int):

        raise NotImplementedError
    
    @abstractmethod
    def get_prope_v_ptr(self, layer_id: int):

        raise NotImplementedError
