from abc import ABC, abstractmethod
import torch

class BaseAttentionBackend(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        **kwargs):

        raise NotImplementedError
    