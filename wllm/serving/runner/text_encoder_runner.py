from wllm.serving.rt_config import RTConfig
from wllm.serving.models.encoders.base import TextEncoder
from wllm.serving.models.loader.component_loader import DitLoader
import torch

class TextEncoderRunner:
    def __init__(self, cfg: RTConfig, dtype: torch.dtype, device: str):
        
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.text_encoder: TextEncoder = None
        self.build()
    
    def build(self):
        
        self._load_model()

    def _load_model(self):
        
        loader = DitLoader()
        self.text_encoder = loader.load(
                self.cfg.text_encoder_name, 
                self.cfg.text_encoder_path, 
                self.device, 
                self.dtype
            )

    def _run(self,
        input_ids: torch.Tensor,
        mask: torch.Tensor):
        
        o = self.text_encoder(input_ids, mask).last_hidden_state
        return o
        

    def run(self,
        input_ids: torch.Tensor,
        mask: torch.Tensor):

        return self._run(input_ids, mask)
    

    