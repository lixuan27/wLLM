from wllm.serving.rt_config import RTConfig
from wllm.serving.models.vae.wan_vae import AutoencoderKLWan
from wllm.serving.loader.safetensor_loader import load_from_safetensors
from wllm.serving.utils.video_process import latent2rgb
from wllm.serving.models.loader.component_loader import DitLoader
import torch

class VAERunner:
    def __init__(self, cfg: RTConfig, dtype: torch.dtype, device: str):
        
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.vae: AutoencoderKLWan = None
        self._latents_mean: torch.Tensor = None
        self._latents_std: torch.Tensor = None

        self.build()
    
    def build(self):
        
        self._load_model()
        self._load_stats()
        self._compile()

    def _load_model(self):
        
        loader = DitLoader()
        self.vae = loader.load(
            self.cfg.vae_name,
            self.cfg.vae_path,
            device=self.device,
            dtype=self.dtype
        )
  
    def _load_stats(self):
        
        self._latents_mean = (
            torch.tensor(self.cfg.vae_config.latents_mean)
            .view(1, self.cfg.vae_config.z_dim, 1, 1, 1)
            .to(self.device, self.dtype)
        )
        self._latents_std = 1.0 / torch.tensor(self.cfg.vae_config.latents_std).view(
                    1, self.cfg.vae_config.z_dim, 1, 1, 1
        ).to(self.device, self.dtype)

    def _compile(self):

        self.vae.decoder = torch.nn.utils.convert_conv3d_weight_memory_format(self.vae.decoder, torch.channels_last_3d)
        self.vae.decoder = torch.nn.utils.convert_conv2d_weight_memory_format(self.vae.decoder, torch.channels_last)
        self.vae.encoder = torch.nn.utils.convert_conv3d_weight_memory_format(self.vae.encoder, torch.channels_last_3d)
        self.vae.encoder = torch.nn.utils.convert_conv2d_weight_memory_format(self.vae.encoder, torch.channels_last)

    def _run(self,
        latents: torch.Tensor,
        is_first_chunk: bool):
        
        latents = latents.to(self.dtype)
        latents  = latents  / self._latents_std + self._latents_mean
        o = self.vae.decode(
            latents,
            return_dict=False,
            is_first_chunk=is_first_chunk
            )[0]
        o = latent2rgb(o)
        return o
        

    def run(self,
        latents: torch.Tensor,
        is_first_chunk: bool):

        return self._run(latents, is_first_chunk)

    def encode(self, image: torch.Tensor, stream: bool = False):
        # stream=False starts a fresh causal encode (the first latent frame
        # is the single-frame anchor); stream=True continues the encoder's
        # temporal cache from the previous chunk, for streaming v2v. That
        # cache lives on the VAE and survives decode calls (see
        # wan_vae.clear_cache / _decode), so no separate cache is needed here.
        o = self.vae.encode(image, return_dict=False, stream=stream)[0].mode()
        return (o - self._latents_mean) * self._latents_std

    def clear(self):
        return self.vae.clear_cache()
