import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from wllm.serving.rt_config import RTConfig

class Wan3DRotaryPosEmbed:
    def __init__(
        self,
        cfg: RTConfig
    ):  
        
        self.cfg = cfg
        h_dim = w_dim = 2 * (self.cfg.dit_config.head_dim // 6)
        t_dim = self.cfg.dit_config.head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 
        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                self.cfg.dit_config.rope_max_seq_len,
                self.cfg.dit_config.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)
            
        self.freqs_cos = torch.cat(freqs_cos, dim=1)
        self.freqs_sin = torch.cat(freqs_sin, dim=1)
        
        p_t, p_h, p_w = self.cfg.dit_config.patch_size
        ppf, pph, ppw = self.cfg.max_num_actions // p_t, self.cfg.latent_height // p_h, self.cfg.latent_width // p_w
        
        split_sizes = [
            (self.cfg.dit_config.head_dim) - 2 * ((self.cfg.dit_config.head_dim) // 3),
            (self.cfg.dit_config.head_dim) // 3,
            (self.cfg.dit_config.head_dim) // 3,
        ]
        
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        c = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
           ppf * pph * ppw, -1
        )[..., 0::2]
        s = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            ppf * pph * ppw, -1
        )[..., 1::2]
        
        self.freq_cis = torch.cat([c, s], dim=1).contiguous()
    

        
        