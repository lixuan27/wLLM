import torch
from typing import Union, Tuple
import numpy as np
from wllm.serving.layers.custom_op import CustomOp

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


@CustomOp.register("rotary_embedding_3d")
class Wan3DRotaryPosEmbed(CustomOp):
    def __init__(
        self,
        attention_head_dim: int,
        theta: float,
        rope_max_seq_len: int,
        grid_size: Tuple[int],
        dtype: torch.dtype
    ):  
        
        self.attention_head_dim = attention_head_dim
        self.theta = theta
        self.rope_max_seq_len = rope_max_seq_len
        self.grid_size = grid_size
        cache = self._compute_cos_sin_cache()
        self.cache = cache.to(dtype)
        
    def _compute_cos_sin_cache(self) -> torch.Tensor:

        h_dim = w_dim = 2 * (self.attention_head_dim// 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 
        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                self.rope_max_seq_len,
                self.heta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)
            
        self.freqs_cos = torch.cat(freqs_cos, dim=1)
        self.freqs_sin = torch.cat(freqs_sin, dim=1)
         

        ppf, pph, ppw = self.grid_size
        
        split_sizes = [
            (self.attention_head_dim) - 2 * ((self.attention_head_dim) // 3),
            (self.attention_head_dim) // 3,
            (self.attention_head_dim) // 3,
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
        
        return torch.cat([c, s], dim=1).contiguous()
    

        
        