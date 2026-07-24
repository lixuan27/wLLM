from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.activation import get_act_fn

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"gelu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        # project in
        self.net.append(ReplicatedLinear(dim, inner_dim, bias=bias))
        # activation function
        self.net.append(get_act_fn(activation_fn))
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(ReplicatedLinear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            if isinstance(module, ReplicatedLinear):
                hidden_states, _ = module(hidden_states)
            else:
                hidden_states = module(hidden_states)
        return hidden_states

