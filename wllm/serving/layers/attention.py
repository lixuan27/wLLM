# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Optional
import torch
from torch import nn
from wllm.serving.layers.linear import ReplicatedLinear
from wllm.serving.layers.layernorm import RMSNorm
from wllm.serving.logger import init_logger
logger = init_logger(__name__)
#from diffusers.models.normalization import RMSNorm
# from diffusers.utils import logging
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        prope: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        is_causal: bool = False,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.norm_q = RMSNorm(dim_head * heads, eps=eps)
        self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        self.to_q = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        self.to_k = ReplicatedLinear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = ReplicatedLinear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias)
        if prope:
            self.to_out_prope = ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias)
        else:
            self.to_out_prope = None
        # if not self.pre_only:
        #     self.to_out = nn.ModuleList([])
        #     self.to_out.append(ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias))
        #     self.to_out.append(nn.Dropout(dropout))
        #     if prope:
        #         self.to_out_prope = nn.ModuleList([])
        #         self.to_out_prope.append(ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias))
        #         self.to_out_prope.append(nn.Dropout(dropout))
        # else:
        #     self.to_out = None
        #     self.to_out_prope = None

        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )