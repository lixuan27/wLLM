# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple, Union, Iterable
import torch
import diffusers.models.autoencoders.vae as diff_vae
import diffusers.models.autoencoders.autoencoder_kl_wan as diff_wan_vae
import torch
import torch.nn as nn
import torch.nn.functional as F
import wllm.kernels_t
from wllm.serving.configs.models.vaes.wanvae import WanVAEConfig
from wllm.serving.models.vae.base import BaseVAE
from wllm.serving.models.loader.weight_utils import maybe_remap_kv_scale_name, default_weight_loader, map_param_name
from wllm.serving.distributed.vae_plan import split_tile, gather_tile
from wllm.serving.distributed.parallel_state import get_world_rank, get_world_size
# from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.modeling_outputs import AutoencoderKLOutput
#from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
#from wan.inference.tile_parallel import split_tile, gather_tile

CACHE_T = 2
#import torch.distributed as dist
class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal downsample / patchify + grouped mean projection.

        Input:
            x: (B, T, H, W, C)  channel-last
        Output:
            y: (B, T', H', W', out_channels)
        """
        pass


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        pass

class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """
    _CACHE_IDX = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        use_cache: bool = False
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.cache_idx = WanCausalConv3d._CACHE_IDX
        WanCausalConv3d._CACHE_IDX += 1
        self.use_cache = use_cache

        # Set up causal padding
        self._padding = (0, 0, self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None):
        
        x = wllm.kernels_t.nhwc_pad.cat_pad_ndhwc_fused(
            x, cache_x, self._padding
        )
        x = x.permute(0, 4, 1, 2, 3)

        assert x.is_contiguous(memory_format=torch.channels_last_3d)
        assert self.weight.is_contiguous(memory_format=torch.channels_last_3d)
        x = x.contiguous(memory_format=torch.channels_last_3d)
        
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)

        # torch.compile(max-autotune) does not emit channels_last_3d for this
        # conv on gfx950, so the permute above leaves a non-contiguous view.
        # The downstream NDHWC path needs it contiguous.
        x = x.contiguous()

        return x

    @torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    def conv(
        self,
        x
    ):
        return super().forward(x)

class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0
        assert not bias

    def forward(self, x, channel_first: bool = False):
        
        if not channel_first:
            return F.normalize(x, dim=-1) * self.scale * self.gamma.squeeze() + self.bias
        else:
            return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias
    


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0), use_cache=True)

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), use_cache=True)
        
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, t, h, w, c = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    # x: [b, t, h, w, c]  —— 缓存最后 CACHE_T 帧（时间维是 dim=1）
                    cache_x = x[:, -CACHE_T:, :, :, :].clone()

                    # 如果缓存帧数不足 2，就从上一块补一帧
                    if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # feat_cache[idx]: [b, t_cache, h, w, c]
                        # 取上一块最后一帧，变成 [b, 1, h, w, c]，拼到前面
                        cache_x = torch.cat(
                            [feat_cache[idx][:, -1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x],
                            dim=1,
                        )

                    if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=1,
                        )

                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])

                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.view(b, t, h, w, 2, c)
                    x = x.permute(0, 1, 4, 2, 3, 5)
                    x = x.reshape(b, 2*t, h, w, c)
                    

        t = x.shape[1]
        x = x.permute(0, 1, 4, 2, 3).view(b * t, c, h, w)
        x = self.conv(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 1, 3, 4, 2).contiguous()
    
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -1:, :, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, -1:,:, :, :], x], 1))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


    @torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    def conv(
        self,
        x
    ):  
        x = self.resample[0](x)
        return self.resample[1](x)




class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1, use_cache=True)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1, use_cache=True)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1, use_cache=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Apply shortcut connection
        if isinstance(self.conv_shortcut, WanCausalConv3d):
            w = self.conv_shortcut.weight.squeeze()
            b = self.conv_shortcut.bias
            h = F.linear(x, w, b)
        # First normalization and activation
        else:
            h = x

        x = wllm.kernels_t.nhwc_norm.rmsnorm_ndhwc_mul_silu(
            x, self.norm1.gamma
        )
        
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, -1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        
        
        x = wllm.kernels_t.nhwc_norm.rmsnorm_ndhwc_mul_silu(
            x, self.norm2.gamma
        )

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, -1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        
        return x + h


class WanAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    @torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    def forward(self, x: torch.Tensor):
        
        
        identity = x

        batch_size, time, height, width, channels = x.size()
        
        x = wllm.kernels_t.nhwc_norm.rmsnorm_ndhwc_mul(
            x, self.norm.gamma
        )

        x = x.view(batch_size * time, height, width, -1).permute(0, 3, 1, 2)
        
        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.permute(0, 2, 3, 1)
        qkv = qkv.view(batch_size * time, 1, height * width, -1)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        
        x = x.squeeze(1).reshape(batch_size * time, height, width, channels).permute(0, 3, 1, 2)

        # output projection
        x = self.proj(x)

        x = x.permute(0, 2, 3, 1).view(batch_size, time, height, width, channels)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache, feat_idx)

        return x

class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
    
        x_copy = x.clone()
        
        for i, resnet in enumerate(self.resnets):
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)
        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache, feat_idx)
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            # x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
            x = wllm.kernels_t.nhwc_dupup.fused_repeat_shuffle_ndhwc_plus_y(
                x_copy, x, ft=self.avg_shortcut.factor_t, fs=self.avg_shortcut.factor_s,
                out_channels=self.avg_shortcut.out_channels, repeats=self.avg_shortcut.repeats, first_chunk=first_chunk
            )
        
        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

    
        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1, use_cache=True)
        
        
        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        
        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        
        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1, use_cache=True)

        self.gradient_checkpointing = False
        self.world_size = get_world_size()
        self.rank = get_world_rank()

    def forward(self, x: torch.Tensor, feat_cache: List[torch.Tensor] = None, feat_idx=[0], first_chunk=False):
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, -1, : ,:, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)
            
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)
        
        if self.world_size > 1:
            x, meta = split_tile(x)

        ## upsamples
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk)
            
        
        x = self.norm_out(x)
        
        x = self.nonlinearity(x)
        
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, -1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            
        else:
            x = self.conv_out(x)
        
        if self.world_size > 1:
            x = gather_tile(x, meta)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x



class WanResidualDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)
        # x_copy = self.avg_shortcut(x_copy)
        x = wllm.kernels_t.nhwc_avgdown.fused_forward_add_y_triton_nthwc(x_copy, x, 
                self.avg_shortcut.factor_t, self.avg_shortcut.factor_s, self.avg_shortcut.out_channels)
        return x


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:,-1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
       
        ## downsamples
        for i, layer in enumerate(self.down_blocks):
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)
        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].clone()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, -1, :, :, :].unsqueeze(1).to(cache_x.device), cache_x], dim=1)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

def patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # x shape: [batch_size, channels, frames, height, width]
    batch_size, channels, frames, height, width = x.shape

    # Ensure height and width are divisible by patch_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")

    # Reshape to [batch_size, channels, frames, height//patch_size, patch_size, width//patch_size, patch_size]
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)

    # Rearrange to [batch_size, channels * patch_size * patch_size, frames, height//patch_size, width//patch_size]
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # x shape: [batch_size, (channels * patch_size * patch_size), frame, height, width]
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)

    # Reshape to [b, c, patch_size, patch_size, f, h, w]
    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)

    # Rearrange to [b, c, f, h * patch_size, w * patch_size]
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)

    return x


class AutoencoderKLWan(BaseVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    #_supports_gradient_checkpointing = False
    #@register_to_config
    _default_config = WanVAEConfig()
    def __init__(
        self,
        config: WanVAEConfig
        # base_dim: int = 96,
        # decoder_base_dim: Optional[int] = None,
        # z_dim: int = 16,
        # dim_mult: Tuple[int] = [1, 2, 4, 4],
        # num_res_blocks: int = 2,
        # attn_scales: List[float] = [],
        # temperal_downsample: List[bool] = [False, True, True],
        # dropout: float = 0.0,
        # latents_mean: List[float] = [
        #     -0.7571,
        #     -0.7089,
        #     -0.9113,
        #     0.1075,
        #     -0.1745,
        #     0.9653,
        #     -0.1517,
        #     1.5508,
        #     0.4134,
        #     -0.0715,
        #     0.5517,
        #     -0.3632,
        #     -0.1922,
        #     -0.9497,
        #     0.2503,
        #     -0.2921,
        # ],
        # latents_std: List[float] = [
        #     2.8184,
        #     1.4541,
        #     2.3275,
        #     2.6558,
        #     1.2196,
        #     1.7708,
        #     2.6052,
        #     2.0743,
        #     3.2687,
        #     2.1526,
        #     2.8652,
        #     1.5579,
        #     1.6382,
        #     1.1253,
        #     2.8251,
        #     1.9160,
        # ],
        # is_residual: bool = False,
        # in_channels: int = 3,
        # out_channels: int = 3,
        # patch_size: Optional[int] = None,
        # scale_factor_temporal: Optional[int] = 4,
        # scale_factor_spatial: Optional[int] = 8,
    ) -> None:
        super().__init__(config=config)

        self.z_dim = self.config.z_dim
        self.temperal_downsample = self.config.temperal_downsample
        self.temperal_upsample = self.temperal_downsample[::-1]

        if self.config.decoder_base_dim is None:
            decoder_base_dim = self.config.base_dim
        else:
            decoder_base_dim = self.config.decoder_base_dim
        
        self.encoder = WanEncoder3d(
            in_channels=self.config.in_channels,
            dim=self.config.base_dim,
            z_dim=self.config.z_dim * 2,
            dim_mult=self.config.dim_mult,
            num_res_blocks=self.config.num_res_blocks,
            attn_scales=self.config.attn_scales,
            temperal_downsample=self.config.temperal_downsample,
            dropout=self.config.dropout,
            is_residual=self.config.is_residual,
        )
        self.quant_conv = WanCausalConv3d(self.config.z_dim * 2, self.config.z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(self.config.z_dim, self.config.z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=self.config.z_dim,
            dim_mult=self.config.dim_mult,
            num_res_blocks=self.config.num_res_blocks,
            attn_scales=self.config.attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=self.config.dropout,
            out_channels=self.config.out_channels,
            is_residual=self.config.is_residual,
        )

        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

    def clear_cache(self):
        self.clear_decoder_cache()
        self.clear_encoder_cache()

    def clear_decoder_cache(self):
        # Use cached conv count to avoid re-iterating modules each call
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def clear_encoder_cache(self):
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor, stream: bool = False):
        """Encode pixel frames -> latent moments.

        ``stream=False`` (default) starts a fresh encode: the encoder's
        causal temporal cache is reset and the first latent frame is the
        single-frame anchor. ``stream=True`` keeps the cache from the
        previous call so consecutive chunks of one video encode
        continuously (4 pixel frames -> 1 latent frame each). The cache is
        left intact afterwards for the next streaming call;
        ``clear_encoder_cache`` / ``clear_cache`` reset it. Decoding never
        touches this cache, so a v2v stream can interleave encode and
        decode of the same session safely.
        """
        _, _, num_frame, _, _ = x.shape

        # Only continue a stream when one has actually been primed.
        primed = (
            getattr(self, "_enc_feat_map", None) is not None
            and self._enc_feat_map[0] is not None
        )
        stream = stream and primed
        if not stream:
            self.clear_encoder_cache()

        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        iter_ = 1 + (num_frame - 1) // 4

        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0 and not stream:
                out = self.encoder(
                    x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                )
                continue
            # Non-stream: the i==0 anchor above already consumed frame 0, so
            # later iters start at offset 1. Stream: the cache holds the
            # boundary, so every iter is a contiguous group of 4 frames.
            offset = 0 if stream else 1
            start = i if stream else i - 1
            out_ = self.encoder(
                x[:, :, offset + 4 * start : offset + 4 * (start + 1), :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = out_ if out is None else torch.cat([out, out_], 1)

        enc = self.quant_conv(out)
        enc = enc.permute(0, 4, 1, 2, 3).contiguous()
        return enc

    def encode(
        self, x: torch.Tensor, return_dict: bool = True, stream: bool = False
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice, stream=stream) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, stream=stream)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    
    def decode(self, z, return_dict=False, is_first_chunk=False):
        # self.enable_tiling()
        return (self._decode(z, is_first_chunk=is_first_chunk).sample,)

    def _decode(self, z: torch.Tensor, return_dict: bool = True, is_first_chunk=False):

        _, _, num_frame, _, _ = z.shape

        if is_first_chunk:
            # Only reset the decoder cache; a streaming encoder cache must survive across decode calls.
            self.clear_decoder_cache()

        z = z.permute(0, 2, 3, 4, 1).contiguous()
        x = self.post_quant_conv(z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=is_first_chunk,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        if self.config.patch_size is not None:
            out = diff_wan_vae.unpatchify(out, patch_size=self.config.patch_size)

        out = torch.clamp(out, min=-1.0, max=1.0)

        # self.clear_cache()
        if not return_dict:
            return (out,)

        return diff_vae.DecoderOutput(sample=out)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for raw_name, loaded_weight in weights:
            
            name = map_param_name(raw_name, self.config.arch_config.param_names_mapping)
            if "rotary_emb.inv_freq" in name:
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                kv_scale_name: str | None = maybe_remap_kv_scale_name(
                    name, params_dict)
                if kv_scale_name is None:
                    continue
                else:
                    name = kv_scale_name
            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

EntryClass = AutoencoderKLWan