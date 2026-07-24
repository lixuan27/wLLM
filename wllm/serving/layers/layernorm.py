# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wllm.kernels_t
from wllm.serving.layers.custom_op import CustomOp


@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight

        self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)
        
    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError("Expected hidden_size to be "
                             f"{self.hidden_size}, but found: {hidden_size}")

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}")

            x_var = x[:, :, :self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        return x
    
    def forward_cuda(self, x: torch.Tensor):

        from wllm.serving.platforms import current_platform
        if current_platform.is_rocm():
            weight = self.weight if self.has_weight else None
            return F.rms_norm(x, (x.shape[-1], ), weight, self.eps)

        import flashinfer
        b, s, d = x.shape
        x = x.reshape(b * s, d)
        x = flashinfer.norm.rmsnorm(x, self.weight, self.eps).reshape(b, s, d)

        return x

        

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.eps}"
        return s


@CustomOp.register("fp32_layernorm")
class FP32LayerNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.has_bias = bias
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias = torch.zeros(hidden_size, dtype=torch.float32)
        if self.elementwise_affine:
            self.weight = nn.Parameter(self.weight)
            if self.has_bias:
                self.bias = nn.Parameter(self.bias)

    def forward_native(self, inputs: torch.Tensor):

        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            (self.hidden_size,),
            self.weight.float() if self.elementwise_affine is not None else None,
            self.bias.float() if (self.elementwise_affine and self.has_bias) is not None else None,
            self.eps,
        ).to(origin_dtype)

    def forward_cuda(self, inputs: torch.Tensor):

        return wllm.kernels_t.norm_ops.norm_fp32(inputs.contiguous(), self.weight, self.bias, self.eps)


@CustomOp.register("fp32_layernorm_scale_shift")
class FP32LayerNormScaleShift(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.has_bias = bias
        assert not self.elementwise_affine
        assert not self.has_bias
    
    def forward_native(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        
        raise NotImplementedError
    

    def forward_cuda(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        
        token_length = x.shape[1] // scale.shape[0]

        return wllm.kernels_t.norm_ops.norm_scale_shift(
            x.contiguous(), 
            scale.contiguous(), 
            shift.contiguous(), 
            token_length, 
            self.eps
        )
    

@CustomOp.register("scale_residual")
class ScaleResidual(CustomOp):
    def __init__(
        self
    ) -> None:
        super().__init__()
        
    def forward_native(self, residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor):
        
        raise NotImplementedError
    

    def forward_cuda(self, residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor):
        
        token_length = x.shape[1] // gate.shape[0]
        return wllm.kernels_t.norm_ops.scale_residual(
                    residual.contiguous(), 
                    x.contiguous(), 
                    gate.contiguous(), 
                    token_length)
    