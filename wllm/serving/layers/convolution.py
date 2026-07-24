from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

from wllm.serving.models.utils import set_weight_attrs
from wllm.serving.logger import init_logger
from wllm.serving.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)

# =========================
# Conv2D
# =========================


class Conv2DMethodBase(QuantizeMethodBase):
    """Base class for different (possibly quantized) Conv2D methods.

    A Conv2D "method" encapsulates:
      1) how weights are created/registered on a layer, and
      2) how the convolution is executed (unquantized, fp8, int8, etc.).
    """

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create and register weights for a 2D convolution layer.

        The created weights must be registered as attributes/parameters on `layer`.

        Args:
            layer: The module that will hold the weights.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: 2D kernel size (kH, kW).
            groups: Number of groups in the convolution.
            channel_last: If True, store weights in channels_last memory format.
            params_dtype: Data type used to store the parameters.
            extra_weight_attrs: Extra metadata for weight loading/sharding, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        channel_last: bool,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Conv2D with weights stored on `layer` to input `x`.

        Expects `create_weights(...)` to have been called on `layer` beforehand.

        Args:
            layer: The module holding the weights.
            x: Input tensor shaped (N, C_in, H, W).
            stride: Convolution stride (sH, sW).
            padding: Convolution padding (pH, pW).
            dilation: Convolution dilation (dH, dW).
            groups: Number of groups.
            channel_last: Whether the layer is configured for channels_last.
            bias: Optional bias tensor.
        """
        raise NotImplementedError


class UnquantizedConv2DMethod(Conv2DMethodBase):
    """Unquantized Conv2D method using torch.nn.functional.conv2d."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # torch.nn.functional.conv2d expects weight of shape:
        #   (out_channels, in_channels // groups, kH, kW)
        weight = Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *kernel_size),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # NOTE: channels_last is a memory format for 4D tensors. You can still mark
        # the parameter tensor as channels_last for better layout compatibility.
        if channel_last:
            weight.data = weight.data.contiguous(memory_format=torch.channels_last)

        # Metadata used by wLLM weight loading/sharding logic.
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        channel_last: bool,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ensure bias dtype matches input dtype for mixed precision.
        # NOTE: `channel_last` does not change conv2d API; it is only a memory format hint.
        if bias is None:
            return F.conv2d(x, layer.weight, None, stride, padding, dilation, groups)
        return F.conv2d(
            x, layer.weight, bias.to(x.dtype), stride, padding, dilation, groups
        )


class Conv2DBase(torch.nn.Module):
    """Base class for Conv2D layers supporting optional quantization.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.
        channel_last: If True, the layer prefers channels_last memory format.
        params_dtype: Data type used to store parameters (weights/bias).
        quant_config: Quantization configuration. If None, uses unquantized method.
        prefix: Name prefix for state dict mapping / weight loading.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        dilation: _size_2_t,
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype | None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # Store layer hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.channel_last = channel_last

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.quant_config = quant_config
        self.prefix = prefix

        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = UnquantizedConv2DMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedConv2D(Conv2DBase):
    """A replicated (non-sharded) Conv2D layer with optional quantization.

    This module stores a single copy of parameters (no tensor-parallel sharding).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        channel_last: bool = False,
        bias: bool = True,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channel_last=channel_last,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Conv2D layer supports a quantization method (unquantized if quant_config is None).
        assert self.quant_method is not None

        self.quant_method.create_weights(
            self,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.groups,
            self.channel_last,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.out_channels, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weights into an existing parameter with strict shape checking."""
        # If the weight on disk does not have a shape, give it one
        # (e.g., scalar scales for some quantization formats).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()} "
            f"to a parameter of size {param.size()}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            self,
            x,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.channel_last,
            self.bias,
        )

    def extra_repr(self) -> str:
        bias_flag = self.bias is not None
        quant_name = type(self.quant_method).__name__ if self.quant_method is not None else "None"
        dtype_name = str(self.params_dtype).replace("torch.", "")

        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={tuple(self.kernel_size)}, stride={tuple(self.stride)}, "
            f"padding={tuple(self.padding)}, dilation={tuple(self.dilation)}, "
            f"groups={self.groups}, bias={bias_flag}, channel_last={self.channel_last}, "
            f"params_dtype={dtype_name}, quant_method={quant_name}"
        )
        if self.prefix:
            s += f", prefix={self.prefix}"
        return s


# =========================
# Conv3D
# =========================


class Conv3DMethodBase(QuantizeMethodBase):
    """Base class for different (possibly quantized) Conv3D methods.

    A Conv3D "method" encapsulates:
      1) how weights are created/registered on a layer, and
      2) how the convolution is executed (unquantized, fp8, int8, etc.).
    """

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create and register weights for a 3D convolution layer.

        The created weights must be registered as attributes/parameters on `layer`.

        Args:
            layer: The module that will hold the weights.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: 3D kernel size (kD, kH, kW).
            groups: Number of groups in the convolution.
            channel_last: If True, store weights in channels_last_3d memory format.
            params_dtype: Data type used to store the parameters.
            extra_weight_attrs: Extra metadata for weight loading/sharding, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        groups: int,
        channel_last: bool,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Conv3D with weights stored on `layer` to input `x`.

        Expects `create_weights(...)` to have been called on `layer` beforehand.

        Args:
            layer: The module holding the weights.
            x: Input tensor shaped (N, C_in, D, H, W).
            stride: Convolution stride (sD, sH, sW).
            padding: Convolution padding (pD, pH, pW).
            dilation: Convolution dilation (dD, dH, dW).
            groups: Number of groups.
            channel_last: Whether the layer is configured for channels_last_3d.
            bias: Optional bias tensor.
        """
        raise NotImplementedError


class UnquantizedConv3DMethod(Conv3DMethodBase):
    """Unquantized Conv3D method using torch.nn.functional.conv3d."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # torch.nn.functional.conv3d expects weight of shape:
        #   (out_channels, in_channels // groups, kD, kH, kW)
        weight = Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *kernel_size),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        if channel_last:
            weight.data = weight.data.contiguous(memory_format=torch.channels_last_3d)

        # Metadata used by wLLM weight loading/sharding logic.
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        groups: int,
        channel_last: bool,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ensure bias dtype matches input dtype for mixed precision.
        # NOTE: `channel_last` does not change conv3d API; it is only a memory format hint.
        if bias is None:
            return F.conv3d(x, layer.weight, None, stride, padding, dilation, groups)
        return F.conv3d(
            x, layer.weight, bias.to(x.dtype), stride, padding, dilation, groups
        )


class Conv3DBase(torch.nn.Module):
    """Base class for Conv3D layers supporting optional quantization.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to all three sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.
        channel_last: If True, the layer prefers channels_last_3d memory format.
        params_dtype: Data type used to store parameters (weights/bias).
        quant_config: Quantization configuration. If None, uses unquantized method.
        prefix: Name prefix for state dict mapping / weight loading.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t,
        padding: _size_3_t,
        dilation: _size_3_t,
        groups: int,
        channel_last: bool,
        params_dtype: torch.dtype | None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # Store layer hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.channel_last = channel_last

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.quant_config = quant_config
        self.prefix = prefix

        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = UnquantizedConv3DMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedConv3D(Conv3DBase):
    """A replicated (non-sharded) Conv3D layer with optional quantization.

    This module stores a single copy of parameters (no tensor-parallel sharding).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        channel_last: bool = False,
        bias: bool = True,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channel_last=channel_last,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Conv3D layer supports a quantization method (unquantized if quant_config is None).
        assert self.quant_method is not None

        self.quant_method.create_weights(
            self,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.groups,
            self.channel_last,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.out_channels, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weights into an existing parameter with strict shape checking."""
        # If the weight on disk does not have a shape, give it one
        # (e.g., scalar scales for some quantization formats).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()} "
            f"to a parameter of size {param.size()}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            self,
            x,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.channel_last,
            self.bias,
        )

    def extra_repr(self) -> str:
        bias_flag = self.bias is not None
        quant_name = type(self.quant_method).__name__ if self.quant_method is not None else "None"
        dtype_name = str(self.params_dtype).replace("torch.", "")

        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={tuple(self.kernel_size)}, stride={tuple(self.stride)}, "
            f"padding={tuple(self.padding)}, dilation={tuple(self.dilation)}, "
            f"groups={self.groups}, bias={bias_flag}, channel_last={self.channel_last}, "
            f"params_dtype={dtype_name}, quant_method={quant_name}"
        )
        if self.prefix:
            s += f", prefix={self.prefix}"
        return s