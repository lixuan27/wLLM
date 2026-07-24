# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py

import traceback
from typing import TYPE_CHECKING
from wllm.serving.platforms.interface import Platform, PlatformEnum
from wllm.serving.utils.utils import resolve_obj_by_qualname
from wllm.serving.logger import init_logger
logger = init_logger(__name__)

def cuda_platform_plugin() -> str | None:
    is_cuda = False

    try:
        from wllm.serving.utils.utils import import_pynvml
        pynvml = import_pynvml()  # type: ignore[no-untyped-call]
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: wllm cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if wllm is built with cpu too.
            # Otherwise, wllm will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (pynvml.nvmlDeviceGetCount() > 0)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") \
                or os.path.exists("/sys/class/tegra-firmware")

        if cuda_is_jetson():
            is_cuda = True
    if is_cuda:
        logger.info("CUDA is available")

    return "wllm.serving.platforms.cuda.CudaPlatform" if is_cuda else None




def rocm_platform_plugin() -> str | None:
    import torch
    # A ROCm/HIP PyTorch build exposes AMD GPUs through torch.version.hip
    # (torch.version.cuda is None). HIP surfaces those GPUs via torch.cuda, so
    # the cuda plugin would also match -- resolve ROCm first.
    if getattr(torch.version, "hip", None) is not None:
        logger.info("ROCm platform detected (HIP %s)", torch.version.hip)
        return "wllm.serving.platforms.rocm.RocmPlatform"
    return None


builtin_platform_plugins = {
    'rocm': rocm_platform_plugin,
    'cuda': cuda_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    for plugin in (rocm_platform_plugin, cuda_platform_plugin):
        platform_cls_qualname = plugin()
        if platform_cls_qualname is not None:
            return platform_cls_qualname
    raise RuntimeError("No platform plugin found. Please check your "
                       "installation.")

_current_platform = None
_init_trace: str = ''

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == 'current_platform':
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from wllm.serving.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `wllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import wllm`, some wllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(
            f"No attribute named '{name}' exists in {__name__}.")


__all__ = ['Platform', 'PlatformEnum', 'current_platform', "_init_trace"]
