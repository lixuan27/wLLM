# SPDX-License-Identifier: Apache-2.0
"""AMD ROCm platform."""

import torch

from wllm.serving.platforms.interface import Platform, PlatformEnum


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # HIP exposes AMD GPUs via the "cuda" device type
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_torch_device(cls):
        return torch.cuda
