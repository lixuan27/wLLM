"""Platform-aware Triton autotune config selection."""
import torch

_IS_ROCM = getattr(torch.version, "hip", None) is not None


def configs_for_platform(configs):
    """Return the autotune configs supported on the current GPU backend.

    ROCm's Triton backend does not pipeline beyond ``num_stages=2``; those
    configs are dropped so autotuning does not attempt them. CUDA keeps all.
    """
    if _IS_ROCM:
        return [c for c in configs if getattr(c, "num_stages", 1) <= 2]
    return configs
