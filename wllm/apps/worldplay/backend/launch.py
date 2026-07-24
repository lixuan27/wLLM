"""Platform dispatcher: routes to this app's CUDA or ROCm variant set.

The two variant sets are agent-generated per hardware and live under
``backend/cuda`` and ``backend/rocm``; the launcher picks by detected platform
so ``python -m wllm.apps.<app>.backend.launch --variant X`` selects the
implementation of ``X`` for the current platform.
"""
import importlib

from wllm.serving.platforms import current_platform


def _platform_launch_module() -> str:
    sub = "rocm" if current_platform.is_rocm() else "cuda"
    # __package__ is this app's backend package under both `-m` and import.
    return f"{__package__}.{sub}.launch"


def main() -> None:
    importlib.import_module(_platform_launch_module()).main()


if __name__ == "__main__":
    main()
