"""In-tree omni-modal staged serving engine.

A self-contained implementation of the async multi-stage engine contract
the apps program against (`AsyncOmni`, stage-config YAMLs, AR/generation
stage schedulers, omni output objects). ``WLLM_OMNI_ENGINE=wllm.omni``
binds this engine at the contract level; running a real app end-to-end
on it additionally requires that app's model runners to be registered
(see :mod:`wllm.omni.stages`) — until then the honest support tier is
contract-verified, not app-verified.

Model execution is pluggable through :mod:`wllm.omni.stages`. Real model
runners register themselves; unknown models fail closed instead of
silently degrading (the deterministic echo stage must be requested
explicitly and is for tests and dry runs only).
"""

from .engine import AsyncOmni, ModelNotSupported
from .sampling import SamplingParams

__version__ = "0.0.1a0"

__all__ = ["AsyncOmni", "ModelNotSupported", "SamplingParams"]
