"""Variant `pipeline_dit4_vae4_async_asr`: the `pipeline_dit4_vae4` topology
(DiT-SP4 ∥ VAE-tile4, pipelined) with **async ASR** stacked on the coordinator,
so the mid-stream prompt updates never stall the gen loop. Stacks all winning
levers: DiT sequence parallelism (latency), VAE spatial-tile (throughput), the
DiT∥VAE pipeline overlap (throughput), and off-critical-path ASR (narration
latency + smoothness). world=8. See `pipe_engine` + `async_asr_mixin`."""

from wllm.apps.longlive.backend.rocm.async_asr_mixin import AsyncASRMixin
from wllm.apps.longlive.backend.rocm.pipe_engine import PipeDiTCoordinator, pipe_main


class CombinedCoordinator(AsyncASRMixin, PipeDiTCoordinator):
    pass


def main(cfg_path: str) -> None:
    pipe_main(cfg_path, num_dit=4, num_vae=4, coordinator_cls=CombinedCoordinator)
