"""Variant `dit_sp4`: DiT sequence-parallel across 4 GPUs (SP group = all 4),
VAE decode on rank 0 only (tiling disabled). Isolates the DiT-SP latency lever.

Launched on 4 ranks by `launch.py --variant dit_sp4`.
"""

from wllm.apps.longlive.backend.rocm.sp_engine import sp_main


def main(cfg_path: str) -> None:
    sp_main(cfg_path, sp_size=4, vae_mode="rank0")
