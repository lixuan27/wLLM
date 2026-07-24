"""Variant `dit_sp2`: DiT sequence-parallel across 2 GPUs, VAE on rank 0 only.
Isolates DiT-SP scaling at N=2. See `sp_engine`."""

from wllm.apps.longlive.backend.rocm.sp_engine import sp_main


def main(cfg_path: str) -> None:
    sp_main(cfg_path, sp_size=2, vae_mode="rank0")
