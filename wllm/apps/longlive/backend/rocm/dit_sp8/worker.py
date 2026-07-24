"""Variant `dit_sp8`: DiT sequence-parallel across all 8 GPUs, VAE on rank 0
only. Max DiT shard. See `sp_engine`."""

from wllm.apps.longlive.backend.rocm.sp_engine import sp_main


def main(cfg_path: str) -> None:
    sp_main(cfg_path, sp_size=8, vae_mode="rank0")
