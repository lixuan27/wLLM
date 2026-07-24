"""Variant `pipeline_dit4_vae4`: DiT sequence-parallel over GPUs 0-3 ∥ VAE
spatial-tile over GPUs 4-7, pipelined across chunks. Stacks all three winning
levers (DiT-SP latency + VAE-tile throughput + DiT∥VAE overlap). The VAE tile
runs over the VAE subgroup via the group-scoped `vendor_vae_plan`. world=8.
See `pipe_engine`."""

from wllm.apps.longlive.backend.rocm.pipe_engine import pipe_main


def main(cfg_path: str) -> None:
    pipe_main(cfg_path, num_dit=4, num_vae=4)
