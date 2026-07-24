"""Variant `pipeline_dit4_vae1`: DiT sequence-parallel over GPUs 0-3 ∥ a
single-GPU VAE on GPU 4, pipelined (VAE decodes chunk N while the SP-DiT
computes chunk N+1). Stacks the DiT-SP latency lever with the DiT∥VAE pipeline
throughput lever. world=5. See `pipe_engine`."""

from wllm.apps.longlive.backend.rocm.pipe_engine import pipe_main


def main(cfg_path: str) -> None:
    pipe_main(cfg_path, num_dit=4, num_vae=1)
