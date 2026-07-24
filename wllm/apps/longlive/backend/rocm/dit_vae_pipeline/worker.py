"""Variant `dit_vae_pipeline`: DiT on GPU 0 ∥ VAE on GPU 1, pipelined across
chunks (VAE decodes chunk N while the DiT computes chunk N+1). Isolates the
pipeline-parallel lever from the IR's 2-stage partition. See `pipe_engine`.

Launched on 2 ranks by `launch.py --variant dit_vae_pipeline`.
"""

from wllm.apps.longlive.backend.rocm.pipe_engine import pipe_main


def main(cfg_path: str) -> None:
    pipe_main(cfg_path, num_dit=1, num_vae=1, vae_tile=False)
