"""Variant `unified_sp4`: one 4-GPU group running **both** DiT sequence-parallel
(SP group = all 4) **and** VAE spatial-tile parallel (world group = all 4) per
chunk — every GPU works on the same chunk, sequentially DiT then VAE. Combines
the two model-parallel levers to minimize single-chunk (first-frame) latency.
See `sp_engine`."""

from wllm.apps.longlive.backend.rocm.sp_engine import sp_main


def main(cfg_path: str) -> None:
    sp_main(cfg_path, sp_size=4, vae_mode="tile")
