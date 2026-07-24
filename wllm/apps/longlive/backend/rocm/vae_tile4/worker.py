"""Variant `vae_tile4`: VAE spatial-tile parallel across 4 GPUs with the DiT
*replicated* (SP size 1). Isolates the VAE-tile lever (the DiT does redundant
full work on every rank; all ranks share identical noise via the broadcast hook
so the tile decode is consistent). See `sp_engine`."""

from wllm.apps.longlive.backend.rocm.sp_engine import sp_main


def main(cfg_path: str) -> None:
    sp_main(cfg_path, sp_size=1, vae_mode="tile")
