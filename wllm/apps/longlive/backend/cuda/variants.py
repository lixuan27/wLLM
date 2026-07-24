"""Registry of LongLive deployment variants (topology only; the launcher maps
each topology onto CUDA devices 0..n-1).

Levers (see the IR analysis in backend/ir/):
  A DiT sequence parallelism (intra-op)      -> dit ranks > 1
  B VAE tile-parallel (intra-op)             -> vae_mode "tile" (world in 2..4)
  C DiT|VAE cross-chunk pipeline (decouple)  -> kind "disagg"
  D ASR overlap (pipeline scheduling)        -> asr_mode "async"
  E device placement / co-location           -> gpu counts per role

SP must divide chunk_size=8, so DiT SP is limited to {1, 2, 4, 8}.
"""
from __future__ import annotations

from typing import Dict

# kind: "mono" (one SP world, all ranks DiT+VAE) or
#       "disagg" (DiT SP group + separate VAE group, cross-chunk pipeline)
VARIANTS: Dict[str, dict] = {
    "reference": dict(
        kind="reference",
        summary="user's sequential single-GPU backend (burst-prone, ~14 fps)"),
    "colocated": dict(
        kind="mono", dit=1, vae="rank0", asr="sync",
        summary="optimized worker on 1 GPU, reference numerics (parity control)"),
    "dit_sp2": dict(
        kind="mono", dit=2, vae="tile", asr="sync",
        summary="DiT sequence-parallel over 2 GPUs, tiled VAE (2-GPU latency pick)"),
    "dit_sp4": dict(
        kind="mono", dit=4, vae="tile", asr="sync",
        summary="DiT SP=4, tiled VAE (4-GPU pick: real-time fps, best latency)"),
    "dit_sp8": dict(
        kind="mono", dit=8, vae="rank0", asr="sync",
        summary="DiT SP=8 (measured: comm-bound, loses to the combined variants)"),
    "dit_sp4_vae_rank0": dict(
        kind="mono", dit=4, vae="rank0", asr="sync",
        summary="SP=4 with the VAE decode on rank 0 only (isolates the VAE-tile share)"),
    "asr_async": dict(
        kind="mono", dit=1, vae="rank0", asr="async",
        summary="ASR in its own process/GPU: removes the prompt-update stall"),
    "vae_decouple": dict(
        kind="disagg", dit=1, vae_ranks=1, asr="sync",
        summary="VAE decode of chunk N overlaps DiT of chunk N+1; bit-exact (2-GPU fps pick)"),
    "combined_sp4_async": dict(
        kind="mono", dit=4, vae="tile", asr="async",
        summary="SP=4 + async ASR"),
    "combined_sp4_decouple": dict(
        kind="disagg", dit=4, vae_ranks=1, asr="async",
        summary="SP=4 DiT + decoupled VAE + async ASR (6-GPU pick)"),
    # The combined_sp4_vae{2,3} pair was developed under the names
    # "combined_best" / "combined_8gpu": the decoupled VAE group grows to 2 or
    # 3 tile ranks because the VAE decode costs about as much per chunk as the
    # SP=4 DiT.
    "combined_sp4_vae2": dict(
        kind="disagg", dit=4, vae_ranks=2, asr="async",
        summary="SP=4 DiT + 2-GPU tile VAE + async ASR (7-GPU pick, recommended)"),
    "combined_sp4_vae3": dict(
        kind="disagg", dit=4, vae_ranks=3, asr="async",
        summary="SP=4 DiT + 3-GPU tile VAE + async ASR (8-GPU pick, max throughput)"),
}


def gpus_needed(name: str) -> int:
    v = VARIANTS[name]
    if v["kind"] == "reference":
        return 1
    n = v.get("dit", 1)
    if v["kind"] == "disagg":
        n += v.get("vae_ranks", 1)
    if v.get("asr") == "async":
        n += 1
    return n
