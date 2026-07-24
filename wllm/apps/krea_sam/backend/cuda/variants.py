"""Variant registry — the live, IR-derived deployment-variant queue.

Each entry is one deployment: a process topology + scheduling choice
over the shared engine (``engine/``). The launcher (``launch.py``)
derives the GPU assignment from the topology (devices 0..n-1) and
writes the per-run config.

Topology fields:
  engine     - "reference" (user backend) or "coordinator" (agent engine)
  n_krea     - number of GPUs for the Krea service (>1 => DiT sequence parallel)
  colocate   - SAM shares the Krea GPU (True) or gets its own (False)
  knobs      - extra BackendConfig knobs (stream_decode, sam_compile, ...)
  gpus       - total GPUs this variant occupies (derived)
"""

from __future__ import annotations


def _total_gpus(spec) -> int:
    if spec["engine"] == "reference":
        return 1
    n = spec["n_krea"]
    if not spec.get("colocate", False):
        n += 1  # separate SAM GPU
    # extra GPUs for a split Krea model-pipeline
    extra = spec.get("knobs", {}).get("krea_pipeline_gpus")
    if extra:
        flat = set()
        for g in extra:
            flat.update(g if isinstance(g, list) else [g])
        n = max(n, len(flat)) + (0 if spec.get("colocate") else 1)
    return n


VARIANTS = {
    # ---- baseline ----
    "reference": dict(
        engine="reference", n_krea=1, colocate=True, knobs={},
        hypothesis="user reference: SAM then Krea, sequential, single GPU.",
        ir_basis="anchor — the correctness oracle and Δ baseline.",
        targets="both"),

    # ---- isolated levers (each vs reference) ----
    "sam_krea_split": dict(
        engine="coordinator", n_krea=1, colocate=False, knobs={},
        hypothesis="run SAM and Krea concurrently on separate GPUs.",
        ir_basis="worker graph: krea_v2v ‖ sam_segment are cross-chunk independent (no shared state).",
        targets="both"),
    # Single-GPU "parallel co-location" control. Implemented as ONE process with
    # SAM and Krea on TWO THREADS (one CUDA context + allocator, separate streams),
    # NOT two processes: on B200 the two-process form crashes Krea with a CUDA
    # launch failure once both run concurrently (two contexts/allocators on one
    # GPU over-subscribe under cudnn.benchmark + cold-cache autotune). The threaded
    # form shares the reference's working single-context model. Worker is a
    # standalone single-process backend (sam_colocate/worker.py), not the
    # coordinator; the `engine`/`colocate` fields only drive the 1-GPU assignment.
    "sam_colocate": dict(
        engine="coordinator", n_krea=1, colocate=True, knobs={},
        hypothesis="SAM ‖ Krea concurrently on ONE GPU via two in-process threads (separate CUDA streams) — isolates the 'separate-GPU' lever from mere async overlap; expected ≈ 1-GPU reference (single GPU is compute-bound).",
        ir_basis="worker-graph independence (krea_v2v ‖ sam_segment share no state) realized intra-process; co-located device placement (device-placement lever).",
        targets="both"),
    "sam_compile": dict(
        engine="coordinator", n_krea=1, colocate=False, knobs={"sam_compile": True},
        hypothesis="torch.compile the SAM black box on top of the split.",
        ir_basis="sam_segment is BLACK_BOX — only its internal compile knob + placement are levers.",
        targets="both"),
    "stream_decode": dict(
        engine="coordinator", n_krea=1, colocate=False, knobs={"stream_decode": True},
        hypothesis="emit composited frames one-by-one as the chunk decodes.",
        ir_basis="streaming edge vae_decode→composite / krea_v2v→composite (per-frame, below-IR-granularity).",
        targets="latency+smoothness"),
    # NOTE: no SP=2 variants — the Krea DiT's Ulysses SP and the
    # wllm.kernels_t prope rope kernel require even frame-count sharding, and
    # chunk_size=3 only shards evenly at SP in {1, 3}.
    "krea_sp3": dict(
        engine="coordinator", n_krea=3, colocate=False, knobs={},
        hypothesis="shard the Krea DiT across 3 GPUs (chunk_size=3 divides evenly).",
        ir_basis="below-IR within-chunk model parallelism; 3 latent frames shard evenly across SP=3.",
        targets="latency"),

    "krea_stream_frames": dict(
        engine="coordinator", n_krea=1, colocate=False, knobs={"krea_stream_frames": True},
        hypothesis="Krea service streams each latent's decoded frames as produced; coordinator composites + emits incrementally.",
        ir_basis="model-graph streaming edge vae_decode→composite, realized *producer-side* (vs stream_decode which only changed the coordinator's write granularity).",
        targets="latency+smoothness"),

    # ---- combinations ----
    "combined_compile_stream": dict(
        engine="coordinator", n_krea=1, colocate=False,
        knobs={"sam_compile": True, "krea_stream_frames": True},
        hypothesis="stack SAM compile + producer-side frame streaming on the split.",
        ir_basis="combine the SAM black-box knob with the producer-side streaming edge.",
        targets="both"),
    "combined_sp3_compile": dict(
        engine="coordinator", n_krea=3, colocate=False, knobs={"sam_compile": True},
        hypothesis="stack DiT SP=3 (Krea latency) with SAM compile (SAM latency).",
        ir_basis="within-chunk DiT parallelism + SAM black-box knob — attack both concurrent stages.",
        targets="latency"),
    "combined_sp3_compile_stream": dict(
        engine="coordinator", n_krea=3, colocate=False,
        knobs={"sam_compile": True, "krea_stream_frames": True},
        hypothesis="full stack: DiT SP=3 + SAM compile + producer-side frame streaming.",
        ir_basis="all three levers across both stages + the producer-side streaming edge.",
        targets="both"),
    # combined_sp2_compile removed for the same SP=2 even-sharding reason as krea_sp2.

    # 3-GPU model-graph pipeline: split the Krea v2v stage into a VAE service
    # (encode+decode) and a DiT service (denoise), pipelined across chunks, with
    # SAM on its own GPU.
    "krea_vae_dit_split": dict(
        engine="coordinator", n_krea=2, colocate=False,
        knobs={"krea_pipeline": "vae_dit"},
        hypothesis="split Krea into VAE-stage ‖ DiT-stage across 2 GPUs, pipelined across chunks (+ SAM).",
        ir_basis="model graph: [vae_encode]/[vae_decode] caches are disjoint from the DiT's → the Krea sub-stages pipeline across chunks.",
        targets="throughput"),

    # 5-GPU full stack: krea_vae_dit_split (VAE service on its own GPU) but with
    # the DiT stage sharded SP=3 across 3 GPUs, plus SAM compile and producer-side
    # per-latent decode streaming. krea_gpus = [VAE, DiT0, DiT1, DiT2] (4 GPUs) and
    # SAM gets its own GPU → 5 GPUs total. The coordinator routes krea_gpus[0] to
    # the VAE service and krea_gpus[1:] to the DiT service (SP=len(krea_gpus[1:])).
    "combined_sp3_vae_split_compile_stream": dict(
        engine="coordinator", n_krea=4, colocate=False,
        knobs={"krea_pipeline": "vae_dit", "sam_compile": True, "krea_stream_frames": True},
        hypothesis="combined_sp3_compile_stream + VAE disaggregation: VAE encode/decode on its own GPU, DiT SP=3 on 3 GPUs, SAM compiled on its own GPU, per-latent decode streaming — a true 5-GPU variant.",
        ir_basis="compose three independently-validated transforms: model-graph VAE‖DiT pipeline (krea_vae_dit_split), within-chunk DiT sequence parallelism SP=3 (krea_sp3), the SAM black-box compile knob, and the producer-side vae_decode→composite streaming edge (krea_stream_frames).",
        targets="both"),
}


def total_gpus(name: str) -> int:
    return _total_gpus(VARIANTS[name])
