"""Launch a LiveAvatar backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/liveavatar/adapter.py``) and reads the same app config
(``wllm/apps/liveavatar/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``LiveAvatar backend READY`` line, then start the frontend. Ctrl-C
stops the backend (and any worker ranks it spawned).

A variant that uses n GPUs runs on CUDA devices 0..n-1: the worker
process on device 0, then any additional DiT ranks, then the LLM and TTS
engines. Placement is managed internally with physical device indices,
so remapping via CUDA_VISIBLE_DEVICES is not supported; the launcher
clears it.

Usage:
  python -m wllm.apps.liveavatar.backend.rocm.launch --variant stream_pp2_sp3
  python -m wllm.apps.liveavatar.backend.rocm.launch --list

The scripts under ``examples/liveavatar/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import importlib
import os
from wllm.serving.paths import app_dir, repo_root
import tempfile

import yaml

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")

_B = "wllm.apps.liveavatar.backend.rocm"
_REFERENCE = "wllm.apps.liveavatar.reference.worker:LiveAvatarWorker"

# Each entry: worker class, GPU count, config overrides (engine placement, as
# physical device indices), extra env (rank counts and device placement read by
# the worker), and a summary shown by --list. Device 0 always hosts the worker
# process; multi-rank variants put rank k on device k.
#
# Ulysses sequence parallelism shards the 3 generate frames across ranks, so
# SP must divide 3: SP=3 is the only legal degree above 1 (SP=2 splits them
# [2,1] and the all-to-all asserts). Extra GPUs go to pipeline stages instead.
REGISTRY = {
    "reference": dict(
        cls=_REFERENCE, gpus=2,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 1}, env={},
        summary="sequential reference: pipeline on device 0, LLM+TTS on 1"),
    "stream_liveavatar": dict(
        cls=f"{_B}.stream_liveavatar.worker:StreamLiveAvatarWorker", gpus=2,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 1}, env={},
        summary="emit each chunk's frames as produced instead of one batch at the end (2-GPU pick)"),
    "stream_full": dict(
        cls=f"{_B}.stream_full.worker:StreamFullWorker", gpus=3,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 2}, env={},
        summary="also start the DiT on the first TTS chunk, overlapping TTS (3-GPU pick)"),
    "place_dedicated": dict(
        cls=_REFERENCE, gpus=4,
        overrides={"asr_gpu_index": 1, "llm_gpu_index": 2, "tts_gpu_index": 3}, env={},
        summary="reference with ASR, LLM and TTS each on their own GPU (placement probe)"),
    "wav2vec_offload": dict(
        cls=f"{_B}.wav2vec_offload.worker:Wav2VecOffloadWorker", gpus=4,
        overrides={"llm_gpu_index": 2, "tts_gpu_index": 3},
        env={"WAV2VEC_DEVICE": "cuda:1"},
        summary="chunk streaming with wav2vec on its own GPU (placement probe)"),
    "dit_sp3": dict(
        cls=f"{_B}.dit_sp.worker:DiTSPWorker", gpus=5,
        overrides={"llm_gpu_index": 3, "tts_gpu_index": 4},
        env={"DIT_SP_WORLD": "3"},
        summary="sequence-parallel DiT over 3 GPUs (5-GPU pick, sustains real-time)"),
    "denoise_pp4": dict(
        cls=f"{_B}.denoise_pp.worker:DenoisePipelineWorker", gpus=6,
        overrides={"llm_gpu_index": 4, "tts_gpu_index": 5},
        env={"DENOISE_PP_DRIVER_ONLY": "0", "DENOISE_PP_WORLD": "4"},
        summary="the 4 denoise steps pipelined across 4 GPUs, one per step (6-GPU pick)"),
    "denoise_pp_driver": dict(
        cls=f"{_B}.denoise_pp.worker:DenoisePipelineWorker", gpus=7,
        overrides={"llm_gpu_index": 5, "tts_gpu_index": 6},
        env={"DENOISE_PP_DRIVER_ONLY": "1"},
        summary="5-rank denoise pipeline where the driver runs no step (rebalance probe)"),
    "denoise_pp2_sp3": dict(
        cls=f"{_B}.denoise_pp_sp.worker:DenoisePPSPWorker", gpus=8,
        overrides={"llm_gpu_index": 6, "tts_gpu_index": 7},
        env={"DENOISE_PP_STAGES": "2", "DENOISE_PP_SP": "3"},
        summary="2 denoise pipeline stages, each sequence-parallel over 3 GPUs"),
    "stream_pp2_sp3": dict(
        cls=f"{_B}.stream_pp_sp.worker:StreamPPSPWorker", gpus=8,
        overrides={"llm_gpu_index": 6, "tts_gpu_index": 7},
        env={"DENOISE_PP_STAGES": "2", "DENOISE_PP_SP": "3"},
        summary="denoise_pp2_sp3 driven straight from the TTS stream (8-GPU pick)"),
}


def _raise_omni_handshake_timeout() -> None:
    """Give the omni-engine stage handshake longer than its 600 s default.

    A cold first launch autotunes kernels and warms up graphs for the TTS
    token2wav stage, which can outlast the hardcoded handshake timeout and
    abort init. This raises only that startup timeout; no model computation
    changes. Override with WLLM_OMNI_HANDSHAKE_TIMEOUT_S.
    """
    try:
        from wllm.engines import omni as omni_engine
        stage_proc = omni_engine.submodule("engine.stage_engine_core_proc")

        stage_proc._HANDSHAKE_POLL_TIMEOUT_S = int(
            os.environ.get("WLLM_OMNI_HANDSHAKE_TIMEOUT_S", "1800")
        )
    except Exception:
        pass


def _derive_config(base_path: str, overrides: dict) -> str:
    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    fd, path = tempfile.mkstemp(prefix="liveavatar_cfg_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a LiveAvatar backend variant for the frontend.")
    ap.add_argument("--variant", help="one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the launch plan, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in REGISTRY) + 2
        for name, spec in REGISTRY.items():
            print(f"{name:<{width}} {spec['gpus']} GPUs  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant not in REGISTRY:
        ap.error(f"unknown variant '{args.variant}'; see --list")
    spec = REGISTRY[args.variant]

    # Engine and rank placement use physical device indices, so the variant
    # must see the machine's devices unremapped.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    for var, val in spec["env"].items():
        os.environ[var] = val

    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {spec['gpus']} (CUDA devices 0..{spec['gpus'] - 1})")
    print(f"[launch] engines = LLM on device {spec['overrides']['llm_gpu_index']}, "
          f"TTS on device {spec['overrides']['tts_gpu_index']}"
          + (f", {', '.join(f'{k}={v}' for k, v in spec['env'].items())}" if spec["env"] else ""))
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        return

    _raise_omni_handshake_timeout()
    derived = _derive_config(args.config, spec["overrides"])
    print(f"[launch] derived config: {derived}")
    print("[launch] starting; wait for 'LiveAvatar backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    # Relative paths in the config (image_path, checkpoints) resolve against the repo root.
    os.chdir(_REPO_ROOT)

    mod_name, cls_name = spec["cls"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    worker = None
    try:
        worker = cls(cfg_path=derived)
        worker.loop()
    finally:
        # Runs on Ctrl+C, graceful TERM, or any exception, so the spawned step
        # ranks and the shm buffers are cleaned up instead of orphaned.
        if worker is not None:
            worker.terminate()


if __name__ == "__main__":
    main()
