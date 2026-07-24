"""Launch a LiveAvatar backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/liveavatar/adapter.py``) and reads the same app config
(``wllm/apps/liveavatar/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``LiveAvatar backend READY`` line, then start the frontend. Ctrl-C
stops the backend (and its cluster ranks).

A variant that uses n GPUs runs on CUDA devices 0..n-1: the worker
(DiT/VAE/ASR/Wav2Vec, or ASR/Wav2Vec only for the dedicated-VAE variant)
on device 0, then any DiT/VAE cluster ranks, then the LLM and TTS
engines. Placement is managed internally with physical device indices,
so remapping via CUDA_VISIBLE_DEVICES is not supported; the launcher
clears it.

Usage:
  python -m wllm.apps.liveavatar.backend.cuda.launch --variant combined_stream_pp
  python -m wllm.apps.liveavatar.backend.cuda.launch --list

The scripts under ``examples/liveavatar/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import importlib
import os
from wllm.serving.paths import app_dir, repo_root
import sys
import tempfile

import yaml

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")

_REFERENCE = "wllm.apps.liveavatar.reference.worker:LiveAvatarWorker"
_STREAM = "wllm.apps.liveavatar.backend.cuda.stream_app.worker:StreamAppWorker"
_PP = "wllm.apps.liveavatar.backend.cuda.pp_steps.worker:PPStepsWorker"
_COMBINED = "wllm.apps.liveavatar.backend.cuda.combined_stream_pp.worker:CombinedStreamPPWorker"
_COMBINED_VAE = "wllm.apps.liveavatar.backend.cuda.combined_stream_pp_vae.worker:CombinedStreamPPVAEWorker"

# Each entry: worker class, GPU count, config overrides (engine placement, as
# physical device indices), extra env (cluster placement), and a summary shown
# by --list. Device 0 always hosts the worker process.
REGISTRY = {
    "reference": dict(
        cls=_REFERENCE, gpus=2,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 1}, env={},
        summary="sequential reference: DiT+VAE+ASR+Wav2Vec on device 0, LLM+TTS on 1"),
    "stream_app": dict(
        cls=_STREAM, gpus=2,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 1}, env={},
        summary="app streaming: overlap TTS with generation, emit frames per chunk (2-GPU pick)"),
    "place_llm_tts_split": dict(
        cls=_REFERENCE, gpus=3,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 2}, env={},
        summary="reference with LLM and TTS on separate GPUs (placement probe)"),
    "place_stream_split": dict(
        cls=_STREAM, gpus=3,
        overrides={"llm_gpu_index": 1, "tts_gpu_index": 2}, env={},
        summary="stream_app with LLM and TTS on separate GPUs (placement probe)"),
    "pp_steps": dict(
        cls=_PP, gpus=6,
        overrides={"llm_gpu_index": 5, "tts_gpu_index": 5},
        env={"PP_CLUSTER_GPUS": "1,2,3,4"},
        summary="cross-chunk DiT step pipeline on 4 GPUs, burst emission (throughput lever)"),
    "combined_stream_pp": dict(
        cls=_COMBINED, gpus=6,
        overrides={"llm_gpu_index": 5, "tts_gpu_index": 5},
        env={"PP_CLUSTER_GPUS": "1,2,3,4"},
        summary="streaming + DiT step pipeline (6-GPU pick: low latency, real-time fps)"),
    "combined_stream_pp_vae": dict(
        cls=_COMBINED_VAE, gpus=7,
        overrides={"llm_gpu_index": 6, "tts_gpu_index": 6},
        env={"PP_CLUSTER_GPUS": "1,2,3,4,5"},
        summary="streaming + 5-rank DiT+VAE pipeline, LLM+TTS shared (7-GPU pick; needs GPU P2P)"),
    "combined_stream_pp_vae_split": dict(
        cls=_COMBINED_VAE, gpus=8,
        overrides={"llm_gpu_index": 6, "tts_gpu_index": 7},
        env={"PP_CLUSTER_GPUS": "1,2,3,4,5"},
        summary="streaming + 5-rank DiT+VAE pipeline, LLM and TTS split (8-GPU pick; needs GPU P2P)"),
}


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

    # Engine and cluster placement use physical device indices, so the variant
    # must see the machine's devices unremapped.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    for var, val in spec["env"].items():
        os.environ[var] = val
    # Long continuous sessions fragment the allocator (per-frame VAE decode);
    # expandable segments keeps the worker from creeping into OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {spec['gpus']} (CUDA devices 0..{spec['gpus'] - 1})")
    print(f"[launch] engines = LLM on device {spec['overrides']['llm_gpu_index']}, "
          f"TTS on device {spec['overrides']['tts_gpu_index']}"
          + (f", cluster on {spec['env']['PP_CLUSTER_GPUS']}" if spec["env"] else ""))
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        return

    derived = _derive_config(args.config, spec["overrides"])
    print(f"[launch] derived config: {derived}")
    print("[launch] starting; wait for 'LiveAvatar backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    # Relative paths in the config (image_path, assets) resolve against the repo root.
    os.chdir(_REPO_ROOT)

    mod_name, cls_name = spec["cls"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    worker = None
    try:
        worker = cls(cfg_path=derived)
        worker.loop()
    finally:
        # Runs on Ctrl+C, graceful TERM, or any exception, so the cluster ranks
        # (spawned setsid, out of the terminal's signal reach) and the shm
        # buffers are cleaned up instead of orphaned. terminate() is idempotent.
        if worker is not None:
            worker.terminate()


if __name__ == "__main__":
    main()
