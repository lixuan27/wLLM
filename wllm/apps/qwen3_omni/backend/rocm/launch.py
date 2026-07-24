"""Launch a Qwen3-Omni backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/qwen3_omni/adapter.py``) and derives its config from the
same app config (``wllm/apps/qwen3_omni/config.yaml``) — a variant is
the app config plus the placement/streaming overlay registered below, so
buffer names and model settings stay identical across variants and the
frontend attaches to whichever backend is running. Run one backend at a
time: launch a variant, wait for the ``Qwen3-Omni backend READY`` line,
then start the frontend. Ctrl-C stops the backend.

A variant that uses n GPUs runs on CUDA devices 0..n-1 (stage placement
comes from the overlay's per-stage GPU index).

Usage:
  python -m wllm.apps.qwen3_omni.backend.rocm.launch --variant full_stream_tuned_chunks
  python -m wllm.apps.qwen3_omni.backend.rocm.launch --list

The scripts under ``examples/qwen3_omni/`` wrap the recommended variants
per GPU budget. Engine startup takes a couple of minutes before the READY
line appears.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import sys
import tempfile

import yaml

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
# The vLLM Talker CUDAGraph replay races with its async input copies on gfx950
# and faults the queue. Eager decode is the supported configuration here; set
# this to 0 explicitly to try the graphs.
os.environ.setdefault("WLLM_DISABLE_TALKER_CUDAGRAPH", "1")

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")
_THINKER_TP2 = "wllm/apps/qwen3_omni/backend/configs/stage_configs/thinker_only_tp2.yaml"

# Each entry: GPU count, which worker to run, the overlay merged onto the
# flattened app config, and a summary shown by --list.
#
# The dominant lever is chunking Code2Wav: vocoding a growing prefix instead of
# the whole response cuts time-to-first-audio by more than an order of
# magnitude. Streaming Thinker->Talker alone, dedicating a GPU per stage, and
# Thinker tensor parallelism were each measured to change nothing on their own.
REGISTRY = {
    "reference": dict(
        gpus=2, worker="reference", overlay={},
        summary="sequential reference: audio bursts out after the whole response is vocoded"),
    "sequential": dict(
        gpus=2, worker="streaming",
        overlay={"pipeline_mode": "sequential",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 1},
        summary="the streaming worker in reference schedule (plumbing control)"),
    "stream_thinker_talker": dict(
        gpus=2, worker="streaming",
        overlay={"pipeline_mode": "stream_thinker_talker",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 1},
        summary="Talker consumes Thinker output incrementally; audio still bursts (lever isolation)"),
    "stream_talker_c2w": dict(
        gpus=2, worker="streaming",
        overlay={"pipeline_mode": "stream_talker_c2w",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 1},
        summary="vocode codec frames in chunks as they arrive (the lever that matters)"),
    "full_stream": dict(
        gpus=2, worker="streaming",
        overlay={"pipeline_mode": "full_stream",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 1},
        summary="all three stages overlapped, Talker and Code2Wav sharing a GPU (2-GPU pick)"),
    "dedicated_gpus": dict(
        gpus=3, worker="streaming",
        overlay={"pipeline_mode": "sequential",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 2},
        summary="sequential schedule with a GPU per stage (measured: no effect)"),
    "full_stream_dedicated": dict(
        gpus=3, worker="streaming",
        overlay={"pipeline_mode": "full_stream",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 2},
        summary="full streaming with a GPU per stage"),
    "c2w_small_first_chunk": dict(
        gpus=3, worker="streaming",
        overlay={"pipeline_mode": "full_stream",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 2,
                 "first_chunk_frames": 6},
        summary="full streaming with a 6-frame first vocode chunk instead of 25"),
    # Developed under the name "combined_best": full streaming, a GPU per
    # stage, and a tuned vocode schedule (6-frame first chunk, then 12).
    "full_stream_tuned_chunks": dict(
        gpus=3, worker="streaming",
        overlay={"pipeline_mode": "full_stream",
                 "thinker_gpu": 0, "talker_gpu": 1, "c2w_gpu": 2,
                 "first_chunk_frames": 6, "codec_chunk_frames": 12},
        summary="full streaming, a GPU per stage, tuned vocode chunks (best overall, 3-GPU pick)"),
    # Tensor-parallel probes. Both were measured to gain nothing: the Thinker is
    # not the binding stage once Code2Wav is chunked. Kept as launchable evidence.
    "thinker_tp2": dict(
        gpus=3, worker="streaming",
        overlay={"pipeline_mode": "sequential",
                 "thinker_gpu": 0, "thinker_tp": 2, "talker_gpu": 2, "c2w_gpu": 2,
                 "thinker_stage_configs_path": _THINKER_TP2},
        summary="Thinker tensor-parallel over 2 GPUs, sequential schedule (measured: no win)"),
    "full_stream_dedicated_thinker_tp2": dict(
        gpus=4, worker="streaming",
        overlay={"pipeline_mode": "full_stream",
                 "thinker_gpu": 0, "thinker_tp": 2, "talker_gpu": 2, "c2w_gpu": 3,
                 "thinker_stage_configs_path": _THINKER_TP2},
        summary="full streaming plus Thinker TP=2 (measured: no win over the 3-GPU variants)"),
}


def _flatten_app_config(data: dict) -> dict:
    """The app config groups settings per component; the streaming worker takes
    a flat schema. Translate one into the other so both read the same file."""
    nested = ("thinker", "talker", "code2wav", "sampling")
    flat = {k: v for k, v in data.items() if k not in nested}
    thinker = data.get("thinker") or {}
    talker = data.get("talker") or {}
    code2wav = data.get("code2wav") or {}
    flat["model_path"] = thinker.get("model_path")
    flat["thinker_stage_configs_path"] = thinker.get("stage_configs_path")
    flat["code2wav_stage_configs_path"] = code2wav.get("stage_configs_path")
    flat["thinker_gpu"] = thinker.get("gpu_index", 0)
    flat["talker_gpu"] = talker.get("gpu_index", 1)
    flat["c2w_gpu"] = code2wav.get("gpu_index", 1)
    flat.update(data.get("sampling") or {})
    return flat


def _derive_config(base_path: str, overlay: dict) -> str:
    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    flat = _flatten_app_config(data)
    flat.update(overlay)
    fd, path = tempfile.mkstemp(prefix="qwen3_omni_cfg_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(flat, f)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Qwen3-Omni backend variant for the frontend.")
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

    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {spec['gpus']} (CUDA devices 0..{spec['gpus'] - 1})")
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        if spec["overlay"]:
            print(f"[launch] overlay = {spec['overlay']}")
        return

    # Relative paths in the config (stage configs) resolve against the repo root.
    os.chdir(_REPO_ROOT)

    if spec["worker"] == "reference":
        argv = [sys.executable, "-u", "-m", "wllm.apps.qwen3_omni.reference.launch_backend"]
    else:
        derived = _derive_config(args.config, spec["overlay"])
        print(f"[launch] derived config: {derived}")
        argv = [sys.executable, "-u", "-m",
                "wllm.apps.qwen3_omni.backend.rocm.streaming.worker", "--cfg", derived]

    print("[launch] starting; wait for 'Qwen3-Omni backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    os.execvpe(argv[0], argv, os.environ.copy())


if __name__ == "__main__":
    main()
