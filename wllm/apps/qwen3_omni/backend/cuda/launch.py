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
comes from the overlay's per-stage ``gpu_index``).

Usage:
  python -m wllm.apps.qwen3_omni.backend.cuda.launch --variant stream_full_windowed_threaded
  python -m wllm.apps.qwen3_omni.backend.cuda.launch --list

The scripts under ``examples/qwen3_omni/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import subprocess
import sys
import tempfile

import yaml

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")
_STAGE_CFGS = "wllm/apps/qwen3_omni/backend/configs/stage_configs"

_FULL = dict(stream_thinker_talker=True, stream_c2w=True, c2w_first_emit_frames=25,
             c2w_emit_interval_frames=25, c2w_lookahead_frames=0, c2w_context_frames=-1)

# Each entry: GPU count, worker kind, and the overlay merged onto the app
# config (stage gpu_index values are CUDA device indices 0..n-1; the app
# config's defaults are Thinker on 0, Talker+Code2Wav on 1).
REGISTRY = {
    "reference": dict(
        gpus=2, worker="reference", overlay={},
        summary="sequential reference: burst output after the full response is vocoded"),
    "stream_c2w": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "stream_thinker_talker": False}},
        summary="stream Talker->Code2Wav: vocode a growing prefix, emit audio incrementally"),
    "stream_c2w_windowed": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "stream_thinker_talker": False, "c2w_context_frames": 200}},
        summary="stream_c2w with a 200-frame left context bounding the O(N^2) vocode"),
    "stream_thinker_talker": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {"stream_thinker_talker": True, "stream_c2w": False}},
        summary="stream Thinker->Talker only; audio still bursts at the end (lever isolation)"),
    "stream_full": dict(
        gpus=2, worker="streaming", overlay={"streaming": dict(_FULL)},
        summary="both streaming edges: ~constant first-audio latency, real-time stream"),
    "stream_full_small_first_chunk": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "c2w_first_emit_frames": 8}},
        summary="stream_full with an 8-frame first vocode chunk (no effect; probe)"),
    "stream_full_chunk50": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "c2w_first_emit_frames": 50, "c2w_emit_interval_frames": 50}},
        summary="stream_full with 50-frame chunks: smoother but higher latency (tradeoff probe)"),
    "stream_full_threaded": dict(
        gpus=3, worker="streaming",
        overlay={"code2wav": {"gpu_index": 2}, "streaming": {**_FULL, "c2w_threaded": True}},
        summary="stream_full + concurrent Code2Wav pump on a dedicated GPU (rate margin win)"),
    "spread_devices": dict(
        gpus=3, worker="streaming",
        overlay={"code2wav": {"gpu_index": 2}, "streaming": dict(_FULL)},
        summary="stream_full with each stage on its own GPU (measured: no effect)"),
    # The stream_full_windowed family was developed under the name
    # "combined_best": full streaming + a bounded 400-frame Code2Wav left
    # context that caps the vocoder cost for arbitrarily long responses.
    "stream_full_windowed": dict(
        gpus=3, worker="streaming",
        overlay={"code2wav": {"gpu_index": 2}, "streaming": {**_FULL, "c2w_context_frames": 400}},
        summary="stream_full + bounded 400-frame context + dedicated GPU per stage"),
    "stream_full_windowed_threaded": dict(
        gpus=3, worker="streaming",
        overlay={"code2wav": {"gpu_index": 2},
                 "streaming": {**_FULL, "c2w_context_frames": 400, "c2w_threaded": True}},
        summary="stream_full_windowed + the threaded Code2Wav pump (best overall, 3-GPU pick)"),
    "stream_full_windowed_2gpu": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "c2w_context_frames": 400}},
        summary="stream_full_windowed with Code2Wav co-located on the Talker GPU"),
    "stream_full_windowed_threaded_2gpu": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {**_FULL, "c2w_context_frames": 400, "c2w_threaded": True}},
        summary="threaded windowed streaming on 2 GPUs (best 2-GPU pick)"),
    # Tensor-parallel probes. All three were measured latency REGRESSIONS on a
    # PCIe-only node (per-layer all-reduce dwarfs the compute); kept as
    # launchable evidence. On NVLink machines they run but still lose to the
    # streaming variants.
    "thinker_tp2": dict(
        gpus=2, worker="streaming",
        overlay={"streaming": {
            "stream_thinker_talker": False, "stream_c2w": False,
            "talker_enforce_eager": True,
            "thinker_visible_devices": "INHERIT",
            "thinker_stage_configs_path": f"{_STAGE_CFGS}/thinker_tp2.yaml",
            "c2w_visible_devices": "INHERIT",
            "c2w_stage_configs_path": f"{_STAGE_CFGS}/code2wav_dev1.yaml"}},
        summary="Thinker tensor-parallel over 2 GPUs (measured regression on PCIe nodes)"),
    "thinker_tp4": dict(
        gpus=4, worker="streaming",
        overlay={"talker": {"gpu_index": 3}, "code2wav": {"gpu_index": 3},
                 "streaming": {
                     "stream_thinker_talker": False, "stream_c2w": False,
                     "talker_enforce_eager": True,
                     "thinker_visible_devices": "INHERIT",
                     "thinker_stage_configs_path": f"{_STAGE_CFGS}/thinker_tp4.yaml",
                     "c2w_visible_devices": "INHERIT",
                     "c2w_stage_configs_path": f"{_STAGE_CFGS}/code2wav_dev3.yaml"}},
        summary="Thinker tensor-parallel over 4 GPUs (measured regression on PCIe nodes)"),
    # talker_max_tokens is capped at 4 here because that is the configuration
    # the variant was measured with (each TP talker frame is extremely slow on
    # PCIe); raise it to run full responses.
    "talker_tp2": dict(
        gpus=3, worker="talker_tp",
        overlay={"sampling": {"talker_max_tokens": 4}, "code2wav": {"gpu_index": 0},
                 "streaming": {"stream_thinker_talker": False, "stream_c2w": False,
                               "talker_tp_size": 2, "talker_tp_devices": "1,2",
                               "thinker_visible_devices": "0"}},
        summary="Talker tensor-parallel over 2 GPUs (measured severe regression on PCIe nodes)"),
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _derive_config(base_path: str, name: str) -> str:
    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _deep_merge(data, REGISTRY[name]["overlay"])
    fd, path = tempfile.mkstemp(prefix="qwen3_omni_cfg_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
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

    # Stage placement uses device indices from the derived config.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    print(f"[launch] variant = {args.variant}")
    devs = "CUDA device 0" if spec["gpus"] == 1 else f"CUDA devices 0..{spec['gpus'] - 1}"
    print(f"[launch] gpus    = {spec['gpus']} ({devs})")
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        return

    derived = _derive_config(args.config, args.variant)
    print(f"[launch] derived config: {derived}")
    print("[launch] starting; wait for 'Qwen3-Omni backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    # stage_configs paths in the config are repo-root relative.
    os.chdir(_REPO_ROOT)

    if spec["worker"] == "talker_tp":
        # The TP driver spawns its follower ranks and must own CUDA_VISIBLE_DEVICES
        # before torch is imported, so it runs as a subprocess by design.
        script = os.path.join(_BACKEND_DIR, "talker_tp", "launch_talker_tp.py")
        raise SystemExit(subprocess.call([sys.executable, "-u", script, derived]))
    if spec["worker"] == "reference":
        from wllm.apps.qwen3_omni.reference.worker import Qwen3OmniWorker
        Qwen3OmniWorker(cfg_path=derived).loop()
        return
    from wllm.apps.qwen3_omni.backend.cuda.streaming.worker import StreamingWorker
    StreamingWorker(derived).loop()


if __name__ == "__main__":
    main()
