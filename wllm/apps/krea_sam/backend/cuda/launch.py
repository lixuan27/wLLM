"""Launch a Krea-Realtime + SAM3 backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/krea_sam/adapter.py``) and reads the same app config
(``wllm/apps/krea_sam/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``KreaSAM backend READY`` line, then start the frontend. Ctrl-C
stops the backend and its service processes.

A variant that uses n GPUs runs on CUDA devices 0..n-1: SAM on device 0
and the Krea services on the rest (for the VAE/DiT-split variants the
first Krea device is the VAE service, the remaining ones the DiT).
Placement is managed internally with physical device indices, so
remapping via CUDA_VISIBLE_DEVICES is not supported; the launcher
clears it.

Usage:
  python -m wllm.apps.krea_sam.backend.cuda.launch --variant combined_sp3_compile_stream
  python -m wllm.apps.krea_sam.backend.cuda.launch --list

The scripts under ``examples/krea_sam/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import importlib
import os
from wllm.serving.paths import app_dir, repo_root
import tempfile

import yaml

from wllm.apps.krea_sam.backend.cuda.variants import VARIANTS, total_gpus

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")


def _assignment(name: str) -> dict:
    spec = VARIANTS[name]
    if spec["engine"] == "reference":
        return {}
    n_krea = spec["n_krea"]
    if spec.get("colocate", False):
        return {"sam_gpu": 0, "krea_gpus": list(range(n_krea))}
    return {"sam_gpu": 0, "krea_gpus": list(range(1, 1 + n_krea))}


def _derive_config(base_path: str, name: str) -> str:
    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    spec = VARIANTS[name]
    if spec["engine"] != "reference":
        knobs = dict(spec.get("knobs", {}))
        knobs.update(_assignment(name))
        knobs["variant_name"] = name
        data["backend"] = knobs
    fd, path = tempfile.mkstemp(prefix="krea_sam_cfg_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def _worker_class(name: str):
    if VARIANTS[name]["engine"] == "reference":
        mod = importlib.import_module("wllm.apps.krea_sam.reference.worker")
        return mod.KreaSAMWorker
    if name == "sam_colocate":
        # Standalone single-process worker (SAM and Krea on two threads of one
        # GPU); the coordinator's two-process form over-subscribes a single GPU.
        mod = importlib.import_module("wllm.apps.krea_sam.backend.cuda.sam_colocate.worker")
        return mod.Worker
    mod = importlib.import_module("wllm.apps.krea_sam.backend.cuda.engine.coordinator")

    class _Worker(mod.KreaSamCoordinator):
        VARIANT = name

    return _Worker


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Krea-Realtime + SAM3 backend variant.")
    ap.add_argument("--variant", help="one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the launch plan, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in VARIANTS) + 2
        for name, spec in VARIANTS.items():
            print(f"{name:<{width}} {total_gpus(name)} GPU{'s' if total_gpus(name) > 1 else ' '}  {spec['hypothesis']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant not in VARIANTS:
        ap.error(f"unknown variant '{args.variant}'; see --list")

    n = total_gpus(args.variant)
    assign = _assignment(args.variant)
    # Service placement uses physical device indices, so the variant must see
    # the machine's devices unremapped.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    print(f"[launch] variant = {args.variant}")
    devs = "CUDA device 0" if n == 1 else f"CUDA devices 0..{n - 1}"
    print(f"[launch] gpus    = {n} ({devs})"
          + (f", SAM on {assign['sam_gpu']}, Krea on {assign['krea_gpus']}" if assign else ""))
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        return

    derived = _derive_config(args.config, args.variant)
    print(f"[launch] derived config: {derived}")
    print("[launch] starting; wait for 'KreaSAM backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    os.chdir(_REPO_ROOT)

    cls = _worker_class(args.variant)
    worker = None
    try:
        worker = cls(cfg_path=derived)
        worker.loop()
    finally:
        # Runs on Ctrl+C, graceful TERM, or any exception, so service
        # subprocesses and shm buffers are cleaned up instead of orphaned.
        if worker is not None and hasattr(worker, "terminate"):
            worker.terminate()


if __name__ == "__main__":
    main()
