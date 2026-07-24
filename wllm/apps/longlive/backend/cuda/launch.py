"""Launch a LongLive backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/longlive/adapter.py``) and reads the same app config
(``wllm/apps/longlive/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``LongLive backend READY`` line from rank 0, then start the
frontend. Ctrl-C stops the backend and all its rank/sidecar processes.

A variant that uses n GPUs runs on CUDA devices 0..n-1: the DiT SP
ranks first, then the VAE group (disaggregated variants), then the
async ASR sidecar.

Usage:
  python -m wllm.apps.longlive.backend.cuda.launch --variant combined_sp4_vae2
  python -m wllm.apps.longlive.backend.cuda.launch --list

The scripts under ``examples/longlive/`` wrap the recommended variants
per GPU budget. The opts-JSON spawner (``backend/spawn.py``) remains
available for custom topologies.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root

from wllm.apps.longlive.backend.cuda.variants import VARIANTS, gpus_needed

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")


def build_opts(name: str) -> dict:
    """Map a variant's topology onto CUDA devices 0..n-1 (DiT ranks, then the
    VAE group, then the ASR sidecar), in the spawner's opts format."""
    v = VARIANTS[name]
    if v["kind"] == "mono":
        world = v["dit"]
        if v.get("asr") == "async":
            # sidecar shares the rank processes' visible set; its device is the
            # extra GPU after the DiT ranks
            visible = ",".join(str(g) for g in range(world + 1))
            return dict(world=world, visible_devices=visible, vae_mode=v["vae"],
                        asr_mode="async", asr_device=f"cuda:{world}")
        visible = ",".join(str(g) for g in range(world))
        return dict(world=world, visible_devices=visible, vae_mode=v["vae"],
                    asr_mode="sync", asr_device="cuda:0")
    dit, vae = v["dit"], v["vae_ranks"]
    opts = dict(kind="disagg", dit_world=dit, vae_world=vae,
                dit_visible=",".join(str(g) for g in range(dit)),
                vae_visible=",".join(str(g) for g in range(dit, dit + vae)),
                asr_mode=v.get("asr", "sync"), asr_device="cuda:0")
    if v.get("asr") == "async":
        opts["asr_visible"] = str(dit + vae)
    return opts


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a LongLive backend variant for the frontend.")
    ap.add_argument("--variant", help="one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the launch plan, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in VARIANTS) + 2
        for name, spec in VARIANTS.items():
            n = gpus_needed(name)
            print(f"{name:<{width}} {n} GPU{'s' if n > 1 else ' '}  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant not in VARIANTS:
        ap.error(f"unknown variant '{args.variant}'; see --list")

    n = gpus_needed(args.variant)
    # Rank/sidecar placement uses device indices 0..n-1 from the opts.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    print(f"[launch] variant = {args.variant}")
    devs = "CUDA device 0" if n == 1 else f"CUDA devices 0..{n - 1}"
    print(f"[launch] gpus    = {n} ({devs})")
    print(f"[launch] config  = {args.config}", flush=True)
    if args.dry_run:
        if VARIANTS[args.variant]["kind"] != "reference":
            print(f"[launch] opts    = {build_opts(args.variant)}")
        return

    print("[launch] starting; wait for the 'LongLive backend READY' line (rank=0), "
          "then start the frontend (Ctrl-C stops it)", flush=True)
    os.chdir(_REPO_ROOT)

    if VARIANTS[args.variant]["kind"] == "reference":
        from wllm.apps.longlive.reference.worker import LongLiveWorker
        worker = None
        try:
            worker = LongLiveWorker(cfg_path=args.config)
            worker.loop()
        finally:
            if worker is not None:
                worker.terminate()
        return

    from wllm.apps.longlive.backend.cuda import spawn
    spawn.run(args.config, build_opts(args.variant))


if __name__ == "__main__":
    main()
