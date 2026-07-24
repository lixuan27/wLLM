"""Launch a Krea-Realtime + SAM3 backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/krea_sam/adapter.py``) and reads the same app config
(``wllm/apps/krea_sam/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``KreaSAM backend READY`` line, then start the frontend. Ctrl-C
stops the backend and the processes it spawned.

A variant that uses n GPUs runs on CUDA devices 0..n-1: the Krea ranks
first, then SAM on the last device. Placement is passed down as physical
device indices, so remapping via CUDA_VISIBLE_DEVICES is not supported;
the launcher clears it.

Usage:
  python -m wllm.apps.krea_sam.backend.rocm.launch --variant stream_sp3_compiled
  python -m wllm.apps.krea_sam.backend.rocm.launch --list

The scripts under ``examples/krea_sam/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")

_SAM_PARALLEL = "wllm.apps.krea_sam.backend.rocm.sam_parallel.launch"
_MGPU = "wllm.apps.krea_sam.backend.rocm.engine.launch_mgpu"

# Each entry: which launcher to run, the Krea rank devices, the SAM device,
# launcher flags, extra env, and a summary shown by --list. SAM always takes
# the last device, so a variant occupies devices 0..n-1 with no gaps.
#
# The pipeline is SAM-bound: parallelizing the Krea side (DiT sequence
# parallelism, VAE width-tiling, the intra-Krea pipeline split) buys latency and
# smoothness rather than frame rate, and compiling SAM's backbone
# (KREA_SAM_COMPILE) is what moves throughput.
REGISTRY = {
    "reference": dict(
        entry=None, krea_gpus=[0], sam_gpu=None, flags=[], env={},
        summary="sequential reference: Krea and SAM share one GPU, one chunk at a time"),
    "sam_parallel": dict(
        entry=_SAM_PARALLEL, krea_gpus=[0], sam_gpu=1, flags=[], env={},
        summary="SAM decoupled onto its own GPU, running concurrently with Krea"),
    "sam_compiled": dict(
        entry=_SAM_PARALLEL, krea_gpus=[0], sam_gpu=1, flags=[],
        env={"KREA_SAM_COMPILE": "1"},
        summary="sam_parallel with SAM's hot submodules compiled"),
    "frame_stream": dict(
        entry=_SAM_PARALLEL, krea_gpus=[0], sam_gpu=1, flags=["--stream"], env={},
        summary="emit each frame as its mask arrives instead of a whole chunk"),
    "frame_stream_compiled": dict(
        entry=_SAM_PARALLEL, krea_gpus=[0], sam_gpu=1, flags=["--stream"],
        env={"KREA_SAM_COMPILE": "1"},
        summary="per-frame emit plus compiled SAM (2-GPU pick)"),
    "krea_pipeline": dict(
        entry=_MGPU, krea_gpus=[0, 1], sam_gpu=2,
        flags=["--sp-size", "1", "--pipeline"], env={},
        summary="Krea split encode+denoise / decode+composite over 2 GPUs, pipelined (3-GPU pick)"),
    "vae_decode_tile": dict(
        entry=_MGPU, krea_gpus=[0, 1], sam_gpu=2, flags=["--sp-size", "1"], env={},
        summary="VAE decoder width-tiled over 2 GPUs, DiT replicated (isolates the VAE lever)"),
    "krea_dit_sp3": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2], sam_gpu=3, flags=[], env={},
        summary="Krea DiT sequence-parallel over 3 GPUs, whole-chunk emit"),
    "combined_sam_sp3": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2], sam_gpu=3, flags=[],
        env={"KREA_SAM_COMPILE": "1"},
        summary="DiT SP=3 plus compiled SAM, whole-chunk emit"),
    "stream_sp3": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2], sam_gpu=3, flags=["--stream"], env={},
        summary="DiT SP=3 plus per-frame emit"),
    "stream_sp3_compiled": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2], sam_gpu=3, flags=["--stream"],
        env={"KREA_SAM_COMPILE": "1"},
        summary="DiT SP=3, per-frame emit and compiled SAM (4-GPU pick)"),
    "sp3_vae_split": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2, 3], sam_gpu=4,
        flags=["--sp-size", "3", "--sp-vae-split"], env={},
        summary="DiT SP=3 over 3 GPUs with VAE decode on a 4th, overlapping the next chunk"),
    "sp3_vae_split_compiled": dict(
        entry=_MGPU, krea_gpus=[0, 1, 2, 3], sam_gpu=4,
        flags=["--sp-size", "3", "--sp-vae-split"],
        env={"KREA_SAM_COMPILE": "1"},
        summary="DiT SP=3 with a split VAE plus compiled SAM (5-GPU pick, smoothest)"),
}


def gpus_needed(variant: str) -> int:
    spec = REGISTRY[variant]
    return len(spec["krea_gpus"]) + (1 if spec["sam_gpu"] is not None else 0)


def _build_argv(variant: str, cfg: str) -> list[str]:
    spec = REGISTRY[variant]
    if spec["entry"] is None:
        return [sys.executable, "-u", "-m", "wllm.apps.krea_sam.reference.launch_backend"]
    argv = [sys.executable, "-u", "-m", spec["entry"], cfg]
    if spec["entry"] == _SAM_PARALLEL:
        argv += ["--krea-gpu", str(spec["krea_gpus"][0])]
    else:
        argv += ["--krea-gpus", ",".join(str(g) for g in spec["krea_gpus"])]
    return argv + ["--sam-gpu", str(spec["sam_gpu"])] + spec["flags"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a Krea+SAM backend variant for the frontend.")
    ap.add_argument("--variant", help="one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in REGISTRY) + 2
        for name, spec in REGISTRY.items():
            n = gpus_needed(name)
            print(f"{name:<{width}} {n} GPU{'s' if n > 1 else ' '}  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant not in REGISTRY:
        ap.error(f"unknown variant '{args.variant}'; see --list")
    spec = REGISTRY[args.variant]

    env = os.environ.copy()
    # The launchers place Krea ranks and SAM by physical device index, so the
    # variant must see the machine's devices unremapped.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.update(spec["env"])
    env["PYTHONUNBUFFERED"] = "1"

    need = gpus_needed(args.variant)
    argv = _build_argv(args.variant, args.config)
    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {need} (CUDA devices 0..{need - 1})")
    if spec["sam_gpu"] is not None:
        print(f"[launch] placement = Krea on {','.join(str(g) for g in spec['krea_gpus'])}, "
              f"SAM on {spec['sam_gpu']}"
              + (", SAM compiled" if spec["env"].get("KREA_SAM_COMPILE") == "1" else ""))
    print(f"[launch] config  = {args.config}")
    print("[launch] command = " + " ".join(argv), flush=True)
    if args.dry_run:
        return
    print("[launch] starting; wait for 'KreaSAM backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    # Relative paths in the config (assets, checkpoints) resolve against the repo root.
    os.chdir(_REPO_ROOT)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
