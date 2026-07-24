"""Launch a WorldPlay backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/worldplay/adapter.py``) and reads the same app config
(``wllm/apps/worldplay/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``WorldPlay backend READY`` line, then start the frontend. Ctrl-C
stops the backend.

A variant that uses n GPUs runs on CUDA devices 0..n-1. To run on other
devices, export ``CUDA_VISIBLE_DEVICES`` before launching.

Usage:
  python -m wllm.apps.worldplay.backend.cuda.launch --variant sp4_stream
  python -m wllm.apps.worldplay.backend.cuda.launch --list

The scripts under ``examples/worldplay/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import socket
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")
_SP_WORKER = os.path.join(_BACKEND_DIR, "runtime", "sp_worker.py")
_STAGE_WORKER = os.path.join(_BACKEND_DIR, "runtime", "stage_split_worker.py")

# Each entry: worker script, number of ranks (= GPUs), extra worker flags,
# and a one-line summary shown by --list.
REGISTRY = {
    "sp1": dict(
        worker=_SP_WORKER, nproc=1, extra=[],
        summary="reference pipeline on the multi-rank worker, 1 rank (worker-overhead control)"),
    "vae_stream": dict(
        worker=_SP_WORKER, nproc=1, extra=["--stream-vae"],
        summary="per-latent VAE streaming on 1 GPU (lowest single-GPU latency)"),
    "vae_batch": dict(
        worker=_SP_WORKER, nproc=1, extra=["--vae-batch"],
        summary="whole-chunk batched VAE decode on 1 GPU (analysis variant)"),
    # Ulysses sequence parallelism splits the 4-frame generate step across
    # ranks, so SP must divide 4: SP is limited to 2 or 4. Larger GPU counts
    # scale through the stage-split variants instead.
    "dit_sp2": dict(
        worker=_SP_WORKER, nproc=2, extra=[],
        summary="sequence-parallel DiT over 2 GPUs"),
    "sp2_stream": dict(
        worker=_SP_WORKER, nproc=2, extra=["--stream-vae"],
        summary="SP=2 DiT plus per-latent VAE streaming (2-GPU latency pick)"),
    "dit_sp4": dict(
        worker=_SP_WORKER, nproc=4, extra=[],
        summary="sequence-parallel DiT over 4 GPUs"),
    "sp4_stream": dict(
        worker=_SP_WORKER, nproc=4, extra=["--stream-vae"],
        summary="SP=4 DiT plus per-latent VAE streaming (4-GPU latency pick)"),
    "sp4_vae_rank0": dict(
        worker=_SP_WORKER, nproc=4, extra=["--vae-mode", "rank0"],
        summary="SP=4 DiT with the VAE decode pinned to rank 0 (placement probe)"),
    # Stage split: a DiT rank group and a VAE rank group run as two pipelined
    # stages (VAE of chunk N overlaps DiT of chunk N+1). The DiT group is
    # sequence-parallel over itself; the VAE group width-tiles the decode.
    "stage_split_2g": dict(
        worker=_STAGE_WORKER, nproc=2,
        extra=["--dit-ranks", "1", "--vae-ranks", "1", "--stream-vae"],
        summary="DiT stage + VAE stage pipelined across 2 GPUs (2-GPU throughput pick)"),
    "stage_split_sp_3g": dict(
        worker=_STAGE_WORKER, nproc=3,
        extra=["--dit-ranks", "2", "--vae-ranks", "1", "--stream-vae"],
        summary="SP=2 DiT stage + 1-GPU VAE stage, pipelined"),
    "stage_split_sp_4g": dict(
        worker=_STAGE_WORKER, nproc=4,
        extra=["--dit-ranks", "2", "--vae-ranks", "2", "--stream-vae"],
        summary="SP=2 DiT stage + 2-GPU tiled VAE stage, pipelined (4-GPU throughput pick)"),
    "stage_split_sp_6g": dict(
        worker=_STAGE_WORKER, nproc=6,
        extra=["--dit-ranks", "4", "--vae-ranks", "2", "--stream-vae"],
        summary="SP=4 DiT stage + 2-GPU tiled VAE stage, pipelined (6-GPU pick)"),
    "stage_split_sp_7g": dict(
        worker=_STAGE_WORKER, nproc=7,
        extra=["--dit-ranks", "4", "--vae-ranks", "3", "--stream-vae"],
        summary="SP=4 DiT stage + 3-GPU tiled VAE stage, pipelined (7-GPU pick)"),
    "stage_split_sp_8g": dict(
        worker=_STAGE_WORKER, nproc=8,
        extra=["--dit-ranks", "4", "--vae-ranks", "4", "--stream-vae"],
        summary="SP=4 DiT stage + 4-GPU tiled VAE stage, pipelined (8-GPU pick)"),
}

# torch.distributed environment inherited from an outer launcher would fight
# with the torchrun world the workers stand up, so scrub it from the child env.
_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS",
    "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build_argv(variant: str, cfg: str, port: int) -> tuple[list[str], int]:
    if variant == "reference":
        return [sys.executable, "-u", "-m", "wllm.apps.worldplay.reference.launch_backend"], 1
    spec = REGISTRY[variant]
    argv = [
        sys.executable, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(spec["nproc"]),
        "--master_addr", "127.0.0.1", "--master_port", str(port),
        spec["worker"], "--cfg", cfg,
    ] + spec["extra"]
    return argv, spec["nproc"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a WorldPlay backend variant for the frontend.")
    ap.add_argument("--variant", help="reference, or one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--port", type=int, default=0, help="torchrun master port (0 = pick a free one)")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in REGISTRY) + 2
        print(f"{'reference':<{width}} 1 GPU   unmodified sequential reference backend")
        for name, spec in REGISTRY.items():
            print(f"{name:<{width}} {spec['nproc']} GPU{'s' if spec['nproc'] > 1 else ' '}  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant != "reference" and args.variant not in REGISTRY:
        ap.error(f"unknown variant '{args.variant}'; see --list")

    argv, need = _build_argv(args.variant, args.config, args.port or _find_free_port())

    env = os.environ.copy()
    for var in _DIST_VARS:
        env.pop(var, None)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {need} (CUDA devices 0..{need - 1}; export CUDA_VISIBLE_DEVICES to remap)")
    print(f"[launch] config  = {args.config}")
    print(f"[launch] command = " + " ".join(argv), flush=True)
    if args.dry_run:
        return
    print("[launch] starting; wait for 'WorldPlay backend READY', then start the frontend (Ctrl-C stops it)",
          flush=True)
    # Relative paths in the config (image_path, assets) resolve against the repo root.
    os.chdir(_REPO_ROOT)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
