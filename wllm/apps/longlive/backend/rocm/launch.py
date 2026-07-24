"""Launch a LongLive backend variant.

Every variant honors the shared adapter contract
(``wllm/apps/longlive/adapter.py``) and reads the same app config
(``wllm/apps/longlive/config.yaml``), including the shared-memory
buffer names, so the frontend attaches to whichever backend is running
with no changes. Run one backend at a time: launch a variant, wait for
the ``LongLive backend READY`` line, then start the frontend. Ctrl-C
stops the backend and its ranks.

A variant that uses n GPUs runs on CUDA devices 0..n-1. On the split
topologies the DiT ranks come first and the VAE ranks follow. To run on
other devices, export ``CUDA_VISIBLE_DEVICES`` before launching.

Usage:
  python -m wllm.apps.longlive.backend.rocm.launch --variant unified_sp4
  python -m wllm.apps.longlive.backend.rocm.launch --list

The scripts under ``examples/longlive/`` wrap the recommended variants
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
_B = "wllm.apps.longlive.backend.rocm"

# Each entry: GPU count, how to run it, and a summary shown by --list.
#   worker=  -> a single-process worker class run by run_worker.py
#   sp_main= -> a variant entry point run on every rank under torchrun
#
# Sequence parallelism shards the 8-latent chunk, so SP must divide 8. The
# co-located variants run DiT then VAE on the same ranks; the pipelined ones
# put the VAE on its own ranks so it overlaps the next chunk's DiT.
REGISTRY = {
    "reference": dict(
        gpus=1, reference=True,
        summary="sequential reference on one GPU"),
    "baseline_single": dict(
        gpus=1, worker=f"{_B}.baseline_single.worker:Worker",
        summary="the agent's core on one GPU (reference-equivalent control)"),
    "decode_first": dict(
        gpus=1, worker=f"{_B}.decode_first.worker:Worker",
        summary="decode frame 0 before the cache-write pass (1-GPU latency pick)"),
    "async_asr": dict(
        gpus=2, worker=f"{_B}.async_asr.worker:Worker",
        env={"LL_ASR_DEVICE": "cuda:1"},
        summary="ASR on its own GPU, prompts applied without stalling generation"),
    "dit_sp2": dict(
        gpus=2, sp_main=f"{_B}.dit_sp2.worker:main",
        summary="DiT sequence-parallel over 2 GPUs (2-GPU latency pick)"),
    "dit_vae_pipeline": dict(
        gpus=2, sp_main=f"{_B}.dit_vae_pipeline.worker:main",
        summary="DiT on one GPU, VAE on the other, pipelined (2-GPU frame-rate pick)"),
    "dit_sp4": dict(
        gpus=4, sp_main=f"{_B}.dit_sp4.worker:main",
        summary="DiT sequence-parallel over 4 GPUs, VAE on rank 0"),
    "vae_tile4": dict(
        gpus=4, sp_main=f"{_B}.vae_tile4.worker:main",
        summary="VAE width-tiled over 4 GPUs, DiT replicated (isolates the VAE lever)"),
    "unified_sp4": dict(
        gpus=4, sp_main=f"{_B}.unified_sp4.worker:main",
        summary="one 4-GPU group running DiT SP and VAE tiling per chunk (4-GPU pick)"),
    "pipeline_dit4_vae1": dict(
        gpus=5, sp_main=f"{_B}.pipeline_dit4_vae1.worker:main",
        summary="DiT SP=4 alongside a single-GPU VAE, pipelined"),
    "dit_sp8": dict(
        gpus=8, sp_main=f"{_B}.dit_sp8.worker:main",
        summary="DiT sequence-parallel over 8 GPUs (lowest measured chunk latency)"),
    "pipeline_dit4_vae4": dict(
        gpus=8, sp_main=f"{_B}.pipeline_dit4_vae4.worker:main",
        summary="DiT SP=4 alongside a 4-GPU tiled VAE, pipelined (8-GPU pick)"),
    # Developed under the name "combined_best": the pipelined topology with the
    # async ASR stacked on. Measured slower than pipeline_dit4_vae4 on both
    # latency metrics -- the async hand-off costs about a chunk of wait.
    "pipeline_dit4_vae4_async_asr": dict(
        gpus=8, sp_main=f"{_B}.pipeline_dit4_vae4_async_asr.worker:main",
        env={"LL_ASR_DEVICE": "cuda:0"},
        summary="pipeline_dit4_vae4 plus off-critical-path ASR"),
}

# torch.distributed environment inherited from an outer launcher would fight
# with the world the ranks stand up, so scrub it from the child env.
_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS",
    "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)


def _build_argv(variant: str, cfg: str) -> list[str]:
    spec = REGISTRY[variant]
    if spec.get("reference"):
        return [sys.executable, "-u", "-m", "wllm.apps.longlive.reference.launch_backend"]
    if "worker" in spec:
        return [sys.executable, "-u", "-m", f"{_B}.run_worker",
                "--worker", spec["worker"], "--cfg", cfg]
    # --standalone lets torchrun pick its own free rendezvous port.
    return [sys.executable, "-u", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", str(spec["gpus"]),
            os.path.join(_BACKEND_DIR, "run_worker_dist.py"),
            "--sp-main", spec["sp_main"], "--cfg", cfg]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a LongLive backend variant for the frontend.")
    ap.add_argument("--variant", help="one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in REGISTRY) + 2
        for name, spec in REGISTRY.items():
            n = spec["gpus"]
            print(f"{name:<{width}} {n} GPU{'s' if n > 1 else ' '}  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant not in REGISTRY:
        ap.error(f"unknown variant '{args.variant}'; see --list")
    spec = REGISTRY[args.variant]

    env = os.environ.copy()
    for var in _DIST_VARS:
        env.pop(var, None)
    env.update(spec.get("env", {}))
    env["PYTHONUNBUFFERED"] = "1"

    argv = _build_argv(args.variant, args.config)
    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {spec['gpus']} (CUDA devices 0..{spec['gpus'] - 1}; "
          "export CUDA_VISIBLE_DEVICES to remap)")
    print(f"[launch] config  = {args.config}")
    print("[launch] command = " + " ".join(argv), flush=True)
    if args.dry_run:
        return
    print("[launch] starting; wait for 'LongLive backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)
    # Relative paths in the config (assets, checkpoints) resolve against the repo root.
    os.chdir(_REPO_ROOT)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
