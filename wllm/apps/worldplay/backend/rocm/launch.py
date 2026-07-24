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

Two backend architectures are available. The *co-located* variants put the
DiT and the VAE on the same GPUs, sequence-parallel and tile-parallel
respectively, which minimizes the latency of a single chunk. The
*disaggregated* variants put them in two processes on disjoint GPUs joined by
a shared-memory latent buffer, so chunk N's VAE decode overlaps chunk N+1's
DiT, which maximizes frame rate and smoothness.

Usage:
  python -m wllm.apps.worldplay.backend.rocm.launch --variant stream_colocated_sp4
  python -m wllm.apps.worldplay.backend.rocm.launch --list

The scripts under ``examples/worldplay/`` wrap the recommended variants
per GPU budget.
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import socket
import subprocess
import sys
import time

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = app_dir(__package__.split(".")[2])
_REPO_ROOT = repo_root()
_DEFAULT_CFG = os.path.join(_APP_DIR, "config.yaml")
_COLOCATED_WORKER = os.path.join(_BACKEND_DIR, "runtime", "colocated_worker.py")
_PIPE_DIT_WORKER = os.path.join(_BACKEND_DIR, "runtime", "pipe_dit_worker.py")
_PIPE_VAE_WORKER = os.path.join(_BACKEND_DIR, "runtime", "pipe_vae_worker.py")

# Names the two halves of a disaggregated backend use to find each other. They
# are internal to the backend and never touched by the frontend.
_SHM_PREFIX = "worldplay_pipe"

# Ulysses sequence parallelism shards the 4-frame generate step across ranks,
# so the DiT's SP degree must divide 4: it is limited to 1, 2 or 4. Extra GPUs
# past that go to the VAE, which tiles spatially instead.
#
# Co-located entries: ``nproc`` ranks in one world, DiT sequence-parallel over
# ``sp`` of them, VAE tiled over all of them unless --no-vae-tile.
# Disaggregated entries: ``dit`` DiT ranks (SP=``sp``) plus ``vae`` VAE ranks,
# as two processes; total GPUs is their sum.
REGISTRY = {
    "stream_frames": dict(
        kind="colocated", nproc=1, sp=1, extra=["--stream-vae"],
        summary="per-latent VAE streaming on 1 GPU (1-GPU latency pick)"),
    "colocated_sp2": dict(
        kind="colocated", nproc=2, sp=2, extra=[],
        summary="SP=2 DiT + 2-way tiled VAE, one frame batch per chunk"),
    # Measured for latency and throughput but, unlike its sp=1 and sp=4
    # siblings, not put through the correctness harness.
    "stream_colocated_sp2": dict(
        kind="colocated", nproc=2, sp=2, extra=["--stream-vae"],
        summary="SP=2 DiT + 2-way tiled VAE, per-latent streaming (2-GPU latency pick)"),
    "sp2_ditonly": dict(
        kind="colocated", nproc=2, sp=2, extra=["--no-vae-tile"],
        summary="SP=2 DiT with a full-frame VAE (isolates the DiT lever)"),
    "colocated_sp4": dict(
        kind="colocated", nproc=4, sp=4, extra=[],
        summary="SP=4 DiT + 4-way tiled VAE, one frame batch per chunk"),
    "stream_colocated_sp4": dict(
        kind="colocated", nproc=4, sp=4, extra=["--stream-vae"],
        summary="SP=4 DiT + 4-way tiled VAE, per-latent streaming (4-GPU latency pick)"),
    "sp4_ditonly": dict(
        kind="colocated", nproc=4, sp=4, extra=["--no-vae-tile"],
        summary="SP=4 DiT with a full-frame VAE (isolates the DiT lever)"),
    "vae_tile4": dict(
        kind="colocated", nproc=4, sp=1, extra=[],
        summary="replicated DiT + 4-way tiled VAE (isolates the VAE lever)"),
    "pipe_dit1_vae1": dict(
        kind="pipe", dit=1, sp=1, vae=1, vae_mode="stream",
        summary="1-GPU DiT || 1-GPU VAE, pipelined (2-GPU throughput pick)"),
    "pipe_dit2_vae2": dict(
        kind="pipe", dit=2, sp=2, vae=2, vae_mode="stream",
        summary="SP=2 DiT || 2-way tiled VAE, pipelined (4-GPU throughput pick)"),
    "pipe_dit4_vae1": dict(
        kind="pipe", dit=4, sp=4, vae=1, vae_mode="stream",
        summary="SP=4 DiT || full-frame VAE, pipelined (the VAE is the bottleneck)"),
    "pipe_dit4_vae2": dict(
        kind="pipe", dit=4, sp=4, vae=2, vae_mode="stream",
        summary="SP=4 DiT || 2-way tiled VAE, pipelined (6-GPU pick)"),
    "pipe_dit4_vae4": dict(
        kind="pipe", dit=4, sp=4, vae=4, vae_mode="stream",
        summary="SP=4 DiT || 4-way tiled VAE, pipelined (8-GPU pick)"),
    "pipe_dit4_vae4_batch": dict(
        kind="pipe", dit=4, sp=4, vae=4, vae_mode="batch",
        summary="as pipe_dit4_vae4 but one frame batch per chunk (isolates streaming)"),
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


def gpus_needed(variant: str) -> int:
    if variant == "reference":
        return 1
    spec = REGISTRY[variant]
    return spec["nproc"] if spec["kind"] == "colocated" else spec["dit"] + spec["vae"]


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _torchrun(nproc: int, worker: str, worker_args: list[str]) -> list[str]:
    return [
        sys.executable, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc),
        "--master_addr", "127.0.0.1", "--master_port", str(_find_free_port()),
        worker,
    ] + worker_args


def _visible_devices(need: int) -> list[str]:
    """The device ids this launch may use, honoring an exported filter."""
    filt = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    ids = [d.strip() for d in filt.split(",") if d.strip()] if filt else [
        str(i) for i in range(need)
    ]
    if len(ids) < need:
        raise SystemExit(
            f"[launch] this variant needs {need} GPUs but only {len(ids)} are visible "
            f"({', '.join(ids)}); unset or widen CUDA_VISIBLE_DEVICES")
    return ids[:need]


def _child_env(base: dict, devices: list[str] | None) -> dict:
    env = dict(base)
    if devices is not None:
        # The two halves run on disjoint GPUs, so each gets its own slice of
        # whatever was visible to the launcher.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env["HIP_VISIBLE_DEVICES"] = ",".join(devices)
    return env


def _run_disaggregated(spec: dict, cfg: str, env: dict, dry_run: bool) -> int:
    d, v = spec["dit"], spec["vae"]
    devices = _visible_devices(d + v)
    dit_argv = _torchrun(d, _PIPE_DIT_WORKER, [
        "--cfg", cfg, "--sp", str(spec["sp"]), "--shm-prefix", _SHM_PREFIX])
    vae_argv = _torchrun(v, _PIPE_VAE_WORKER, [
        "--cfg", cfg, "--tiles", str(v), "--shm-prefix", _SHM_PREFIX,
        "--vae-mode", spec["vae_mode"]])

    print(f"[launch] dit     = devices {','.join(devices[:d])} :: " + " ".join(dit_argv))
    print(f"[launch] vae     = devices {','.join(devices[d:])} :: " + " ".join(vae_argv),
          flush=True)
    if dry_run:
        return 0
    print("[launch] starting; wait for 'WorldPlay backend READY', then start the frontend "
          "(Ctrl-C stops it)", flush=True)

    procs = []
    try:
        procs.append(subprocess.Popen(dit_argv, env=_child_env(env, devices[:d])))
        procs.append(subprocess.Popen(vae_argv, env=_child_env(env, devices[d:])))
        while True:
            for proc in procs:
                rc = proc.poll()
                if rc is not None:
                    return rc
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        # Neither half is useful without the other, so whichever is still up
        # goes down too.
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a WorldPlay backend variant for the frontend.")
    ap.add_argument("--variant", help="reference, or one of the names shown by --list")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="app runtime config YAML")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not launch")
    args = ap.parse_args()

    if args.list:
        width = max(len(n) for n in REGISTRY) + 2
        print(f"{'reference':<{width}} 1 GPU   unmodified sequential reference backend")
        for name, spec in REGISTRY.items():
            n = gpus_needed(name)
            print(f"{name:<{width}} {n} GPU{'s' if n > 1 else ' '}  {spec['summary']}")
        return

    if not args.variant:
        ap.error("--variant is required (or use --list)")
    if args.variant != "reference" and args.variant not in REGISTRY:
        ap.error(f"unknown variant '{args.variant}'; see --list")

    env = os.environ.copy()
    for var in _DIST_VARS:
        env.pop(var, None)
    env["PYTHONUNBUFFERED"] = "1"

    need = gpus_needed(args.variant)
    print(f"[launch] variant = {args.variant}")
    print(f"[launch] gpus    = {need} (CUDA devices 0..{need - 1}; export CUDA_VISIBLE_DEVICES to remap)")
    print(f"[launch] config  = {args.config}")

    # Relative paths in the config (image_path, checkpoints) resolve against the repo root.
    os.chdir(_REPO_ROOT)

    if args.variant != "reference" and REGISTRY[args.variant]["kind"] == "pipe":
        raise SystemExit(_run_disaggregated(REGISTRY[args.variant], args.config, env, args.dry_run))

    if args.variant == "reference":
        argv = [sys.executable, "-u", "-m", "wllm.apps.worldplay.reference.launch_backend"]
    else:
        spec = REGISTRY[args.variant]
        argv = _torchrun(spec["nproc"], _COLOCATED_WORKER,
                         ["--cfg", args.config, "--sp", str(spec["sp"])] + spec["extra"])

    print("[launch] command = " + " ".join(argv), flush=True)
    if args.dry_run:
        return
    print("[launch] starting; wait for 'WorldPlay backend READY', then start the frontend (Ctrl-C stops it)",
          flush=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
