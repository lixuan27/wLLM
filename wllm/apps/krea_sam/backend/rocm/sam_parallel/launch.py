"""Launch the `sam_parallel` backend: Krea orchestrator on one GPU + a
decoupled SAM worker on another, connected by a SamLink. 2 GPUs total.

Usage (the harness builds this command):
  python -m wllm.apps.krea_sam.backend.rocm.sam_parallel.launch <cfg_path> \
      --krea-gpu 0 --sam-gpu 1

This process IS the Krea orchestrator (rank 0). It spawns the SAM worker as a
child in its own dist-scrubbed env pinned to --sam-gpu, then runs the loop.
On exit it tears the child down.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys

import torch  # noqa: E402

from wllm.apps.krea_sam.backend.rocm.engine.orchestrator import KreaOrchestrator  # noqa: E402
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_stream import KreaOrchestratorStream  # noqa: E402

_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCHELASTIC_ERROR_FILE",
)


def spawn_sam_worker(cfg_path: str, link_name: str, sam_gpu: int) -> subprocess.Popen:
    env = os.environ.copy()
    for v in _DIST_VARS:
        env.pop(v, None)
    env["CUDA_VISIBLE_DEVICES"] = str(sam_gpu)   # SAM sees exactly one GPU as cuda:0
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u",
           os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine", "sam_worker.py"),
           cfg_path, link_name, "cuda:0"]
    return subprocess.Popen(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_path")
    ap.add_argument("--krea-gpu", type=int, default=0)
    ap.add_argument("--sam-gpu", type=int, default=1)
    ap.add_argument("--stream", action="store_true", help="per-frame streaming emit")
    args = ap.parse_args()

    link_name = f"samlink_{os.path.basename(args.cfg_path).replace('.yaml','').replace('cfg_','')}"

    sam_proc = spawn_sam_worker(args.cfg_path, link_name, args.sam_gpu)

    def _cleanup():
        if sam_proc.poll() is None:
            try:
                sam_proc.terminate()
                sam_proc.wait(timeout=10)
            except Exception:
                try:
                    sam_proc.kill()
                except Exception:
                    pass
    atexit.register(_cleanup)

    cls = KreaOrchestratorStream if args.stream else KreaOrchestrator
    orch = cls(
        cfg_path=args.cfg_path, sam_link_name=link_name,
        device=torch.device(f"cuda:{args.krea_gpu}"), rank=0, world=1)
    try:
        orch.loop()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
