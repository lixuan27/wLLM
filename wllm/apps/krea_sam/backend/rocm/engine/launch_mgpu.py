"""Generic multi-GPU launcher: spawns a K-rank Krea SP group + a decoupled SAM
worker. K = len(--krea-gpus). Rank 0 owns the adapter contract + SAM link.

Usage:
  python -m wllm.apps.krea_sam.backend.rocm.engine.launch_mgpu <cfg> \
      --krea-gpus 0,1,2 --sam-gpu 3 [--master-port 29511]

GPU indices are physical (this process is launched with all GPUs visible); each
Krea rank gets CUDA_VISIBLE_DEVICES=<krea-gpus> + LOCAL_RANK=i so it lands on
krea_gpus[i]; SAM gets CUDA_VISIBLE_DEVICES=<sam-gpu>, dist env scrubbed.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time

_LIB = os.path.dirname(__file__)
_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCHELASTIC_ERROR_FILE",
)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_path")
    ap.add_argument("--krea-gpus", required=True)   # comma list, physical
    ap.add_argument("--sam-gpu", type=int, required=True)
    ap.add_argument("--master-port", type=int, default=0)
    ap.add_argument("--sp-size", type=int, default=0)  # 0 => SP = world (full frame-SP)
    ap.add_argument("--stream", action="store_true")   # per-frame streaming emit
    ap.add_argument("--pipeline", action="store_true")  # disaggregated encode+denoise|decode
    # DiT-SP over ranks 0..N-2 || VAE decode on the last rank (needs --sp-size)
    ap.add_argument("--sp-vae-split", action="store_true")
    args = ap.parse_args()

    krea_gpus = [int(x) for x in args.krea_gpus.split(",")]
    world = len(krea_gpus)
    port = args.master_port or _free_port()
    link_name = f"samlink_{os.path.basename(args.cfg_path).replace('.yaml','').replace('cfg_','')}"

    procs: list[subprocess.Popen] = []

    def cleanup():
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        t0 = time.time()
        for p in procs:
            try:
                p.wait(timeout=max(0.1, 12 - (time.time() - t0)))
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    atexit.register(cleanup)

    # SAM worker (own GPU, dist scrubbed)
    sam_env = os.environ.copy()
    for v in _DIST_VARS:
        sam_env.pop(v, None)
    sam_env["CUDA_VISIBLE_DEVICES"] = str(args.sam_gpu)
    sam_env["PYTHONUNBUFFERED"] = "1"
    procs.append(subprocess.Popen(
        [sys.executable, "-u", os.path.join(_LIB, "sam_worker.py"),
         args.cfg_path, link_name, "cuda:0"], env=sam_env))

    # Krea SP ranks
    krea_cvd = ",".join(str(g) for g in krea_gpus)
    for rank in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = krea_cvd
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world)
        env["LOCAL_RANK"] = str(rank)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(port)
        if args.sp_size:
            env["SP_SIZE"] = str(args.sp_size)
        if args.stream:
            env["KREA_STREAM"] = "1"
        if args.pipeline:
            env["KREA_PIPELINE"] = "1"
        if args.sp_vae_split:
            env["KREA_SP_VAE_SPLIT"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        procs.append(subprocess.Popen(
            [sys.executable, "-u", os.path.join(_LIB, "krea_rank.py"),
             args.cfg_path, link_name], env=env))

    # Wait: if any process dies, tear the rest down.
    try:
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    return
            time.sleep(0.5)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
