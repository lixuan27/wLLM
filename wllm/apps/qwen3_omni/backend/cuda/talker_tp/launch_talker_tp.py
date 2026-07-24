"""Launch the SPMD tensor-parallel Talker backend (variant talker_tp2).

Entry mode (no --rank): becomes rank 0 (driver), picks a free tcp://
rendezvous, spawns ranks 1..N-1 as child processes (each with its own
CUDA_VISIBLE_DEVICES), then runs rank 0 in THIS process so the harness's
process-group teardown reaps the whole tree.

Follower mode (--rank R): runs talker TP rank R.

CUDA_VISIBLE_DEVICES MUST be set before torch is imported, so this module
sets it at top-level from the config before importing the worker.

Usage (harness launches the entry):
  python wllm/apps/qwen3_omni/backend/talker_tp/launch_talker_tp.py CONFIG
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")


def _parse_cfg(cfg_path):
    import yaml
    with open(cfg_path) as f:
        d = yaml.safe_load(f) or {}
    s = d.get("streaming", {}) or {}
    talker_devs = [int(x) for x in str(s.get("talker_tp_devices", "")).split(",") if x != ""]
    world = int(s.get("talker_tp_size", len(talker_devs)))
    thinker_gpu = int((d.get("thinker") or {}).get("gpu_index", 0))
    c2w_gpu = int((d.get("code2wav") or {}).get("gpu_index", thinker_gpu))
    return talker_devs, world, thinker_gpu, c2w_gpu


def _rank_cvd(rank, talker_devs, thinker_gpu, c2w_gpu):
    # vLLM TP requires every rank to SEE all tp GPUs (device_count >= tp_size)
    # and pick its own via local_rank, so each rank's CVD lists ALL talker
    # GPUs in the same order (rank r -> cuda:r -> talker_devs[r]). rank 0 also
    # appends the thinker/c2w GPU(s).
    devs = list(talker_devs)
    if rank == 0:
        for g in (thinker_gpu, c2w_gpu):
            if g not in devs:
                devs.append(g)
    return ",".join(str(d) for d in devs)


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--init-method", default=None)
    ap.add_argument("--parity-fixture", default=None)
    ap.add_argument("--parity-out", default=None)
    args = ap.parse_args()
    cfg_path = os.path.abspath(args.config)
    talker_devs, world, thinker_gpu, c2w_gpu = _parse_cfg(cfg_path)

    if args.rank is not None:
        # follower mode: CVD already set by parent; just run our rank.
        from wllm.apps.qwen3_omni.backend.cuda.talker_tp.worker_tp import TalkerTPWorker
        w = TalkerTPWorker(cfg_path, rank=args.rank, world_size=args.world,
                           init_method=args.init_method,
                           parity_mode=(args.parity_out is not None))
        if args.parity_out is not None:
            w.run_parity(args.parity_fixture, args.parity_out)
        else:
            w.run()
        return

    # entry / rank 0
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    children = []
    for r in range(1, world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _rank_cvd(r, talker_devs, thinker_gpu, c2w_gpu)
        child_argv = [sys.executable, "-u", os.path.abspath(__file__), cfg_path,
                      "--rank", str(r), "--world", str(world), "--init-method", init_method]
        if args.parity_out is not None:
            child_argv += ["--parity-out", args.parity_out, "--parity-fixture", args.parity_fixture or ""]
        children.append(subprocess.Popen(child_argv, env=env))
    # rank 0 in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = _rank_cvd(0, talker_devs, thinker_gpu, c2w_gpu)
    try:
        from wllm.apps.qwen3_omni.backend.cuda.talker_tp.worker_tp import TalkerTPWorker
        w = TalkerTPWorker(cfg_path, rank=0, world_size=world, init_method=init_method,
                           parity_mode=(args.parity_out is not None))
        if args.parity_out is not None:
            w.run_parity(args.parity_fixture, args.parity_out)
        else:
            w.run()
    finally:
        for c in children:
            try:
                c.terminate()
            except Exception:
                pass
        for c in children:
            try:
                c.wait(timeout=15)
            except Exception:
                try:
                    c.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
