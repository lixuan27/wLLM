"""Process spawner for an optimized LongLive backend (monolithic SP or
disaggregated). The variant launcher (``launch.py``) builds the opts dict from
the registry; this direct CLI form takes it as JSON:

  python -m wllm.apps.longlive.backend.cuda.spawn <config.yaml> <opts.json>

Mono opts:  world, visible_devices, vae_mode, asr_mode, asr_device
Disagg opts: kind="disagg", dit_world, vae_world, dit_visible, vae_visible,
             asr_mode, asr_device, asr_visible

Spawns rank processes (env: RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* + per-group
CUDA_VISIBLE_DEVICES) and, for async ASR, one sidecar. All children share this
launcher's process group so the harness's killpg tears everything down; when any
child exits (e.g. rank 0 on terminate) the launcher kills the rest.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

SERVER = [sys.executable, "-u", "-m", "wllm.apps.longlive.backend.cuda.server"]
VAE = [sys.executable, "-u", "-m", "wllm.apps.longlive.backend.cuda.disagg"]
SIDECAR = [sys.executable, "-u", "-m", "wllm.apps.longlive.backend.cuda.asr_sidecar"]


class _Done(Exception):
    pass


def _spawn(argv, base, **env):
    e = base.copy()
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.Popen(argv, env=e)


def main():
    run(sys.argv[1],
        json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.loads(os.environ.get("LL_OPTS", "{}")))


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(cfg_path, opts):
    portA = _free_port()
    portB = _free_port()

    base = os.environ.copy()
    base["CONFIG_PATH"] = cfg_path
    base["PYTHONUNBUFFERED"] = "1"

    procs = []
    if opts.get("kind") == "disagg":
        if opts.get("asr_mode") == "async":
            procs.append(_spawn(SIDECAR, base, CUDA_VISIBLE_DEVICES=opts["asr_visible"],
                                ASR_DEVICE="cuda:0", LL_OPTS=json.dumps(opts)))
        dit_opts = dict(opts); dit_opts["role"] = "dit"; dit_opts["world"] = opts["dit_world"]
        for r in range(opts["dit_world"]):
            procs.append(_spawn(SERVER, base, CUDA_VISIBLE_DEVICES=opts["dit_visible"],
                                RANK=r, LOCAL_RANK=r, WORLD_SIZE=opts["dit_world"],
                                MASTER_ADDR="127.0.0.1", MASTER_PORT=portA,
                                LL_OPTS=json.dumps(dit_opts)))
        for r in range(opts["vae_world"]):
            procs.append(_spawn(VAE, base, CUDA_VISIBLE_DEVICES=opts["vae_visible"],
                                RANK=r, LOCAL_RANK=r, WORLD_SIZE=opts["vae_world"],
                                MASTER_ADDR="127.0.0.1", MASTER_PORT=portB,
                                LL_OPTS=json.dumps(opts)))
    else:
        world = int(opts.get("world", 1))
        visible = opts.get("visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if opts.get("asr_mode") == "async":
            procs.append(_spawn(SIDECAR, base, CUDA_VISIBLE_DEVICES=visible,
                                ASR_DEVICE=opts.get("asr_device", "cuda:0"),
                                LL_OPTS=json.dumps(opts)))
        for r in range(world):
            procs.append(_spawn(SERVER, base, CUDA_VISIBLE_DEVICES=visible,
                                RANK=r, LOCAL_RANK=r, WORLD_SIZE=world,
                                MASTER_ADDR="127.0.0.1", MASTER_PORT=portA,
                                LL_OPTS=json.dumps(opts)))

    rc = 0
    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    rc = p.returncode or 0
                    raise _Done()
            time.sleep(0.3)
    except (_Done, KeyboardInterrupt):
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        t0 = time.time()
        for p in procs:
            while p.poll() is None and time.time() - t0 < 15:
                time.sleep(0.2)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
