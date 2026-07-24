"""LIBERO episode client — runs in the simulator's environment and calls
the VLA action server over the Unix socket.  One episode per
libero_spatial task (N tasks = N episodes), success + per-action latency
recorded; results JSON per dtype arm."""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import sys
import time
from pathlib import Path

SOCK = os.environ.get("WLLM_VLA_SOCK", "/tmp/wllm_vla.sock")
DTYPE = os.environ.get("WLLM_VLA_DTYPE", "bfloat16")
N_TASKS = int(os.environ.get("WLLM_LIBERO_TASKS", "10"))
MAX_STEPS = int(os.environ.get("WLLM_LIBERO_MAX_STEPS", "300"))
OUT_DIR = Path("/public/home/lixuan/lixuan/wllm-infra/benchmarks/results")


def _send_obj(conn, obj):
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


def _recv_obj(conn):
    def rx(n):
        buf = b""
        while len(buf) < n:
            c = conn.recv(n - len(buf))
            if not c:
                raise ConnectionError("server closed")
            buf += c
        return buf
    (n,) = struct.unpack(">Q", rx(8))
    return pickle.loads(rx(n))


def main() -> int:
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(SOCK)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    results = []
    for task_id in range(min(N_TASKS, suite.n_tasks)):
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                 camera_widths=256)
        env.seed(7)
        env.reset()
        init_states = suite.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[0])
        for _ in range(10):   # settle physics
            obs, *_ = env.step([0.0] * 6 + [-1.0])

        success, lat_ms, steps = False, [], 0
        while steps < MAX_STEPS:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            _send_obj(conn, {"image": img, "instruction": task.language,
                             "unnorm_key": None})
            rep = _recv_obj(conn)
            lat_ms.append(rep["ms"])
            chunk = np.asarray(rep["action"], dtype=np.float64).reshape(-1, 7)
            done = False
            for act in chunk:
                obs, _, done, _ = env.step(act.tolist())
                steps += 1
                if done or steps >= MAX_STEPS:
                    break
            if done:
                success = True
                break
        env.close()
        med = sorted(lat_ms)[len(lat_ms) // 2] if lat_ms else None
        results.append({"task": task.name, "success": bool(success),
                        "steps": steps, "n_predicts": len(lat_ms),
                        "median_predict_ms": med})
        print(f"[client] task{task_id} {task.name}: success={success} "
              f"steps={steps} med={med:.0f}ms" if med else
              f"[client] task{task_id}: no predicts", flush=True)

    summary = {"dtype": DTYPE, "n_tasks": len(results),
               "successes": sum(r["success"] for r in results),
               "success_rate": sum(r["success"] for r in results) / max(len(results), 1),
               "median_predict_ms": sorted(
                   r["median_predict_ms"] for r in results
                   if r["median_predict_ms"])[len(results) // 2]
               if any(r["median_predict_ms"] for r in results) else None,
               "tasks": results}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"libero_bridge_{DTYPE}_{time.strftime('%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}),
          flush=True)
    _send_obj(conn, {"op": "shutdown"})
    print(f"BRIDGE_{DTYPE.upper()}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
