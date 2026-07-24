"""Shared runtime helpers for the LiveAvatar backend variants.

Importing this module patches CPython's multiprocessing resource tracker so
that processes which only ATTACH to shared-memory buffers (`create=False`)
never unlink them on exit. Without the patch, a cluster rank or adapter
process exiting destroys the live worker's buffers. Processes that create
buffers still clean them up explicitly in their terminate paths.
"""

from __future__ import annotations

import glob
import os
from wllm.serving.paths import repo_root
import signal
import subprocess
import sys
import tempfile


def _detach_shm_resource_tracker() -> None:
    try:
        from multiprocessing import resource_tracker as _rt
    except Exception:
        return

    def _register(name, rtype, *a, **k):
        if rtype == "shared_memory":
            return
        return _rt._resource_tracker.register(name, rtype, *a, **k)

    def _unregister(name, rtype, *a, **k):
        if rtype == "shared_memory":
            return
        return _rt._resource_tracker.unregister(name, rtype, *a, **k)

    _rt.register = _register
    _rt.unregister = _unregister
    if hasattr(_rt, "_CLEANUP_FUNCS"):
        _rt._CLEANUP_FUNCS.pop("shared_memory", None)


_detach_shm_resource_tracker()


REPO_ROOT = repo_root()
# Interpreter used to launch cluster-rank subprocesses: the same one running
# the worker, so children inherit the environment.
ENV_PY = sys.executable

# torch.distributed env vars to scrub before launching a child that does its
# own distributed init (see the repo-root AGENTS.md known-pitfall section).
_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS",
    "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)


def default_cluster_log_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "wllm_liveavatar_cluster_logs")
    os.makedirs(path, exist_ok=True)
    return path


def clean_shm(prefix: str) -> None:
    for p in glob.glob(f"/dev/shm/{prefix}_*"):
        try:
            os.remove(p)
        except OSError:
            pass


def kill_gpu_stragglers(cvd: str, protect=()) -> list:
    """Kill any compute-app PIDs left on the given physical GPUs (vLLM/TTS
    spawn workers can setsid out of the launcher's process group and survive
    killpg). `cvd` is a comma list of physical GPU indices. Returns the PIDs
    it killed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", cvd, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    killed = []
    for line in out.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid in protect or pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            pass
    return killed
