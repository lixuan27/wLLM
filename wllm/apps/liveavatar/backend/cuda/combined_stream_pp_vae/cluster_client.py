"""Client for the 5-stage (4 DiT + 1 VAE) combined_stream_pp_vae cluster.

Vendored from pp_steps/cluster_client.py with three changes:
  1. spawns `world = len(gpus)` ranks where the LAST rank is the VAE-decode stage,
     so the DiT steps are distributed over `world-1` ranks (PP_NDIT);
  2. owns a VIDEO queue (rank4 -> worker) instead of a latents queue -- the worker
     receives already-decoded frames;
  3. keeps the VAE inside the NCCL world as a first-class pipeline rank, which
     needs GPU-to-GPU P2P (NVLink-class nodes; see cluster.py for why pp_steps
     keeps the VAE off the cluster instead).
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time

import numpy as np

from wllm.serving.rt_config import RTConfig
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.apps.liveavatar.backend.cuda.combined_stream_pp_vae.cluster import (
    shm_names, OP_INIT, OP_TERM, OP_RESET,
)
from wllm.apps.liveavatar.backend.cuda import runtime_common as common  # applies resource_tracker patch

CLUSTER_PY = os.path.join(os.path.dirname(__file__), "cluster.py")


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ClusterClientVAE:
    def __init__(self, cfg_path: str, prefix: str, cluster_gpus, *, log_dir: str = None):
        self.cfg_path = cfg_path
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)
        self.prefix = prefix
        self.gpus = list(cluster_gpus)
        self.world = len(self.gpus)
        self.n_dit = self.world - 1                       # last gpu hosts the VAE
        assert self.n_dit >= 1, "need >=2 gpus (>=1 DiT rank + 1 VAE rank)"
        assert int(self.cfg.num_inference_steps) % self.n_dit == 0, \
            (f"5-stage cluster needs num_inference_steps ({self.cfg.num_inference_steps}) "
             f"divisible by #DiT ranks ({self.n_dit}) = #gpus-1")
        self.names = shm_names(prefix)
        self.log_dir = log_dir or common.default_cluster_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        self._seq = 0

        step_frames = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
        self._af_shape = (25, 1024, step_frames)

        common.clean_shm(prefix)
        # client owns: audio-feature queue + control buffer (worker->rank0),
        # the VIDEO queue (rank4 VAE -> worker), and the ref buffer (rank3 -> DiT
        # ranks). No latents queue -- the VAE decode lives on rank4.
        self.audio_buf = SharedTensorBuffer(self.names["audio"], frame_shape=self._af_shape,
                                            dtype=np.float32, max_len=8192, create=True)
        self.ctrl_buf = SharedControlBuffer(self.names["ctrl"], create=True)
        self.video_buf = SharedTensorBuffer(
            self.names["video"],
            frame_shape=(int(self.cfg.height), int(self.cfg.width), 3),
            dtype=np.uint8, max_len=int(self.cfg.max_num_frames), create=True)
        C = int(self.cfg.dit_config.out_channels)
        self.ref_buf = SharedTensorBuffer(
            self.names["ref"],
            frame_shape=(C, 1, int(self.cfg.latent_height), int(self.cfg.latent_width)),
            dtype=np.float32, max_len=256, create=True)
        self.procs = []
        self._spawn()

    def _send_ctrl(self, op, timeout_s=1800.0):
        self._seq += 1
        ok = self.ctrl_buf.send((self._seq << 3) | op, timeout_s=timeout_s)
        if not ok:
            raise TimeoutError(f"cluster ctrl op={op} ack timeout")

    def _spawn(self):
        port = _free_port()
        base = os.environ.copy()
        # scrub any inherited torch elastic vars, then set OUR world's vars
        for v in common._DIST_VARS:
            base.pop(v, None)
        # This node has NVLink P2P -> keep it ENABLED (=0) so the VAE rank's eager
        # cudnn work does not stall inside the NCCL world. pp_steps' VAE-in-world
        # stall (bug #6) was on a node without NVLink that forced
        base["MASTER_ADDR"] = "127.0.0.1"
        base["MASTER_PORT"] = str(port)
        base["WORLD_SIZE"] = str(self.world)
        # Eager DiT (no per-rank max-autotune recompile); the win is the cross-chunk
        # PIPELINE, orthogonal to kernel fusion (pp_steps bug #5).
        base["TORCHDYNAMO_DISABLE"] = "1"
        for rank, gpu in enumerate(self.gpus):
            env = base.copy()
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PP_RANK"] = str(rank)
            env["PP_WORLD"] = str(self.world)
            env["PP_NDIT"] = str(self.n_dit)           # DiT-rank count (world-1)
            env["PP_CFG"] = self.cfg_path
            env["PP_PREFIX"] = self.prefix
            logf = open(os.path.join(self.log_dir, f"pp_rank{rank}_{self.prefix}.log"), "w")
            p = subprocess.Popen([common.ENV_PY, "-u", CLUSTER_PY],
                                 stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=common.REPO_ROOT, env=env, preexec_fn=os.setsid)
            p._logf = logf
            p._log_path = os.path.join(self.log_dir, f"pp_rank{rank}_{self.prefix}.log")
            self.procs.append(p)

    def wait_ready(self, timeout=1800.0):
        t0 = time.time()
        ready = [False] * self.world
        while time.time() - t0 < timeout:
            for i, p in enumerate(self.procs):
                if p.poll() is not None:
                    raise RuntimeError(f"cluster rank {i} died; see {p._log_path}")
                if not ready[i]:
                    try:
                        with open(p._log_path, errors="ignore") as f:
                            # matches both "ready (steps ...)" and "ready (vae decode)"
                            if "] ready (" in f.read():
                                ready[i] = True
                    except FileNotFoundError:
                        pass
            if all(ready):
                return True
            time.sleep(1.0)
        raise RuntimeError("cluster not ready in time")

    def init_session(self):
        self._send_ctrl(OP_INIT)

    def reset_session(self):
        self._send_ctrl(OP_RESET)

    def push_features(self, feats_np: np.ndarray):
        # feats_np shape (25,1024,step_frames) float32
        self.audio_buf.write(np.ascontiguousarray(feats_np, dtype=np.float32))

    def terminate(self):
        try:
            self._send_ctrl(OP_TERM, timeout_s=10.0)
        except Exception:
            pass
        time.sleep(1.0)
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                p._logf.close()
            except Exception:
                pass
        common.kill_gpu_stragglers(",".join(str(g) for g in self.gpus))
        common.clean_shm(self.prefix)
