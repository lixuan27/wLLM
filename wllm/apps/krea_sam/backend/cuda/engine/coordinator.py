"""Orchestrated Krea+SAM coordinator (the agent backend).

Owns the frontend-facing adapter contract (the three shm buffers + the
3-state control opcode) exactly like the reference worker, but instead
of running SAM then Krea sequentially on one GPU it:

  * launches the SAM service and the Krea service as independent
    subprocesses on their own GPU(s),
  * fans each input chunk out to *both* services concurrently
    (the worker-graph fact: krea_v2v ‖ sam_segment, no shared state),
  * pipelines across chunks (dispatch ahead, collect + composite in
    order), keeping both services busy,
  * composites (background swap) and writes the result to the output
    buffer with the same warmup-frame drop the reference uses.

This same engine runs every variant; ``BackendConfig`` selects the
process topology / scheduling. The compositing + input polling logic is
ported verbatim from the reference worker so output stays faithful.
"""

from __future__ import annotations

import os
from wllm.serving.paths import app_dir, repo_root
import queue
import socket
import subprocess
import tempfile
import sys
import threading
import time
from typing import Optional

import numpy as np
import yaml

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig

from wllm.apps.krea_sam.backend.cuda.engine.config import BackendConfig
from wllm.apps.krea_sam.backend.cuda.engine.ipc import CoordinatorLink

logger = init_logger(__name__)

_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCHELASTIC_ERROR_FILE",
)
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = repo_root()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _set_pdeathsig():
    """Make this child receive SIGKILL if the coordinator (parent) dies,
    so a killed/crashed coordinator never leaks GPU-holding services."""
    try:
        import ctypes
        import signal as _sig
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, _sig.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class KreaSamCoordinator:
    # subclasses set this; falls back to the cfg's backend.variant_name
    VARIANT: Optional[str] = None

    def __init__(self, cfg_path: str):
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        self.cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True).to_runtime_config()
        self.backend = self._load_backend_config(raw)
        self.cfg_path = cfg_path

        self.height = int(self.cfg.height)
        self.width = int(self.cfg.width)
        self.scale_t = int(self.cfg.vae_config.scale_factor_temporal)

        # --- adapter-facing shm buffers (created up-front so the harness
        #     sees them quickly; services load behind a [READY] marker) ---
        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name, frame_shape=(self.height, self.width, 3),
            max_len=int(self.cfg.max_num_frames), dtype=np.uint8, create=True)
        self.video_input_buffer = SharedTensorBuffer(
            name=self.cfg.video_input_buffer_name, frame_shape=(self.height, self.width, 3),
            max_len=int(self.cfg.video_input_max_frames), dtype=np.uint8, create=True)
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)

        self.session_started = False
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = 0
        self.dispatch_idx = 0  # chunk counter for input sizing

        self._procs: list[subprocess.Popen] = []
        self._sam_link: Optional[CoordinatorLink] = None
        self._krea_link: Optional[CoordinatorLink] = None
        self._dit_link: Optional[CoordinatorLink] = None
        self._vae_dit = (self.backend.krea_pipeline == "vae_dit")
        self._prefix = os.path.basename(self.cfg.ctrl_buffer_name)

        self._launch_services()
        self._start_io_threads()
        logger.info("Krea+SAM coordinator up: %s", self.backend.to_json())
        print("KreaSAM backend READY", flush=True)

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    def _load_backend_config(self, raw: dict) -> BackendConfig:
        knobs = dict(raw.get("backend", {}))
        if self.VARIANT and "variant_name" not in knobs:
            knobs["variant_name"] = self.VARIANT
        return BackendConfig(**knobs)

    # ------------------------------------------------------------------
    # service launch
    # ------------------------------------------------------------------
    def _child_env(self, gpu, dist_env: Optional[dict] = None) -> dict:
        env = os.environ.copy()
        for v in _DIST_VARS:
            env.pop(v, None)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        if dist_env:
            env.update(dist_env)
        return env

    def _connect_service(self, link, name):
        """Accept the service's connection and wait for its ready ack,
        failing fast if any service process dies instead of hanging."""
        while True:
            try:
                link.accept(timeout_s=2.0)
                break
            except TimeoutError:
                self._check_services_alive(name)
        while not link.poll(2.0):
            self._check_services_alive(name)
        assert link.recv().get("ack") == "ready"

    def _check_services_alive(self, waiting_for):
        for p in self._procs:
            if p.poll() is not None:
                log_dir = os.path.join(tempfile.gettempdir(),
                                       "wllm_krea_sam_service_logs")
                raise RuntimeError(
                    f"a backend service exited while waiting for the {waiting_for} "
                    f"service; see the logs under {log_dir}")

    def _spawn(self, argv, env, log_name):
        log_dir = os.path.join(tempfile.gettempdir(), "wllm_krea_sam_service_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"_{self._prefix}_{log_name}.log")
        fh = open(log_path, "w")
        # NOT setsid: keep services in the coordinator's process group so a
        # killpg on the coordinator reaps them too; pdeathsig is the backstop.
        p = subprocess.Popen(argv, env=env, cwd=_REPO_ROOT, stdout=fh,
                             stderr=subprocess.STDOUT, preexec_fn=_set_pdeathsig)
        self._procs.append(p)
        return p

    def _launch_services(self):
        sam_addr = f"/tmp/{self._prefix}_sam.sock"
        krea_addr = f"/tmp/{self._prefix}_krea.sock"
        for a in (sam_addr, krea_addr):
            if os.path.exists(a):
                os.remove(a)
        self._sam_link = CoordinatorLink(sam_addr)
        self._krea_link = CoordinatorLink(krea_addr)

        # SAM service
        sam_argv = [sys.executable, "-u", os.path.join(_BACKEND_DIR, "sam_service.py"),
                    "--address", sam_addr, "--cfg", self.cfg_path]
        if self.backend.sam_compile:
            sam_argv.append("--compile")
        self._spawn(sam_argv, self._child_env(self.backend.sam_gpu), "sam")

        # vae_dit: split Krea into a VAE service (krea link, encode+decode) and a
        # DiT service (dit link, denoise). krea_gpus[0] is the VAE GPU; the rest
        # (krea_gpus[1:]) run the DiT, sharded with Ulysses sequence parallelism
        # (DiT SP = len(dit_gpus); a single process when that is 1). Only DiT
        # rank 0 connects to the coordinator. The VAE streams each latent's
        # decoded frames when krea_stream_frames is set.
        if self._vae_dit:
            dit_addr = f"/tmp/{self._prefix}_dit.sock"
            if os.path.exists(dit_addr):
                os.remove(dit_addr)
            self._dit_link = CoordinatorLink(dit_addr)
            vae_gpu = self.backend.krea_gpus[0]
            dit_gpus = self.backend.krea_gpus[1:]
            dit_sp = len(dit_gpus)

            # VAE service (encode + decode), optionally streaming the decode.
            vae_argv = [sys.executable, "-u", os.path.join(_BACKEND_DIR, "vae_service.py"),
                        "--address", krea_addr, "--cfg", self.cfg_path]
            if self.backend.krea_stream_frames:
                vae_argv.append("--stream-frames")
            self._spawn(vae_argv, self._child_env(vae_gpu), "vae")

            # DiT service(s): single process at SP=1, else one rank per dit_gpu
            # forming a torch.distributed world (mirrors the krea_service SP path).
            if dit_sp == 1:
                self._spawn([sys.executable, "-u", os.path.join(_BACKEND_DIR, "dit_service.py"),
                             "--address", dit_addr, "--cfg", self.cfg_path, "--sp", "1"],
                            self._child_env(dit_gpus[0]), "dit0")
            else:
                port = _free_port()
                for r in range(dit_sp):
                    dist_env = {"RANK": str(r), "WORLD_SIZE": str(dit_sp), "LOCAL_RANK": "0",
                                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
                    self._spawn([sys.executable, "-u", os.path.join(_BACKEND_DIR, "dit_service.py"),
                                 "--address", dit_addr, "--cfg", self.cfg_path, "--sp", str(dit_sp)],
                                self._child_env(dit_gpus[r], dist_env), f"dit{r}")

            self._connect_service(self._sam_link, "sam")
            self._connect_service(self._krea_link, "krea")
            self._connect_service(self._dit_link, "dit")
            return

        # Krea service(s)
        sp = self.backend.krea_sp
        stream_args = ["--stream-frames"] if self.backend.krea_stream_frames else []
        if sp == 1:
            argv = [sys.executable, "-u", os.path.join(_BACKEND_DIR, "krea_service.py"),
                    "--address", krea_addr, "--cfg", self.cfg_path, "--sp", "1"] + stream_args
            self._spawn(argv, self._child_env(self.backend.krea_gpus[0]), "krea0")
        else:
            port = _free_port()
            for r in range(sp):
                dist_env = {"RANK": str(r), "WORLD_SIZE": str(sp), "LOCAL_RANK": "0",
                            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
                argv = [sys.executable, "-u", os.path.join(_BACKEND_DIR, "krea_service.py"),
                        "--address", krea_addr, "--cfg", self.cfg_path, "--sp", str(sp)] + stream_args
                self._spawn(argv, self._child_env(self.backend.krea_gpus[r], dist_env), f"krea{r}")

        # accept connections (rank-0 krea + sam), then wait for ready acks
        self._connect_service(self._sam_link, "sam")
        self._connect_service(self._krea_link, "krea")

    # ------------------------------------------------------------------
    # IO threads: per-service sender + receiver
    # ------------------------------------------------------------------
    def _start_io_threads(self):
        self._sam_send_q: queue.Queue = queue.Queue()
        self._krea_send_q: queue.Queue = queue.Queue()
        self._ack_q: queue.Queue = queue.Queue()
        self._results_lock = threading.Lock()
        self._pending: dict = {}            # id -> {"raw","krea","masks"}
        self._stop_io = threading.Event()

        self._dit_send_q: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._sender, args=(self._sam_link, self._sam_send_q), daemon=True),
            threading.Thread(target=self._sender, args=(self._krea_link, self._krea_send_q), daemon=True),
            threading.Thread(target=self._receiver, args=(self._sam_link, "masks"), daemon=True),
        ]
        if self._vae_dit:
            self._threads += [
                threading.Thread(target=self._sender, args=(self._dit_link, self._dit_send_q), daemon=True),
                threading.Thread(target=self._receiver_vae, daemon=True),
                threading.Thread(target=self._receiver_dit, daemon=True),
            ]
        else:
            self._threads.append(
                threading.Thread(target=self._receiver, args=(self._krea_link, "krea"), daemon=True))
        for t in self._threads:
            t.start()

    def _receiver_vae(self):
        """vae_dit: VAE service returns encode results (→ chain to DiT) and
        decode results (→ the chunk's krea_frames for compositing)."""
        link = self._krea_link
        while not self._stop_io.is_set():
            try:
                if not link.poll(0.2):
                    continue
                msg = link.recv()
            except (EOFError, OSError):
                return
            if "ack" in msg:
                self._ack_q.put(("krea", msg["ack"]))
                continue
            cid = msg.get("id")
            stage = msg.get("stage")
            if stage == "encode":
                latents = msg.get("latents")
                if latents is None:
                    # streaming encoder still priming → this chunk yields no
                    # latents; mark it empty so the compositor advances past it.
                    with self._results_lock:
                        slot = self._pending.setdefault(cid, {"raw": None, "krea": None, "masks": None})
                        if self.backend.krea_stream_frames:
                            slot.setdefault("krea_partials", [])
                            slot["krea_done"] = True
                        else:
                            slot["krea"] = np.zeros((0,))
                else:
                    self._dit_send_q.put({"cmd": "denoise", "id": cid, "latents": latents})
            elif stage == "decode":
                with self._results_lock:
                    slot = self._pending.setdefault(cid, {"raw": None, "krea": None, "masks": None})
                    if self.backend.krea_stream_frames:
                        # producer-streaming: accumulate per-latent partials
                        # (drained incrementally by _drain_composites_stream).
                        if "seq" in msg:
                            slot.setdefault("krea_partials", []).append(msg["frames"])
                        if msg.get("final"):
                            slot["krea_done"] = True
                    else:
                        slot["krea"] = msg["out"]

    def _receiver_dit(self):
        """vae_dit: DiT service returns denoised latents (→ chain to VAE decode)."""
        link = self._dit_link
        while not self._stop_io.is_set():
            try:
                if not link.poll(0.2):
                    continue
                msg = link.recv()
            except (EOFError, OSError):
                return
            if "ack" in msg:
                self._ack_q.put(("dit", msg["ack"]))
                continue
            cid = msg.get("id")
            self._krea_send_q.put({"cmd": "decode", "id": cid, "latents": msg["out"],
                                   "block_idx": cid})

    def _sender(self, link, q):
        while not self._stop_io.is_set():
            try:
                msg = q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                link.send(msg)
            except Exception:
                return

    def _receiver(self, link, key):
        while not self._stop_io.is_set():
            try:
                if not link.poll(0.2):
                    continue
                msg = link.recv()
            except (EOFError, OSError):
                return
            if "ack" in msg:
                self._ack_q.put((key, msg["ack"]))
                continue
            cid = msg.get("id")
            if cid is None:
                continue
            with self._results_lock:
                slot = self._pending.setdefault(cid, {"raw": None, "krea": None, "masks": None})
                if key == "krea" and self.backend.krea_stream_frames:
                    # producer-streaming: accumulate per-latent partials
                    if "seq" in msg:
                        slot.setdefault("krea_partials", []).append(msg["frames"])
                    if msg.get("final"):
                        slot["krea_done"] = True
                else:
                    slot[key] = msg["out"]

    # ------------------------------------------------------------------
    # session control (matches reference worker semantics)
    # ------------------------------------------------------------------
    def _clear_send_qs(self):
        """Drop any queued (unsent) data chunks so a control command is not
        stuck behind them (otherwise the adapter's 10s ack times out)."""
        qs = [self._sam_send_q, self._krea_send_q]
        if self._vae_dit:
            qs.append(self._dit_send_q)
        for q in qs:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _send_both(self, msg):
        self._sam_send_q.put(dict(msg))
        self._krea_send_q.put(dict(msg))
        if self._vae_dit:
            self._dit_send_q.put(dict(msg))

    def _await_acks(self, expected):
        n_services = 3 if self._vae_dit else 2
        got = set()
        deadline = time.time() + 1800.0
        while len(got) < n_services and time.time() < deadline:
            try:
                key, ack = self._ack_q.get(timeout=1.0)
                if ack == expected:
                    got.add(key)
            except queue.Empty:
                pass
            # bail out fast if a service process has died (don't hang)
            if any(p.poll() is not None for p in self._procs):
                logger.error("a service process died while awaiting %r ack", expected)
                break

    def start(self):
        self.session_started = True
        self.num_consumed_input_frames = 0
        self.dispatch_idx = 0
        self._next_composite_id = 0
        self._output_frame_skip_frames = max(0, self.scale_t - 1)
        with self._results_lock:
            self._pending.clear()
        self.video_input_buffer.clear()
        self.video_buffer.clear()
        self._clear_send_qs()
        self._send_both({"cmd": "start"})
        self._await_acks("start")
        self.ctrl_buffer.commit()
        logger.info("session started")

    def reset(self):
        self.session_started = False
        self.num_consumed_input_frames = 0
        self.dispatch_idx = 0
        self._output_frame_skip_frames = 0
        with self._results_lock:
            self._pending.clear()
        self._clear_send_qs()
        self._send_both({"cmd": "reset"})
        self._await_acks("reset")
        self.video_input_buffer.clear()
        self.video_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("session reset")

    def terminate(self):
        # Idempotent: loop() calls this on a graceful ctrl-buffer TERM, and the
        # launcher's finally calls it on Ctrl+C / exit -- both must be safe.
        if getattr(self, "_terminated", False):
            return
        self._terminated = True
        self.session_started = False
        self._send_both({"cmd": "stop"})
        time.sleep(0.5)
        self._stop_io.set()
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self._procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    import signal
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self._sam_link:
            self._sam_link.close()
        if self._krea_link:
            self._krea_link.close()
        if self._dit_link:
            self._dit_link.close()
        self.ctrl_buffer.unlink()
        self.video_input_buffer.unlink()
        self.video_buffer.unlink()

    # ------------------------------------------------------------------
    # input polling (ported from reference worker)
    # ------------------------------------------------------------------
    def _required_input_frames(self) -> int:
        cs = int(self.cfg.chunk_size)
        if self.dispatch_idx == 0:
            return 1 + max(0, cs - 1) * self.scale_t
        return cs * self.scale_t

    def _resample_frames(self, frames, target):
        if len(frames) == target:
            return frames
        idx = np.round(np.linspace(0, len(frames) - 1, target)).astype(np.int64)
        return frames[idx]

    def _poll_input_chunk(self):
        target = self._required_input_frames()
        available = max(0, int(self.video_input_buffer.num) - int(self.num_consumed_input_frames))
        if available < target:
            return None
        self.num_consumed_input_frames, frames = self.video_input_buffer.read(
            self.num_consumed_input_frames, available)
        if frames is None:
            return None
        selected = np.ascontiguousarray(self._resample_frames(frames, target))
        return selected

    # ------------------------------------------------------------------
    # compositing (ported from reference worker.loop)
    # ------------------------------------------------------------------
    @staticmethod
    def _composite(krea_frames, original_frames, masks):
        if masks is None:
            return krea_frames
        m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
        return original_frames * m3 + krea_frames * (1 - m3)

    def _emit_chunk(self, raw, krea_frames, masks):
        if krea_frames is None or len(krea_frames) == 0:
            return
        if self._output_frame_skip_frames > 0:
            skip = min(self._output_frame_skip_frames, int(krea_frames.shape[0]))
            krea_frames = krea_frames[skip:]
            raw = raw[skip:]
            if masks is not None:
                masks = masks[skip:]
            self._output_frame_skip_frames -= skip
            if krea_frames.shape[0] == 0:
                return
        n_out = int(krea_frames.shape[0])
        originals = raw[:n_out] if raw.shape[0] >= n_out else None
        if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
            self.video_buffer.write(krea_frames)
            return
        if masks is not None and masks.shape[0] >= n_out:
            masks = masks[:n_out]
        else:
            masks = None
        if self.backend.stream_decode:
            comp = self._composite(krea_frames, originals, masks)
            for fr in comp:
                self.video_buffer.write(fr)
        else:
            self.video_buffer.write(self._composite(krea_frames, originals, masks))

    def _drain_composites(self):
        while True:
            cid = self._next_composite_id
            with self._results_lock:
                slot = self._pending.get(cid)
                ready = slot is not None and slot["krea"] is not None and slot["masks"] is not None
                if ready:
                    raw, krea, masks = slot["raw"], slot["krea"], slot["masks"]
                    del self._pending[cid]
            if not ready:
                return
            self._emit_chunk(raw, krea, masks)
            self._next_composite_id += 1

    def _emit_partial(self, slot, frames, masks):
        """Emit one streamed latent's decoded frames (background-swapped),
        honoring the session warmup-frame drop across partials."""
        n = int(frames.shape[0])
        idx = slot.get("emit_idx", 0)
        raw = slot["raw"]
        raw_k = raw[idx:idx + n] if raw.shape[0] >= idx + n else None
        masks_k = masks[idx:idx + n] if (masks is not None and masks.shape[0] >= idx + n) else None
        slot["emit_idx"] = idx + n
        fr = frames
        if self._output_frame_skip_frames > 0:
            s = min(self._output_frame_skip_frames, n)
            fr = fr[s:]
            raw_k = raw_k[s:] if raw_k is not None else None
            masks_k = masks_k[s:] if masks_k is not None else None
            self._output_frame_skip_frames -= s
            if fr.shape[0] == 0:
                return
        if raw_k is None or raw_k.shape[:3] != fr.shape[:3]:
            for f in fr:
                self.video_buffer.write(f)
            return
        comp = self._composite(fr, raw_k, masks_k)
        for f in comp:
            self.video_buffer.write(f)

    def _drain_composites_stream(self):
        """Producer-streaming compositor: emit each krea partial as it
        arrives, once the (batched) SAM masks for the chunk are ready."""
        while True:
            cid = self._next_composite_id
            with self._results_lock:
                slot = self._pending.get(cid)
                if slot is None or slot.get("masks") is None:
                    return
                partials = slot.get("krea_partials", [])
                proc = slot.get("krea_proc", 0)
                new = partials[proc:]
                slot["krea_proc"] = len(partials)
                masks = slot["masks"]
                done = slot.get("krea_done", False)
                fully = done and slot["krea_proc"] >= len(partials)
            for frames in new:
                self._emit_partial(slot, frames, masks)
            if fully:
                with self._results_lock:
                    self._pending.pop(cid, None)
                self._next_composite_id += 1
            else:
                return

    # ------------------------------------------------------------------
    # control opcode helpers + main loop
    # ------------------------------------------------------------------
    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    PIPELINE_DEPTH = 4

    def loop(self):
        while True:
            if self.is_terminate():
                self.terminate()
                break
            if self.is_start() and not self.session_started:
                self.start()
            elif self.is_reset() and self.session_started:
                self.reset()

            if not self.session_started:
                time.sleep(0.005)
                continue

            # dispatch new chunks while pipeline has room
            inflight = self.dispatch_idx - self._next_composite_id
            if inflight < self.PIPELINE_DEPTH:
                chunk = self._poll_input_chunk()
                if chunk is not None:
                    cid = self.dispatch_idx
                    with self._results_lock:
                        self._pending.setdefault(cid, {"raw": None, "krea": None, "masks": None})["raw"] = chunk
                    self._sam_send_q.put({"cmd": "chunk", "id": cid, "frames": chunk})
                    if self._vae_dit:
                        # start the VAE→DiT→VAE chain with an encode request
                        self._krea_send_q.put({"cmd": "encode", "id": cid, "frames": chunk,
                                               "block_idx": cid})
                    else:
                        self._krea_send_q.put({"cmd": "chunk", "id": cid, "frames": chunk})
                    self.dispatch_idx += 1
                else:
                    time.sleep(0.001)

            if self.backend.krea_stream_frames:
                self._drain_composites_stream()
            else:
                self._drain_composites()
