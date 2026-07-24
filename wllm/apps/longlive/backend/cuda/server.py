"""LongLive deployment server: one rank of a (possibly multi-GPU) backend.

Topology (monolithic SP world):
  * Every rank runs a ``LongLiveCore`` (DiT + VAE). When world_size>1 the DiT
    uses the shared-runtime Ulysses sequence parallelism (SP group == world).
  * Rank 0 is the driver: owns the IPC buffers (ctrl/audio/video/signal),
    handles ASR (sync on its own device, or async via a separate sidecar
    process), and broadcasts a per-iteration command to the followers so all
    ranks step the DiT in lockstep (the SP all-to-all is collective).
  * VAE decode: ``tile`` mode (world in {2,3,4}) decodes tile-parallel across
    all ranks (collective); ``rank0`` mode decodes only on rank 0.

Honors the same adapter contract as the reference backend.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.rt_config import RTConfig
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.distributed.parallel_state import (
    get_world_group, get_world_rank, get_world_size, get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from wllm.serving.distributed.communication_op import warmup_sequence_parallel_communication

from wllm.apps.longlive.backend.cuda.pipeline import LongLiveCore
from wllm.apps.longlive.backend.cuda import generation as G
from wllm.apps.longlive.backend.cuda.vad import StreamingVADSegmenter

logger = init_logger(__name__)
READY_MARKER = "LongLive backend READY"
_MAX_AUDIO_CHUNKS_PER_TICK = 200
_LATENT_RING = 8       # DiT->VAE latent ring depth (disagg); bounded by _max_ahead backpressure


class LongLiveServer:
    def __init__(self, cfg: RTConfig, opts: dict):
        self.cfg = cfg
        self.opts = opts
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_rank0 = (self.rank == 0)

        if self.world > 1:
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=self.world)
            self.device = get_local_torch_device()
        else:
            self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.role = opts.get("role", "mono")                # mono | dit (latent sink)
        self.vae_mode = opts.get("vae_mode") or ("tile" if self.world in (2, 3, 4) else "rank0")
        self.asr_mode = opts.get("asr_mode", "sync")        # sync | async
        self.asr_device = opts.get("asr_device", "cuda:0")  # device for ASR model (rank0)
        self._lat_name = (cfg.video_buffer_name or "ll") + "_lat"
        self._vcmd_name = (cfg.video_buffer_name or "ll") + "_vcmd"
        self._vae_seq = 0

        self.core = LongLiveCore(cfg, self.device)
        self.core.set_vae_world_size(self.world if self.vae_mode == "tile" else 1)
        self.core.warmup_vae()
        if self.world > 1:
            warmup_sequence_parallel_communication(self.device)
        self._warmup_dit()

        self.session_started = False
        self.has_prompt = False
        self._prompt_seq = 0       # monotonic id of the active prompt (latency tagging)
        self._pending_tag = None   # mono: a prompt_seq whose first video frame isn't tagged yet
        self._apply_time = 0.0     # wall-clock of the last prompt apply -> ASR-less latency
        self._apply_ms = 0
        # disagg backpressure: hold the DiT to at most _max_ahead chunks beyond what the
        # VAE group has consumed, so the latent ring stays shallow (a prompt update
        # surfaces in ~1 chunk instead of seconds) and the DiT never overwrites unread
        # latents. =1 keeps the full DiT∥VAE overlap (the VAE reads then decodes, so the
        # slot frees on read and the DiT computes the next chunk during the VAE decode).
        self._max_ahead = max(1, min(int(os.environ.get("LL_MAX_AHEAD", "1")), _LATENT_RING - 2))

        if self.is_rank0:
            self._setup_ipc()
            self._setup_asr()
            if self.role == "dit":
                # the video buffer is created by the VAE group; READY waits
                # for it so the frontend can always attach
                try:
                    self._video_probe = SharedTensorBuffer(
                        self.cfg.video_buffer_name,
                        frame_shape=(self.cfg.height, self.cfg.width, 3),
                        max_len=self.cfg.max_num_frames, dtype=np.uint8,
                        create=False, wait=True, timeout_s=1800.0)
                except TimeoutError as e:
                    raise RuntimeError(
                        "VAE group never created the video buffer — its ranks are "
                        "stuck or dead; see their output above") from e
            if self.asr_mode == "async":
                # wait for the sidecar's model load before advertising READY
                t0 = time.time()
                while self._asr_ready_flag.num < 1:
                    if time.time() - t0 > 900.0:
                        raise RuntimeError(
                            "ASR sidecar never came up; see its output above")
                    time.sleep(0.2)
            logger.info("serving: world=%d device=%s vae=%s asr=%s",
                        self.world, self.device, self.vae_mode, self.asr_mode)
            logger.info(READY_MARKER)
        else:
            logger.info("rank %d up (world=%d device=%s)", self.rank, self.world, self.device)

    # ------------------------------------------------------------------ setup
    def _warmup_dit(self):
        self.core.init_session(self.cfg.prompt or "warmup")
        blocks = G.ring_capacity_latents(self.cfg) // int(self.cfg.chunk_size) + 1
        for _ in range(blocks):
            latents = self.core.step_compute()
            if self.role != "dit" and (self.vae_mode == "tile" or self.is_rank0):
                self.core.decode_chunk(latents)
        self.core.reset()

    def _setup_ipc(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.audio_buffer = SharedTensorBuffer(
            self.cfg.audio_buffer_name, frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32, max_len=int(self.cfg.audio_max_chunks), create=True)
        self.signal_buffer = (SharedControlBuffer(self.cfg.signal_buffer_name, create=True)
                              if self.cfg.signal_buffer_name else None)
        self.num_read_input_chunks = 0
        self.segmenter = StreamingVADSegmenter()
        if self.role == "dit":
            # disaggregated: emit clean latents to the VAE group; command channel
            self.latent_buf = SharedTensorBuffer(
                self._lat_name,
                frame_shape=(self.cfg.dit_config.out_channels, int(self.cfg.chunk_size),
                             self.cfg.latent_height, self.cfg.latent_width),
                dtype=np.float32, max_len=_LATENT_RING, create=True)
            # parallel ring tagging each emitted latent chunk with (prompt_seq, apply_ms)
            # so the VAE group can mark the first *video* frame of each prompt and stamp
            # its ASR-less (apply -> frame) latency.
            self.latent_tag = SharedTensorBuffer(
                self._lat_name + "_tag", frame_shape=(2,), dtype=np.int64,
                max_len=_LATENT_RING, create=True)
            # the VAE group bumps this with its consumed-chunk index; the DiT reads it
            # for backpressure (don't outrun the VAE -> keep the ring shallow).
            self.vae_prog = SharedTensorBuffer(
                self._lat_name + "_prog", frame_shape=(1,), dtype=np.int64,
                max_len=1, create=True)
            self.vae_cmd = SharedControlBuffer(self._vcmd_name, create=True)
        else:
            self.video_buffer = SharedTensorBuffer(
                self.cfg.video_buffer_name, frame_shape=(self.cfg.height, self.cfg.width, 3),
                max_len=self.cfg.max_num_frames, dtype=np.uint8, create=True)
            # (prompt_seq, first_video_frame_index, asr_less_ms) per applied prompt; the
            # benchmark harness reads this to time prompt -> first corresponding frame,
            # plus the ASR-less apply -> frame latency stamped by the producer.
            self.tag_buffer = SharedTensorBuffer(
                self.cfg.video_buffer_name + "_tag", frame_shape=(3,), dtype=np.int64,
                max_len=512, create=True)

    def _vae_signal(self, op: int):
        """dit role rank0 -> VAE group lifecycle (3=reset/clear, 2=terminate).
        Distinct sequence numbers make each command edge-detectable; send()
        blocks until the VAE group acks, synchronizing reset/terminate."""
        if self.role == "dit" and self.is_rank0:
            self._vae_seq += 1
            self.vae_cmd.send(self._vae_seq * 4 + op,
                              timeout_s=(10.0 if op == 2 else 1800.0))

    def _setup_asr(self):
        self._asr = None
        self._asr_reader = None
        if self.asr_mode == "sync":
            from qwen_asr import Qwen3ASRModel
            self._asr = Qwen3ASRModel.from_pretrained(
                self.cfg.asr_model_name, dtype=torch.bfloat16, device_map=self.asr_device,
                attn_implementation="flash_attention_2", max_inference_batch_size=1,
                max_new_tokens=256)
            if os.getenv("WLLM_SKIP_ASR_WARMUP", "0") != "1":
                try:
                    self._asr.transcribe(audio=(np.zeros(int(self.cfg.audio_sample_rate*0.5),
                                                          dtype=np.float32),
                                                int(self.cfg.audio_sample_rate)), language="English")
                except Exception as e:
                    logger.warning("ASR warmup skipped: %s", e)
        else:  # async sidecar writes transcripts into a shm uint8 buffer
            self._asr_reader = SharedTensorBuffer(
                self.cfg.audio_buffer_name + "_txt", frame_shape=(512,), dtype=np.uint8,
                max_len=256, create=True)
            self._asr_txt_read = 0
            # set by the sidecar once its model is loaded; READY waits on it
            self._asr_ready_flag = SharedTensorBuffer(
                self.cfg.audio_buffer_name + "_asr_ready", frame_shape=(1,),
                dtype=np.int64, max_len=1, create=True)

    # ------------------------------------------------------------------ ASR
    def _drain_for_prompt_sync(self) -> Optional[str]:
        utt = None
        for _ in range(_MAX_AUDIO_CHUNKS_PER_TICK):
            self.num_read_input_chunks, ch = self.audio_buffer.read(self.num_read_input_chunks, 1)
            if ch is None:
                break
            ok, seg = self.segmenter.process_chunk(ch[0])
            if ok and seg is not None:
                utt = seg
        if utt is None:
            return None
        try:
            res = self._asr.transcribe(audio=(np.asarray(utt, np.float32).reshape(-1),
                                              int(self.cfg.audio_sample_rate)), language="English")
        except Exception:
            logger.exception("ASR failed")
            return None
        txt = (res[0].text or "").strip()
        return txt or None

    def _get_prompt_async(self) -> Optional[str]:
        # read newest transcript posted by the sidecar (non-blocking)
        latest = None
        while True:
            self._asr_txt_read, fr = self._asr_reader.read(self._asr_txt_read, 1)
            if fr is None:
                break
            raw = bytes(fr[0].tobytes()).split(b"\x00", 1)[0]
            latest = raw.decode("utf-8", "ignore").strip() or latest
        return latest

    def _get_prompt(self) -> Optional[str]:
        return self._drain_for_prompt_sync() if self.asr_mode == "sync" else self._get_prompt_async()

    # ------------------------------------------------------------------ loop
    def _decide(self) -> dict:
        ctrl = int(self.ctrl_buffer.recv())
        if ctrl == 2 and self.session_started:
            return {"op": "terminate"}
        if ctrl == 1 and not self.session_started:
            return {"op": "start"}
        if ctrl == 3 and self.session_started:
            return {"op": "reset"}
        if not self.session_started:
            return {"op": "idle"}
        new_prompt = self._get_prompt()
        is_first = (new_prompt is not None) and (not self.has_prompt)
        do_step = self.has_prompt or (new_prompt is not None)
        # disagg backpressure: don't get more than _max_ahead chunks ahead of the VAE
        # group's consumed index (keeps the latent ring shallow -> responsive prompts).
        if do_step and self.role == "dit":
            consumed = int(self.vae_prog.tensor[0, 0]) if self.vae_prog.num > 0 else 0
            if self.latent_buf.num - consumed >= self._max_ahead:
                do_step = False
        return {"op": "tick", "prompt": new_prompt, "is_first": is_first, "step": do_step}

    def _write_frames(self, frames):
        for f in frames:
            arr = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
            self.video_buffer.write(arr)

    def _run_chunk(self):
        latents = self.core.step_compute()
        if self.role == "dit":
            if self.is_rank0:
                # tag BEFORE the latent (seq + apply wall-clock for the ASR-less metric)
                self.latent_tag.write(np.array([self._prompt_seq, self._apply_ms], dtype=np.int64))
                self.latent_buf.write(latents[0].detach().cpu().float().numpy())
            return
        # first video frame of a freshly-applied prompt (mono): note (seq, frame_idx); the
        # ASR-less latency (apply -> this frame) is stamped right after frame 0 is written.
        tag = None
        if self.is_rank0 and self._pending_tag is not None:
            tag = (self._pending_tag, self.video_buffer.num)
            self._pending_tag = None
        # Stream each frame to the video buffer as soon as it is decoded
        # (don't batch all chunk_size decodes) for minimal first-frame latency.
        for l in range(int(self.cfg.chunk_size)):
            if self.vae_mode == "tile":
                frame = self.core.decode_one(latents, l)   # all ranks (collective)
                if self.is_rank0:
                    self._write_frames([frame])
            elif self.is_rank0:
                self._write_frames([self.core.decode_one(latents, l)])
            if tag is not None and l == 0 and self.is_rank0:
                asr_less = int((time.time() - self._apply_time) * 1000)
                self.tag_buffer.write(np.array([tag[0], tag[1], asr_less], dtype=np.int64))
                tag = None

    def _clear_video(self):
        if self.role != "dit":
            self.video_buffer.clear()

    def _apply(self, cmd: dict) -> bool:
        op = cmd["op"]
        if op == "terminate":
            if self.is_rank0:
                self._vae_signal(2)
                self._teardown_ipc()
            return False
        if op == "start":
            self.core.reset(); self.has_prompt = False; self.session_started = True
            if self.is_rank0:
                self._vae_signal(3)         # VAE clears its video + decode state
                self._clear_video(); self.audio_buffer.clear(); self.num_read_input_chunks = 0
                self.segmenter = StreamingVADSegmenter(); self.ctrl_buffer.commit()
            return True
        if op == "reset":
            # Mirror the reference worker: reset ends the session
            # (session_started=False); the driver must start() to resume.
            self.core.reset(); self.has_prompt = False; self.session_started = False
            if self.is_rank0:
                self._vae_signal(3)
                self._clear_video(); self.audio_buffer.clear(); self.num_read_input_chunks = 0
                self.segmenter = StreamingVADSegmenter()
                if self.asr_mode == "async":
                    # skip any buffered transcripts from before reset
                    self._asr_txt_read = self._asr_reader.num
                self.ctrl_buffer.commit()
            return True
        if op == "idle":
            time.sleep(0.005); return True
        if op == "tick":
            if cmd["prompt"] is not None:
                if cmd["is_first"]:
                    self.core.init_session(cmd["prompt"])
                else:
                    self.core.update_prompt(cmd["prompt"])
                self.has_prompt = True
                self._prompt_seq += 1            # new prompt -> new latency tag
                self._pending_tag = self._prompt_seq
                self._apply_time = time.time()   # ASR output ready -> the deployment path begins
                self._apply_ms = int(self._apply_time * 1000)
            if cmd["step"] and self.has_prompt:
                self._run_chunk()
            elif self.role == "dit":
                time.sleep(0.0005)               # idle / backpressure tick — don't spin the loop
            return True
        return True

    def _teardown_ipc(self):
        for b in (getattr(self, "ctrl_buffer", None), getattr(self, "audio_buffer", None),
                  getattr(self, "video_buffer", None), getattr(self, "signal_buffer", None),
                  getattr(self, "_asr_reader", None), getattr(self, "latent_buf", None),
                  getattr(self, "vae_cmd", None), getattr(self, "tag_buffer", None),
                  getattr(self, "latent_tag", None), getattr(self, "vae_prog", None)):
            if b is not None:
                try:
                    b.unlink()
                except Exception:
                    pass

    def run(self):
        try:
            if self.is_rank0:
                while True:
                    cmd = self._decide()
                    if self.world > 1:
                        get_world_group().broadcast_object(cmd, src=0)
                    if not self._apply(cmd):
                        break
            else:
                while True:
                    cmd = get_world_group().broadcast_object(None, src=0)
                    if not self._apply(cmd):
                        break
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main_rank():
    """Entry for one rank. Reads CONFIG_PATH + LL_OPTS (json) from env;
    RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT are set by the launcher.
    """
    import json
    from wllm.serving.utils.torch_utils import set_torch_options
    set_torch_options()
    cfg = RTConfig.from_yaml(os.environ["CONFIG_PATH"], is_path=True)
    opts = json.loads(os.environ.get("LL_OPTS", "{}"))
    LongLiveServer(cfg, opts).run()


if __name__ == "__main__":
    main_rank()
