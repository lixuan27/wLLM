"""SPMD tensor-parallel Talker worker (variant talker_tp2).

The exposed Talker is the only model whose internals we can shard. Its
per-frame forward is the within-chunk parallelism axis the IR cannot
surface (it is below operator granularity); we attack it with vLLM
tensor parallelism across `talker_tp_size` GPUs.

Architecture (SPMD over `talker_tp_size` ranks, one process per rank):
  * Every rank builds the SAME talker, sharded TP across the ranks via
    vLLM's parallel_state (the substrate the talker model itself reads:
    get_tensor_model_parallel_world_size etc.). Ranks rendezvous on a
    shared tcp:// init_method.
  * rank 0 is the DRIVER: it owns the shm buffers, the Thinker + Code2Wav
    AsyncOmni engines, and the control loop. Per prompt it runs the
    Thinker, broadcasts the ThinkerOutput tensors to the other ranks over
    the TP process group, then ALL ranks prime + step the talker in
    lockstep. The TP all-reduces keep every rank's logits/hidden
    bit-consistent, and identical RNG seeds make the sampling identical,
    so every rank emits identical codec frames; rank 0 collects them and
    vocodes + writes audio. The other ranks discard their copies.
  * Control: rank 0 broadcasts a small opcode tensor (PROMPT / TERMINATE)
    each iteration so the followers stay in lockstep without polling shm.

Schedule: reference (full Thinker -> SPMD Talker -> full Code2Wav) to
ISOLATE the talker-TP lever. Correctness vs the reference is validated by
teacher-forced logit parity because TP
changes reduction order -> stochastic sampling diverges (allowed
fp noise); end-to-end audio is reported but not required to match.

On a node without NVLink the TP all-reduce runs over PCIe (NCCL host
transport with no GPU P2P) -- this is exactly the cost being measured.
"""

from __future__ import annotations

import os
import time
import uuid
import asyncio
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from wllm.apps.qwen3_omni.adapter import TEXT_FRAME_BYTES, decode_text_frame
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.utils.rand import set_global_seed
from wllm.apps.qwen3_omni.reference.worker import ThinkerOutput

from wllm.apps.qwen3_omni.backend.cuda import blocks
from wllm.apps.qwen3_omni.backend.cuda.streaming.config import StreamingConfig
from wllm.apps.qwen3_omni.backend.cuda.talker_tp.runner_tp import Qwen3OmniTalkerRunner

logger = init_logger(__name__)

_OP_IDLE, _OP_PROMPT, _OP_TERMINATE = 0, 1, 2


def _bcast_obj_tensor(t: Optional[torch.Tensor], src: int, device, dtype=None):
    """Broadcast a tensor of unknown shape from `src` to all ranks.

    Protocol: broadcast ndim, then the shape, then the data. Returns the
    received tensor on `device`."""
    rank = dist.get_rank()
    if rank == src:
        x = t.to(device)
        meta = torch.tensor([x.ndim, *x.shape], dtype=torch.long, device=device)
        n = torch.tensor([meta.numel()], dtype=torch.long, device=device)
        dist.broadcast(n, src)
        dist.broadcast(meta, src)
        xf = x.to(dtype) if dtype is not None else x
        xf = xf.contiguous()
        dist.broadcast(xf, src)
        return t
    else:
        n = torch.empty(1, dtype=torch.long, device=device)
        dist.broadcast(n, src)
        meta = torch.empty(int(n.item()), dtype=torch.long, device=device)
        dist.broadcast(meta, src)
        ndim = int(meta[0].item())
        shape = [int(s) for s in meta[1:1 + ndim].tolist()]
        out = torch.empty(shape, dtype=dtype if dtype is not None else torch.bfloat16, device=device)
        dist.broadcast(out, src)
        return out


def _bcast_int_list(lst, src: int, device):
    rank = dist.get_rank()
    if rank == src:
        x = torch.tensor(lst, dtype=torch.long, device=device)
    else:
        x = None
    x = _bcast_obj_tensor(x if rank == src else None, src, device, dtype=torch.long)
    return [int(v) for v in x.tolist()]


class TalkerTPWorker:
    def __init__(self, cfg_path: str, rank: int, world_size: int, init_method: str,
                 parity_mode: bool = False):
        self.parity_mode = parity_mode
        self.scfg = StreamingConfig.from_yaml(cfg_path)
        self.cfg = self.scfg.base
        self.sp = self.scfg.streaming
        self.rank = rank
        self.world_size = world_size
        self.is_driver = (rank == 0)
        set_global_seed(self.cfg.seed)

        talker_devs = [int(d) for d in (self.sp.talker_tp_devices or "").split(",") if d != ""]
        if len(talker_devs) < world_size:
            raise ValueError(f"talker_tp_devices={self.sp.talker_tp_devices} needs >= {world_size} GPUs")
        self.talker_gpu = talker_devs[rank]

        # Build the TP-sharded talker on this rank (rendezvous via init_method).
        # Eager (no torch.compile) to coexist cleanly with the rank-0
        # thinker/c2w engines and avoid compile-cache device collisions.
        #
        # GREEDY sampling is REQUIRED for SPMD lockstep: the reference talker
        # samples stochastically (temp 0.9), but under TP the per-rank logits
        # carry reduction-order fp noise, so independent stochastic sampling
        # on each rank would draw DIFFERENT tokens -> different EOS/continue
        # control flow -> the next all-reduce deadlocks. With greedy (temp 0,
        # mtp top_k=1) both ranks argmax the bit-identical all-reduced logits,
        # giving identical control flow and codes. The audio is therefore
        # greedy-talker speech (valid, deterministic) -- it already cannot
        # match the stochastic reference under TP anyway (see parity test).
        self.talker_runner = Qwen3OmniTalkerRunner(
            self.cfg.talker.model_path, gpu_index=self.talker_gpu,
            temperature=0.0, top_k=self.cfg.sampling.talker_top_k,
            top_p=self.cfg.sampling.talker_top_p, repetition_penalty=self.cfg.sampling.talker_repetition_penalty,
            seed=self.cfg.seed, max_tokens=self.cfg.sampling.talker_max_tokens,
            max_seq_len=self.cfg.sampling.talker_max_seq_len,
            mtp_top_k=1, mtp_top_p=1.0,
            tp_size=world_size, tp_rank=rank, tp_world_size=world_size,
            tp_init_method=init_method, enforce_eager=True)
        self.device = self.talker_runner.device
        logger.info("[rank %d] TP talker ready on gpu %d (cuda %s)", rank, self.talker_gpu, self.device)

        if self.is_driver and not self.parity_mode:
            self._init_driver()

    def _init_driver(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.text_input_buffer = SharedTensorBuffer(
            name=self.cfg.text_input_buffer_name, frame_shape=(TEXT_FRAME_BYTES,),
            max_len=int(self.cfg.text_max_pending), dtype=np.uint8, create=True)
        self.audio_output_buffer = SharedTensorBuffer(
            self.cfg.audio_output_buffer_name, frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32, max_len=int(self.cfg.audio_max_chunks), create=True)
        self.audio_meta_buffer = SharedControlBuffer(self.cfg.audio_meta_buffer_name, create=True)
        # thinker + c2w engines (rank 0 only), eager-friendly placement
        self.thinker_engine = blocks.make_thinker_engine(
            self.cfg, visible_devices=self.sp.thinker_visible_devices,
            stage_configs_path=self.sp.thinker_stage_configs_path)
        self.c2w_engine = blocks.make_c2w_engine(
            self.cfg, visible_devices=self.sp.c2w_visible_devices,
            stage_configs_path=self.sp.c2w_stage_configs_path)
        self._async = asyncio.Runner()
        try:
            self._async.run(blocks._c2w_collect(
                self.c2w_engine, self.cfg, [torch.zeros(16, dtype=torch.long) for _ in range(8)],
                "warmup-c2w-0"))
        except Exception:
            logger.exception("c2w warmup failed")
        self.session_started = False
        self._next_text_index = 0
        self.audio_meta_buffer.send(int(self.cfg.audio_sample_rate), timeout_s=10.0)
        logger.info("serving: talker_tp driver world=%d", self.world_size)
        logger.info("Qwen3-Omni backend READY")

    # ---- prime + step the talker SPMD from a ThinkerOutput on all ranks ----
    @staticmethod
    def _select_cond(runner, gen_step: int):
        trailing = runner._trailing_decode_embeds
        n = len(trailing)
        if gen_step < n:
            return trailing[gen_step]
        if runner._thinker_session_finished:
            return runner._tts_eos_embed if gen_step == n else runner._tts_pad_embed
        return None

    def _spmd_step(self, cond):
        """One codec frame, kept in lockstep across ranks by having rank 0
        sample the first-layer token + EOS flag and BROADCASTING them, so
        every rank follows identical control flow (no all-reduce deadlock).
        Everything downstream is deterministic given the (broadcast) token +
        the bit-identical TP all-reduce, so the frames match rank 0's."""
        runner = self.talker_runner
        last_pos_logits = runner._last_logits[:, -1, :]
        if self.is_driver:
            first_id = runner._sample_first_layer(last_pos_logits, generator=runner._sampling_generator)
            is_eos = int(first_id == runner.codec_eos_token_id)
            sig = torch.tensor([first_id, is_eos], dtype=torch.long, device=self.device)
            dist.broadcast(sig, 0)
        else:
            sig = torch.empty(2, dtype=torch.long, device=self.device)
            dist.broadcast(sig, 0)
            first_id, is_eos = int(sig[0].item()), int(sig[1].item())
        runner._sampled_token_history.append(first_id)
        if is_eos:
            runner._codec_eos_seen = True
            return None
        first_token = torch.tensor([[first_id]], dtype=torch.long, device=runner.device)
        layer0_embed = runner.talker.get_input_embeddings()(first_token)
        last_layer_hidden = runner._last_hidden[-1][:, -1:].to(
            device=layer0_embed.device, dtype=layer0_embed.dtype)
        cond_dev = cond.to(device=runner.device, dtype=runner.dtype).reshape(1, 1, -1)
        next_input_embed, all_token_ids = runner.talker.talker_mtp_forward(
            first_token, layer0_embed, last_talker_hidden=last_layer_hidden, text_step=cond_dev)
        residual = all_token_ids.to(torch.long)[:, 1:].clone()
        decode_pos = torch.tensor([[runner._cache_len]], dtype=torch.long, device=runner.device)
        cache_pos = torch.tensor(runner._cache_len, dtype=torch.long, device=runner.device)
        with torch.inference_mode():
            logits, hidden = runner.talker.forward_decode(next_input_embed, decode_pos, cache_pos)
        runner._last_hidden = (hidden,)
        runner._last_logits = logits
        runner._generation_step += 1
        runner._cache_len += 1
        return torch.cat([first_token, residual], dim=-1).reshape(-1).to(torch.long).clone()

    def _prime_and_generate(self, thinker: ThinkerOutput) -> List[torch.Tensor]:
        runner = self.talker_runner
        blocks.prime_talker(runner, thinker, self.cfg, push_all=True)  # SPMD lockstep
        frames: List[torch.Tensor] = []
        while True:
            gen_step = runner._generation_step
            if gen_step >= runner.max_tokens or runner._cache_len >= runner.talker.max_seq_len:
                break
            cond = self._select_cond(runner, gen_step)
            if cond is None:
                break
            frame = self._spmd_step(cond)
            if frame is None:
                break
            frames.append(frame)
        return frames

    def _broadcast_thinker(self, thinker: Optional[ThinkerOutput]) -> ThinkerOutput:
        dev = self.device
        if self.is_driver:
            ptoks = _bcast_int_list(thinker.prompt_token_ids, 0, dev)
            otoks = _bcast_int_list(thinker.output_token_ids, 0, dev)
            embed = _bcast_obj_tensor(thinker.embed_table, 0, dev, dtype=self.talker_runner.dtype)
            hidden = _bcast_obj_tensor(thinker.hidden_table, 0, dev, dtype=self.talker_runner.dtype)
            bos = _bcast_obj_tensor(thinker.tts_bos_embed, 0, dev, dtype=self.talker_runner.dtype)
            eos = _bcast_obj_tensor(thinker.tts_eos_embed, 0, dev, dtype=self.talker_runner.dtype)
            pad = _bcast_obj_tensor(thinker.tts_pad_embed, 0, dev, dtype=self.talker_runner.dtype)
            return thinker
        else:
            ptoks = _bcast_int_list(None, 0, dev)
            otoks = _bcast_int_list(None, 0, dev)
            embed = _bcast_obj_tensor(None, 0, dev, dtype=self.talker_runner.dtype)
            hidden = _bcast_obj_tensor(None, 0, dev, dtype=self.talker_runner.dtype)
            bos = _bcast_obj_tensor(None, 0, dev, dtype=self.talker_runner.dtype)
            eos = _bcast_obj_tensor(None, 0, dev, dtype=self.talker_runner.dtype)
            pad = _bcast_obj_tensor(None, 0, dev, dtype=self.talker_runner.dtype)
            return ThinkerOutput(prompt_token_ids=ptoks, output_token_ids=otoks,
                                 embed_table=embed, hidden_table=hidden,
                                 tts_bos_embed=bos, tts_eos_embed=eos, tts_pad_embed=pad, text="")

    # ---- follower loop (non-driver ranks) ----
    def follower_loop(self):
        dev = self.device
        while True:
            op = torch.empty(1, dtype=torch.long, device=dev)
            dist.broadcast(op, 0)
            opcode = int(op.item())
            if opcode == _OP_TERMINATE:
                break
            if opcode == _OP_PROMPT:
                thinker = self._broadcast_thinker(None)
                self._prime_and_generate(thinker)  # SPMD lockstep; frames discarded
        logger.info("[rank %d] follower terminating", self.rank)

    # ---- driver loop ----
    def _write_audio_full(self, audio: np.ndarray):
        fs = int(self.cfg.audio_frame_samples)
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_full = a.shape[0] // fs
        if n_full:
            self.audio_output_buffer.write(a[: n_full * fs].reshape(n_full, fs))
        rem = a[n_full * fs:]
        if rem.size:
            tail = np.zeros(fs, dtype=np.float32); tail[: rem.size] = rem
            self.audio_output_buffer.write(tail)

    def _handle_prompt(self, user_text: str):
        req_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            # tell followers a prompt is coming
            dist.broadcast(torch.tensor([_OP_PROMPT], dtype=torch.long, device=self.device), 0)
            thinker = blocks.run_thinker(self.thinker_engine, self.cfg, user_text,
                                         f"{req_id}-thinker", runner=self._async)
            thinker = self._broadcast_thinker(thinker)
            frames = self._prime_and_generate(thinker)
            audio, _ = blocks.vocode_full(self.c2w_engine, self.cfg, frames, f"{req_id}-c2w",
                                          runner=self._async)
            self._write_audio_full(audio)
            logger.info("[driver] prompt done in %.2fs (%d frames)", time.time() - t0, len(frames))
        except Exception:
            logger.exception("talker_tp inference failed")

    def is_cmd(self, c):
        return int(self.ctrl_buffer.recv()) == c

    def driver_loop(self):
        _CMD_START, _CMD_TERMINATE, _CMD_RESET = 1, 2, 3
        try:
            while True:
                if self.is_cmd(_CMD_TERMINATE):
                    dist.broadcast(torch.tensor([_OP_TERMINATE], dtype=torch.long, device=self.device), 0)
                    self._terminate()
                    break
                if self.is_cmd(_CMD_START) and not self.session_started:
                    self.session_started = True
                    self.audio_output_buffer.clear()
                    self._next_text_index = self.text_input_buffer.num
                    self.ctrl_buffer.commit()
                elif self.is_cmd(_CMD_RESET) and self.session_started:
                    self.session_started = False
                    self.audio_output_buffer.clear()
                    self._next_text_index = self.text_input_buffer.num
                    self.ctrl_buffer.commit()
                if not self.session_started:
                    time.sleep(0.005); continue
                idx, frame = self.text_input_buffer.read(self._next_text_index, 1)
                self._next_text_index = idx
                if frame is not None:
                    txt = decode_text_frame(frame[0])
                    if txt.strip():
                        logger.info("[driver] handling prompt: %r", txt[:80])
                        self._handle_prompt(txt)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            dist.broadcast(torch.tensor([_OP_TERMINATE], dtype=torch.long, device=self.device), 0)
            self._terminate()

    def _terminate(self):
        for nm, eng in (("thinker", getattr(self, "thinker_engine", None)),
                        ("c2w", getattr(self, "c2w_engine", None))):
            try:
                if eng is not None:
                    eng.shutdown()
            except Exception:
                logger.exception("error shutting down %s", nm)
        try:
            self._async.close()
        except Exception:
            pass
        for buf in (self.ctrl_buffer, self.text_input_buffer,
                    self.audio_output_buffer, self.audio_meta_buffer):
            try:
                buf.unlink()
            except Exception:
                pass
        logger.info("[driver] terminated")

    def run(self):
        if self.is_driver:
            self.driver_loop()
        else:
            self.follower_loop()

    # ---- teacher-forced parity: prime from a saved ThinkerOutput fixture
    #      (SPMD) and dump rank-0's pre-sampling layer-0 logits, so a TP=1
    #      run can be compared to it within fp tolerance. ----
    def run_parity(self, fixture_path: str, out_path: str):
        if self.is_driver:
            dist.broadcast(torch.tensor([_OP_PROMPT], dtype=torch.long, device=self.device), 0)
            d = torch.load(fixture_path, weights_only=False)
            thinker = ThinkerOutput(
                prompt_token_ids=d["prompt_token_ids"], output_token_ids=d["output_token_ids"],
                embed_table=d["embed_table"], hidden_table=d["hidden_table"],
                tts_bos_embed=d["tts_bos_embed"], tts_eos_embed=d["tts_eos_embed"],
                tts_pad_embed=d["tts_pad_embed"], text="")
            thinker = self._broadcast_thinker(thinker)
            blocks.prime_talker(self.talker_runner, thinker, self.cfg, push_all=True)
            logits = self.talker_runner._last_logits[:, -1, :].detach().float().cpu()
            # a few SPMD-lockstep steps, timed (per-frame SPMD forward rate).
            frames = []
            t0 = time.time()
            for _ in range(4):
                cond = self._select_cond(self.talker_runner, self.talker_runner._generation_step)
                if cond is None:
                    break
                f = self._spmd_step(cond)
                if f is None:
                    break
                frames.append(f.detach().cpu())
            per_frame_s = (time.time() - t0) / max(1, len(frames))
            torch.save({"logits": logits,
                        "frames": torch.stack(frames) if frames else torch.empty(0),
                        "per_frame_s": per_frame_s, "n_frames": len(frames)}, out_path)
            dist.broadcast(torch.tensor([_OP_TERMINATE], dtype=torch.long, device=self.device), 0)
            logger.info("[driver] parity saved -> %s (per_frame=%.3fs over %d frames)",
                        out_path, per_frame_s, len(frames))
        else:
            op = torch.empty(1, dtype=torch.long, device=self.device)
            dist.broadcast(op, 0)
            if int(op.item()) == _OP_PROMPT:
                thinker = self._broadcast_thinker(None)
                blocks.prime_talker(self.talker_runner, thinker, self.cfg, push_all=True)
                for _ in range(4):
                    cond = self._select_cond(self.talker_runner, self.talker_runner._generation_step)
                    if cond is None or self._spmd_step(cond) is None:
                        break
            op = torch.empty(1, dtype=torch.long, device=self.device)
            dist.broadcast(op, 0)  # terminate
