"""Variant `stream_pp2_sp3` — the maximal stack: denoise_pp_sp (L2+L3, overall best
DiT core) + stream_full's TTS overlap (L1-deep).

denoise_pp_sp already stacks L1 (per-chunk streaming) + L2 (2-stage denoise
pipeline) + L3 (sp=3), and is the overall best (3.02 s latency, 74.2 fps). Its
latency-to-first-output still includes the *full* TTS wait: the base worker
generates the entire TTS audio before the DiT starts. This variant removes that
wait — it drives the 2-stage×sp3 pipeline directly from the TTS stream, starting
the DiT on the first 480 ms of TTS audio while TTS keeps generating the rest in its
own vLLM-Omni process. wav2vec offload (L6) is intentionally NOT included: it was
measured inert (the DiT, not wav2vec, is the bottleneck) and there is no free GPU
on the 8-GPU layout anyway.

IR basis: L1 tts→liveavatar streaming edge (VARIABLE_RATE) composed with the L2+L3
DiT core. The hard part is the distributed control protocol: the base denoise_pp_sp
broadcasts the chunk count `m` up front, which streaming cannot know. This worker
replaces that with a per-stage streaming control — a "more" flag SP-broadcast within
each stage and P2P-forwarded between stages — with NO per-chunk world collective, so
the depth-2 pipeline overlap (stage 0 on chunk c+1 while stage 1 finishes chunk c) is
preserved. Only the one-time chunk-0 ref-latents sync remains a world broadcast.

Attacks: latency-to-first-output (removes the full-TTS wait on top of the best DiT
core) while keeping denoise_pp_sp's sustainable rate.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from wllm.serving.logger import init_logger
from wllm.apps.liveavatar.reference.worker import truncate_to_last_sentence
from wllm.apps.liveavatar.backend.rocm.denoise_pp.worker import (
    CTRL_RUN_CHUNKS, CTRL_RESET, _audio_shape, _latent_shape,
)
from wllm.apps.liveavatar.backend.rocm.denoise_pp_sp.worker import (
    PPSPStepRank, DenoisePPSPWorker,
)
from wllm.apps.liveavatar.backend.rocm.stream_full.worker import StreamFullWorker

logger = init_logger(__name__)


class CombinedStepRank(PPSPStepRank):
    """Same 2D-parallel rank as denoise_pp_sp, but the chunk loop is STREAMING:
    it runs until a "more=0" flag arrives (SP-broadcast within the stage; P2P
    forwarded from the previous stage) instead of a fixed broadcast count."""

    def _run_chunks(self):
        prev0 = (self.stage - 1) * self.sp
        nxt0 = (self.stage + 1) * self.sp
        last0 = (self.n_stages - 1) * self.sp
        while True:
            more = torch.zeros(1, dtype=torch.int64, device=self.device)
            if self.sp_rank == 0 and self.stage > 0:
                torch.distributed.recv(more, src=prev0, group=self.pg)   # flag from prev stage sp0
            self.spg.broadcast(more, src=0)                              # fan flag to my stage
            if int(more.item()) == 0:
                if self.sp_rank == 0 and self.stage < self.n_stages - 1:
                    torch.distributed.send(more.contiguous(), dst=nxt0, group=self.pg)  # propagate stop
                break
            ci = self.counter
            latent = torch.empty(_latent_shape(self.cfg, ci), dtype=self.dtype, device=self.device)
            audio = torch.empty(_audio_shape(self.cfg), dtype=self.dtype, device=self.device)
            if self.sp_rank == 0 and self.stage > 0:
                torch.distributed.recv(latent, src=prev0, group=self.pg)
                torch.distributed.recv(audio, src=prev0, group=self.pg)
            self.spg.broadcast(latent, src=0)
            self.spg.broadcast(audio, src=0)
            latent, new_ref = self._run_stage_steps(latent, audio, ci)
            if self.sp_rank == 0:
                if self.stage < self.n_stages - 1:
                    torch.distributed.send(more.contiguous(), dst=nxt0, group=self.pg)   # more=1 ahead
                    torch.distributed.send(latent.contiguous(), dst=nxt0, group=self.pg)
                    torch.distributed.send(audio.contiguous(), dst=nxt0, group=self.pg)
                else:
                    torch.distributed.send(latent.contiguous(), dst=0, group=self.pg)    # final -> driver
            if ci == 0:
                nr = new_ref if new_ref is not None else torch.empty_like(self.ref)
                torch.distributed.broadcast(nr, src=last0, group=self.pg)                # one-time ref sync
                self.ref = nr
            self.counter += 1


class StreamPPSPWorker(DenoisePPSPWorker):
    STEP_MODULE = "wllm.apps.liveavatar.backend.rocm.stream_pp_sp.worker"

    # reuse stream_full's TTS chunk generator (unbound; only touches base-class attrs)
    def _tts_audio_stream(self, content):
        return StreamFullWorker._tts_audio_stream(self, content)

    async def _stream_pipeline(self, next_chunk, write=True):
        """Driver: pull step_samples-sized audio chunks from `next_chunk` (async) and
        feed them through the 2-stage×sp pipeline as they arrive, depth-`n_stages`."""
        pipe = self.pipe
        ctx = self._ctx
        tgt = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
        stage1_sp0 = self.sp
        last0 = (self.n_stages - 1) * self.sp
        DEPTH = self.n_stages
        fed = drained = 0
        pending = {}
        done_feeding = False
        while (not done_feeding) or pending:
            if (not done_feeding) and (fed - drained) < DEPTH:
                chunk = await next_chunk()
                if chunk is None:
                    done_feeding = True
                    stop = torch.zeros(1, dtype=torch.int64, device=self.device)
                    self.spg.broadcast(stop, src=0)                                    # stop stage-0 group
                    torch.distributed.send(stop.contiguous(), dst=stage1_sp0, group=self.pg)  # stop stage 1
                else:
                    ci = ctx.chunk_idx = pipe._session_ctx["latent_chunk_idx"]
                    noise = self._draw_noise(ci)
                    audio = self.extract_audio_features(chunk, target_frames=tgt)
                    more = torch.ones(1, dtype=torch.int64, device=self.device)
                    self.spg.broadcast(more, src=0)
                    self.spg.broadcast(noise, src=0)
                    self.spg.broadcast(audio, src=0)
                    latent, new_ref = self._run_stage0(noise, audio, ci)
                    torch.distributed.send(more.contiguous(), dst=stage1_sp0, group=self.pg)
                    torch.distributed.send(latent.contiguous(), dst=stage1_sp0, group=self.pg)
                    torch.distributed.send(audio.contiguous(), dst=stage1_sp0, group=self.pg)
                    buf = torch.empty(_latent_shape(self.cfg, ci), dtype=pipe.dtype, device=self.device)
                    req = torch.distributed.irecv(buf, src=last0, group=self.pg)
                    pending[ci] = (req, buf, chunk)
                    if ci == 0:
                        nr = new_ref if new_ref is not None else torch.empty_like(pipe._ref_latents)
                        torch.distributed.broadcast(nr, src=last0, group=self.pg)
                        pipe._ref_latents = nr
                    pipe._session_ctx["latent_chunk_idx"] = ci + 1
                    fed += 1
            if pending:
                dci = min(pending)
                req, buf, ch = pending[dci]
                if req.is_completed() or done_feeding or (fed - drained) >= DEPTH:
                    pending.pop(dci)
                    req.wait()
                    self._decode(buf, ch, write)
                    drained += 1

    async def _stream_from_tts(self, content):
        self._ensure_ctx()
        if not self._bcast_done:
            self._broadcast_session()
        self._bcast_ctrl(CTRL_RUN_CHUNKS)                 # step ranks enter streaming _run_chunks
        step_samples = int(self.cfg.tts_chunk_size)
        state = {"data": np.empty((0,), dtype=np.float32), "done": False}
        agen = self._tts_audio_stream(content)

        async def next_chunk():
            while state["data"].size < step_samples and not state["done"]:
                try:
                    r = await agen.__anext__()
                    state["data"] = np.concatenate([state["data"], r])
                except StopAsyncIteration:
                    state["done"] = True
            if state["data"].size >= step_samples:
                c = state["data"][:step_samples]
                state["data"] = state["data"][step_samples:]
                return c
            if state["data"].size > 0 and state["done"]:
                c = np.pad(state["data"], (0, step_samples - state["data"].size), mode="constant")
                state["data"] = np.empty((0,), dtype=np.float32)
                return c
            return None

        await self._stream_pipeline(next_chunk, write=True)

    def warmup(self):
        from wllm.serving.utils.rand import set_global_seed
        self._ensure_ctx()
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=self.cfg.image_path)
        self._broadcast_session()
        n_warm = max(3, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 1)
        self._bcast_ctrl(CTRL_RUN_CHUNKS)
        step_samples = int(self.cfg.tts_chunk_size)
        state = {"i": 0}

        async def next_chunk():
            if state["i"] >= n_warm:
                return None
            state["i"] += 1
            return np.zeros((step_samples,), dtype=np.float32)

        self._async_runner.run(self._stream_pipeline(next_chunk, write=False))
        self.pipe.reset()
        self._bcast_ctrl(CTRL_RESET)
        self._bcast_done = False

    def inference(self, audio):
        asr_results = self.asr_model.transcribe(audio=(audio[0].flatten(), audio[1]), language="English")
        text = asr_results[0].text
        self.message_history.append({"role": "user", "content": text})
        content = self._async_runner.run(self._generate_llm_response())
        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})
        self._async_runner.run(self._stream_from_tts(content))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-rank", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=6)
    ap.add_argument("--sp", type=int, default=3)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()
    if args.step_rank:
        CombinedStepRank(args.config, args.rank, args.world, args.sp, args.stages).serve()
    else:
        StreamPPSPWorker(cfg_path=args.config).loop()


if __name__ == "__main__":
    main()
