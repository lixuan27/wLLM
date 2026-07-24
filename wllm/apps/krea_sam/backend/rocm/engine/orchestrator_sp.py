"""Sequence-parallel Krea orchestrator.

Extends KreaOrchestrator to run the Krea pipeline across a `world`-rank SP group
(DiT frame-sharding + VAE-decoder width-tiling are handled *inside* the shared
models via the SP / world groups). Rank 0 owns all I/O + SAM coordination; the
other ranks are pure compute followers. Each chunk, rank 0 reads the input,
broadcasts a compact command + the input tensor, and all ranks run pipe.step in
lockstep (the SP collectives synchronize them).

Correctness: pipe.step is the same reference computation; SP only distributes it
(mathematically equivalent, bf16-shard differences are the documented tolerance).
"""

from __future__ import annotations

import time

import torch

from wllm.serving.distributed.communication_op import global_broadcast
from wllm.serving.logger import init_logger
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator import KreaOrchestrator, READY_MARKER

logger = init_logger("orchestrator_sp")

IDLE, START, TERMINATE, RESET, CHUNK = 0, 1, 2, 3, 4


class KreaOrchestratorSP(KreaOrchestrator):
    def _bcast_cmd(self, action: int, T: int):
        t = torch.tensor([action, T], device=self.device, dtype=torch.int64)
        global_broadcast(t, src=0)
        return int(t[0].item()), int(t[1].item())

    def _bcast_input(self, T: int, krea_input):
        if not self.is_lead:
            krea_input = torch.empty((T, 3, self.cfg.height, self.cfg.width),
                                     device=self.device, dtype=self.pipe.dtype)
        global_broadcast(krea_input, src=0)
        return krea_input

    def loop(self):
        while True:
            action, T = IDLE, 0
            polled = None
            if self.is_lead:
                op = int(self.ctrl_buffer.recv())
                if op == 2:
                    action = TERMINATE
                elif op == 1 and not self.session_started:
                    action = START
                elif op == 3 and self.session_started:
                    action = RESET
                elif self.session_started:
                    polled = self._poll_input_frames()
                    if polled is not None:
                        action = CHUNK
                        T = int(polled[1].shape[0])
            action, T = self._bcast_cmd(action, T)

            if action == TERMINATE:
                self.terminate()
                break
            elif action == START:
                self.start()
            elif action == RESET:
                self.reset()
            elif action == CHUNK:
                krea_input = polled[0] if self.is_lead else None
                raw = polled[1] if self.is_lead else None
                krea_input = self._bcast_input(T, krea_input)
                mask_start = 0
                if self.is_lead:
                    self.sam_link.push_frames(raw)
                    mask_start = self._sam_push_count
                    self._sam_push_count += int(raw.shape[0])
                self._process_chunk(krea_input, raw, mask_start)
            else:
                time.sleep(0.005 if not self.session_started else 0.002)

    def _process_chunk(self, krea_input, raw, mask_start):
        """Whole-chunk path: all ranks run pipe.step (SP), lead composites."""
        krea_frames = self.pipe.step(krea_input)   # SP collective on all ranks
        if self.is_lead:
            self._finish_chunk(krea_frames, raw, mask_start)

    def _finish_chunk(self, krea_frames, raw_frames_np, mask_start):
        if krea_frames is None or len(krea_frames) == 0:
            return
        n_pushed = int(raw_frames_np.shape[0])
        masks = self.sam_link.read_masks(mask_start, n_pushed)
        if self._output_frame_skip_frames > 0:
            skip = min(self._output_frame_skip_frames, int(krea_frames.shape[0]))
            krea_frames = krea_frames[skip:]
            raw_frames_np = raw_frames_np[skip:]
            if masks is not None:
                masks = masks[skip:]
            self._output_frame_skip_frames -= skip
            if krea_frames.shape[0] == 0:
                return
        n_out = int(krea_frames.shape[0])
        originals = raw_frames_np[:n_out] if raw_frames_np.shape[0] >= n_out else None
        if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
            self.video_buffer.write(krea_frames)
            return
        masks = masks[:n_out] if (masks is not None and masks.shape[0] >= n_out) else None
        self.video_buffer.write(self._composite(krea_frames, originals, masks))
