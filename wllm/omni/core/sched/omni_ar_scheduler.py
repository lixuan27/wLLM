"""Autoregressive stage scheduler: continuous batching at token steps.

Every ``step`` runs one decode step for *all* running requests in a
single batched call; new requests join between steps (continuous
admission) and finished requests retire without stalling the batch.
"""

from __future__ import annotations

from typing import Callable

from .base import BaseScheduler, ScheduledRequest


class OmniARScheduler(BaseScheduler):
    def step(self, decode_batch: Callable[[list[ScheduledRequest]], list[int]]
             ) -> list[ScheduledRequest]:
        """One decode step; returns requests that finished this step.

        ``decode_batch`` receives the running requests and returns one
        next-token id per request, in order.
        """
        self._admit()
        if not self.running:
            return []
        batch = list(self.running)
        try:
            tokens = decode_batch(batch)
            if len(tokens) != len(batch):
                raise RuntimeError(
                    f"decode_batch returned {len(tokens)} tokens for "
                    f"{len(batch)} requests; refusing to guess the mapping")
        except Exception as exc:
            # fail the whole batch and retire it: a poisoned request must
            # not stay resident and re-crash every later step
            self._fail_batch(batch, exc)
            raise
        self.stats.record_step(len(batch))
        finished: list[ScheduledRequest] = []
        for req, tok in zip(batch, tokens):
            req.output_token_ids.append(int(tok))
            if tok in req.stop_token_ids:
                req.finished, req.finish_reason = True, "stop"
            elif len(req.output_token_ids) >= req.max_tokens:
                req.finished, req.finish_reason = True, "length"
            if req.finished:
                finished.append(req)
        self._retire_finished()
        return finished
