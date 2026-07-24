"""Whole-request stage scheduler (diffusion / codec / vocoder stages).

Requests complete in one scheduled execution rather than token steps;
batching groups whole requests admitted in the same step.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import BaseScheduler, ScheduledRequest


class OmniGenerationScheduler(BaseScheduler):
    def step(self, generate_batch: Callable[[list[ScheduledRequest]],
                                            list[Any]]
             ) -> list[ScheduledRequest]:
        """Run every admitted request to completion in one batched call.

        ``generate_batch`` returns one result payload per request; the
        payload lands in ``req.context['result']``.
        """
        self._admit()
        if not self.running:
            return []
        batch = list(self.running)
        try:
            results = generate_batch(batch)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"generate_batch returned {len(results)} results for "
                    f"{len(batch)} requests; refusing to guess the mapping")
        except Exception as exc:
            self._fail_batch(batch, exc)
            raise
        self.stats.record_step(len(batch))
        for req, result in zip(batch, results):
            req.context["result"] = result
            req.finished, req.finish_reason = True, "complete"
        finished = list(batch)
        self._retire_finished()
        return finished
