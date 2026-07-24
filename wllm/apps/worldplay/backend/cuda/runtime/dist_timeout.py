"""Widen the torch.distributed process-group timeout for the WorldPlay workers.

Why this exists (cold-cache rank desync):
On a fresh node with a cold inductor/Triton cache, the ranks compile + autotune
their kernels at very different rates during the worker's startup `warmup()`
(the stage-split worker is worst: DiT ranks and VAE ranks run *different* code).
A rank that finishes compiling early reaches a collective and blocks waiting for
a still-compiling peer. The default PyTorch PG watchdog (NCCL ~10 min, gloo
~30 min) then aborts that collective (SIGABRT / rc=1) before the slow rank
finishes — even though it *would* finish given a few more minutes. Observed on a
cold p5en.48xlarge: stage_split_sp_4g/6g/8g died this way (an `_ALLGATHER_BASE`
on the SP sub-group ran 636 s before the NCCL watchdog killed it).

The fix is to widen the PG timeout so a legitimately-progressing-but-desynced
cold start is waited out instead of killed. The sub-group timeouts are the ones
that matter (the SP all-gather, the VAE-tile all-gather, the gloo object
broadcast) and the shared `GroupCoordinator` creates those via
`torch.distributed.new_group(ranks, backend=...)` with no timeout argument
exposed — so we wrap `new_group` (and `init_process_group` for the world group)
to inject a large default `timeout`. On a warm cache this is a no-op: nothing
compiles during warmup, no rank lags, and collectives complete in microseconds
regardless of the ceiling.

Trade-off: a *genuine* deadlock now takes WORLDPLAY_PG_TIMEOUT_S to surface
instead of ~10 min. That is the intended cost of tolerating slow cold compiles.
"""
from __future__ import annotations

import os
import functools
import datetime

import torch.distributed as dist

DEFAULT_PG_TIMEOUT_S = 5400  # 90 min — generous enough for a cold full-model compile


def patch_pg_timeout(seconds: int | None = None) -> int:
    """Monkeypatch torch.distributed.{new_group,init_process_group} so every
    process group (world + every SP/TP/DP sub-group, NCCL and gloo) is created
    with a large default timeout. Idempotent. Returns the timeout used."""
    secs = int(seconds if seconds is not None
               else os.environ.get("WORLDPLAY_PG_TIMEOUT_S", DEFAULT_PG_TIMEOUT_S))
    tmo = datetime.timedelta(seconds=secs)
    for name in ("new_group", "init_process_group"):
        orig = getattr(dist, name, None)
        if orig is None or getattr(orig, "_worldplay_timeout_patched", False):
            continue

        @functools.wraps(orig)
        def wrapped(*args, _orig=orig, _tmo=tmo, **kwargs):
            kwargs.setdefault("timeout", _tmo)  # don't override an explicit caller timeout
            return _orig(*args, **kwargs)

        wrapped._worldplay_timeout_patched = True
        setattr(dist, name, wrapped)
    return secs
