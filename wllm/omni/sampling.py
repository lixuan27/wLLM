"""Sampling parameters accepted by the in-tree omni engine.

Duck-typed on purpose: the engine reads attributes (``max_tokens``,
``temperature``, ``seed``, ...) off whatever object the caller passes,
so apps built against an external engine's params class run unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: int | None = None
    detokenize: bool = True
    stop_token_ids: tuple[int, ...] = ()


def read_param(params, name: str, default):
    """Attribute-or-dict lookup so foreign params objects work."""
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)
