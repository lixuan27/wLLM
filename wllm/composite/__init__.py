"""Composite multimodal graph runtime.

A composite model (reasoner + generator + codecs + action head) is a
*component graph*; a request is a *walk* over that graph — sequential
steps, parallel branches, iterative loops (AR decode, diffusion denoise,
world rollout), and streaming edges. Components carry their own state
engines and placement; the same graph serves image, video, audio, and
action walks without duplicating the model.

    graph.py     typed components + edges, structural validation
    walk.py      walk algebra: Seq / Par / Loop / Stream
    executor.py  walk execution with session-state isolation + placement
    batching.py  cross-request step batching with per-request parity
    lowering.py  DeploymentPlan -> component placement, fail-closed
"""

from .graph import Component, ComponentGraph, Edge
from .walk import Loop, Par, Seq, Stream, Walk
from .executor import SessionStore, WalkExecutor
from .batching import StepBatcher
from .lowering import LoweringReport, lower_plan, require

__all__ = [
    "Component", "ComponentGraph", "Edge",
    "Seq", "Par", "Loop", "Stream", "Walk",
    "SessionStore", "WalkExecutor", "StepBatcher",
    "LoweringReport", "lower_plan", "require",
]
