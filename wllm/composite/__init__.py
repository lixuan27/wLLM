"""Composite multimodal graph runtime.

A composite model (reasoner + generator + codecs + action head) is a
*component graph*; a request is a *walk* over that graph — sequential
steps, parallel branches, iterative loops (AR decode, diffusion denoise,
world rollout), and streaming edges. Components carry their own state
engines and placement; the same graph serves image, video, audio, and
action walks without duplicating the model.

    graph.py     typed components + edges, structural validation
    walk.py      walk algebra: Seq / Par / Loop / Stream
    walks.py     named walk sets + per-request walk state machines
    executor.py  walk execution with session-state isolation + placement
    batching.py  cross-request step batching with per-request parity
    rendezvous.py  concurrent walks -> one batched step call
    chunking.py  chunk policies on streaming edges
    lowering.py  DeploymentPlan -> component placement, fail-closed
"""

from .graph import Component, ComponentGraph, Edge
from .walk import Loop, Par, Seq, Stream, Walk
from .walks import RequestResult, WalkSet, WalkStateMachine, run_request
from .executor import SessionStore, WalkExecutor, current_session
from .batching import StepBatcher, StepRequest
from .rendezvous import GateStats, StepGate
from .chunking import (ChunkedChannel, ChunkPolicy, FixedChunk,
                       LeftContext, SlidingWindow)
from .lowering import LoweringReport, lower_plan, require

__all__ = [
    "Component", "ComponentGraph", "Edge",
    "Seq", "Par", "Loop", "Stream", "Walk",
    "RequestResult", "WalkSet", "WalkStateMachine", "run_request",
    "SessionStore", "WalkExecutor", "current_session",
    "StepBatcher", "StepRequest", "GateStats", "StepGate",
    "ChunkPolicy", "ChunkedChannel", "FixedChunk", "LeftContext",
    "SlidingWindow",
    "LoweringReport", "lower_plan", "require",
]
