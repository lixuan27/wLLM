"""wGraph — the typed, stateful, hierarchical IR of wLLM."""

from .program import Program
from .quality import QualityContract, QualityMode
from .regions import Node, NodeOp, Region, RegionKind
from .states import DeadlinePolicy, StateKind, StateScope, StateSpec
from .streams import Backpressure, Modality, StreamSpec

__all__ = [
    "Program",
    "QualityContract",
    "QualityMode",
    "Node",
    "NodeOp",
    "Region",
    "RegionKind",
    "DeadlinePolicy",
    "StateKind",
    "StateScope",
    "StateSpec",
    "Backpressure",
    "Modality",
    "StreamSpec",
]
