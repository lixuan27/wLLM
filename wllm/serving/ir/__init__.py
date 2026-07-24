"""Public entry points for the IR framework."""

from wllm.serving.ir.analysis import (
    CrossChunkDep,
    CrossChunkDepInfo,
    PipelineStage,
    StreamingOverlap,
    analyze_cross_chunk_dependencies,
    compute_critical_path,
    find_pipeline_stages,
    find_streaming_overlaps,
    summarize_graph,
)
from wllm.serving.ir.executor import SequentialExecutor
from wllm.serving.ir.graph import (
    IREdge,
    IRGraph,
    IROperator,
    OpType,
    StateObject,
    StateStore,
    StreamingInfo,
    StreamingPattern,
    StreamMode,
    TensorPort,
)
from wllm.serving.ir.ops_base import ir_operator

__all__ = [
    "CrossChunkDep",
    "CrossChunkDepInfo",
    "IREdge",
    "IRGraph",
    "IROperator",
    "OpType",
    "PipelineStage",
    "SequentialExecutor",
    "StateObject",
    "StateStore",
    "StreamMode",
    "StreamingInfo",
    "StreamingOverlap",
    "StreamingPattern",
    "TensorPort",
    "analyze_cross_chunk_dependencies",
    "compute_critical_path",
    "find_pipeline_stages",
    "find_streaming_overlaps",
    "ir_operator",
    "summarize_graph",
]
