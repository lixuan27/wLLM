"""IR conversion for the WorldPlay pipeline (Phase 1-2).

`graph_builder.build_chunk_graph` / `build_worker_graph` construct the IR;
`pipeline_decomposed.WorldPlayDecomposedPipeline` provides the faithful
per-operator execution the ops call into, so `SequentialExecutor` reproduces
the reference backend's frames.
"""
