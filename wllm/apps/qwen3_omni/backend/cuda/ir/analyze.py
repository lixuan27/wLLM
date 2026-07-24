"""Print summarize_graph() for both Qwen3-Omni IR graphs.

Structural check + analysis dump (no models loaded). The per-op latency
numbers (if passed) come from real measurements recorded in the
experiment log; here we pass the measured per-stage / per-op latencies so
the critical-path / bottleneck sections are populated.
"""

from __future__ import annotations

import argparse
import json
import sys

from wllm.serving.ir import summarize_graph
from wllm.apps.qwen3_omni.backend.cuda.ir.graph_builder import (
    build_worker_graph, build_talker_model_graph,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latencies", default=None,
                    help="JSON file {op_name: seconds} for critical path")
    args = ap.parse_args()
    lat = None
    if args.latencies:
        with open(args.latencies) as f:
            lat = json.load(f)

    worker = build_worker_graph()
    talker = build_talker_model_graph()

    print("################ WORKER GRAPH ################")
    print(summarize_graph(worker, op_latencies=(lat or {}).get("worker")))
    print("\n################ TALKER MODEL GRAPH ################")
    print(summarize_graph(talker, op_latencies=(lat or {}).get("talker")))


if __name__ == "__main__":
    main()
