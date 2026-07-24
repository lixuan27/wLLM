"""Build the LiveAvatar IR graphs and print the analysis (summarize_graph).

Pure structure + analysis; no model weights / GPU needed. Optionally accepts a
JSON of measured per-op latencies to compute the critical path and pipeline
bottleneck.

Usage:
    python wllm/apps/liveavatar/backend/ir/analyze.py [op_latencies.json]
"""
import sys
import json

from wllm.serving.rt_config import RTConfig
from wllm.serving.ir import summarize_graph
from wllm.apps.liveavatar.backend.cuda.ir.graph_builder import build_model_graph, build_worker_graph

CFG = "wllm/apps/liveavatar/backend/configs/reference.yaml"


def main():
    cfg = RTConfig.from_yaml(CFG, is_path=True)
    lat = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lat = json.load(f)

    print("derived dims:",
          f"kv_spatial={cfg.kv_spatial} latent_h={cfg.latent_height} "
          f"latent_w={cfg.latent_width} kv_cond_tokens={cfg.kv_cond_tokens} "
          f"sf_t={cfg.vae_config.scale_factor_temporal} "
          f"num_steps={cfg.num_inference_steps} chunk={cfg.chunk_size}")
    print("\n" + "#" * 70)
    print("# WORKER GRAPH")
    print("#" * 70)
    print(summarize_graph(build_worker_graph(cfg)))

    print("\n" + "#" * 70)
    print("# MODEL GRAPH (exposed sound-to-video)")
    print("#" * 70)
    print(summarize_graph(build_model_graph(cfg), op_latencies=lat or None))


if __name__ == "__main__":
    main()
