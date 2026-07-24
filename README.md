# wLLM: World and Multimodal Model Serving

> **wLLM is not an LLM-only engine.** It deploys autoregressive, diffusion,
> video, world, and world-action models — automatically.

Given a runnable multimodal model (an MLLM, a video generator, an
interactive world model, or a world-action policy), a target hardware
environment, and a set of SLOs, **wLLM synthesizes a verified, efficient
deployment plan** — placement, parallelism, streaming, batching — with
correctness, quality, and safety evidence attached to every plan.

## Components

| Component | Role |
|---|---|
| **wGraph** | Typed, stateful, hierarchical IR: execution regions (AR / diffusion / chunk-rollout / multi-agent / feedback), semantic state contracts (KV / recurrent / rolling-context / feedback-critical, with `verified` probe gating), rate-and-deadline-aware streams, exact-vs-bounded quality contracts |
| **Tessera Planner** | Budget-controlled deployment search: rule-based candidates from region semantics → constraint filtering (memory / state placement / deadlines, with rejection reasons) → analytic cost model (critical-path latency, bottleneck-resource period) → successive-halving measurement |
| **wRuntime** | Reference executor (always-correct fallback anchor), staged pipelines, cold/warm/hot lifecycle contract, deployment fingerprinting, plan fallback chain |
| **wBench** | Five-level verification: structural → numerical (fixed-seed) → application quality → stress → fault |

## Integration levels

- **L0 Opaque** — wrap any existing CLI / server / catalog runner as a
  subprocess with an artifact contract; optimize placement, replicas, and
  multi-model pipelines without touching model code.
- **L1 Pipeline** — `Application.from_callable(run, example_inputs=...)`:
  under 30 lines to get baselining, planning, and a reference fallback.
- **L2 Model-aware** — verified state/loop contracts unlock co-location vs
  disaggregation, sequence/step parallelism, and cross-chunk overlap.
- **L3 Kernel-native** — optional static-graph fast path (CUDA-graph
  capture, declarative weight mapping) for the hottest models.

Model support is reported honestly in six tiers —
`Discovered / Launchable / Profiled / Structure-aware / Optimized / Verified`
— never as a blanket "supported".

## Quick start (developer preview)

```python
from wllm.api import Application

app = Application.from_callable(run, example_inputs={"prompt": "..."})
report = app.baseline(repeats=5)                       # measured evidence
plans  = app.optimize(objective="first-output-latency", num_gpus=4)
print(plans.report())                                   # kept + rejected(why)
```

```bash
python -m pytest tests/          # 28 tests and counting
```

## Design principles

1. **Measured or it didn't happen** — plans are ranked by a cost model but
   promoted only by real end-to-end measurement.
2. **Verified contracts only** — an agent hypothesis about state semantics
   never unlocks a transformation; counterfactual probes do.
3. **Honest surfaces** — an opaque node exposes placement optimization
   only; wLLM never fabricates parallelism it cannot prove legal.
4. **Always a correct path** — optimized plan → last-known-good →
   reference; the user is never stranded.

## Status

Developer preview (alpha). Current coverage: catalog import for a
258-entry external model substrate, L0/L1 integration, toy-level exact
planning verified end-to-end, first real-model baselines in progress on
8× H200. APIs will change.

## License

Apache-2.0
