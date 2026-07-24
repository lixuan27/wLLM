# wLLM: Agent-Operable Deployment Optimizer for World & Multimodal Models

> **wLLM is not an LLM-only engine — and not another inference engine.**
> Tell your coding agent *"optimize this project with wLLM"* and a real,
> agent-independent infrastructure inspects the project, plans legal
> optimizations, measures them, verifies quality, and leaves a receipt
> and a rollback path. The agent expresses intent; wLLM decides what is
> true.

Given a runnable multimodal model (an MLLM, a video generator, an
interactive world model, or a world-action policy), a target hardware
environment, and a set of SLOs, **wLLM synthesizes a verified, efficient
deployment plan** — placement, parallelism, streaming, batching — with
correctness, quality, and safety evidence attached to every plan.

## One sentence in, receipts out

```text
User → agent : "Optimize this video project. 4 GPUs, first-frame
                latency first, no quality loss."
Agent → wLLM : a typed OptimizeSpec + CLI calls (nothing else)
wLLM         : inspect → baseline → legal candidates → real measurement
               → quality gates → receipt → apply (reversible)
```

```bash
wllm inspect  .                          # evidence-listing project manifest
wllm plan     . --model <id> --spec s.yaml   # backends + legal passes (+why rejected)
wllm verify   --receipt r.json           # promote gate: measured, authentic, no fallback
wllm apply    . --receipt r.json         # fail-closed promotion
wllm rollback .                          # optimized → last-known-good → reference
wllm report   .
```

The same commands run identically in CI with no agent present — if it
only works when a model is improvising, it is a prompt, not infra.

Agents connect over MCP (stdio): point a client at the bundled server
and the six tools above appear with typed schemas — the server adds
transport, never judgment.

```json
{"mcpServers": {"wllm": {"command": "wllm-mcp"}}}
```

The receipt pipeline is dogfooded on real evidence: the measured
Wan2.2 CFG branch-parallel run (1.44× end-to-end, frame-level
bit-exact, 2×H200) imports into a promotable receipt, while the
batched-CFG variant from the same job (max deviation 251/255) is
refused by the same gate — see `scripts/receipt_wan22_cfgpar.py`.

## Components

| Component | Role |
|---|---|
| **Control plane** | Typed OptimizeSpec, project inspector (absence recorded, never guessed), declarative backend-capability registry (requires/conflicts/fail-closed log invariants), measured receipts with deployment fingerprints, apply/rollback state machine |
| **Composite runtime** | Component graphs with requests as walks (sequential / parallel / loop / streaming); placement is data, session state is hard-isolated with provable reset, streams carry bounded queues + backpressure, and cross-request step batching preserves per-request parity by construction |
| **Omni stage engine** | In-tree async multi-stage engine implementing the `AsyncOmni` contract the apps program against: stage-config YAMLs, continuous-batching AR scheduler + whole-request generation scheduler, pluggable model stages that fail closed on unregistered models (`stats().max_step_batch` is the batching authenticity signal) |
| **Technique executors** | Optimization techniques (step-residual cache, quantization simulation, …) that carry declared authenticity signals; an orchestrator runs every candidate against the frozen exact reference and rejects crashed / never-engaged / over-budget candidates with reasons — a technique can never grade itself |
| **wGraph** | Typed, stateful, hierarchical IR: execution regions (AR / diffusion / chunk-rollout / multi-agent / feedback), semantic state contracts (KV / recurrent / rolling-context / feedback-critical, with `verified` probe gating), rate-and-deadline-aware streams, exact-vs-bounded quality contracts |
| **Tessera Planner** | Budget-controlled deployment search: rule-based candidates from region semantics → constraint filtering (memory / state placement / deadlines, with rejection reasons) → analytic cost model (critical-path latency, bottleneck-resource period) → successive-halving measurement |
| **wRuntime** | Reference executor (always-correct fallback anchor), staged pipelines, cold/warm/hot lifecycle contract, deployment fingerprinting, plan fallback chain |
| **wBench** | Five-level verification: structural → numerical (fixed-seed, tie-aware) → application quality → stress → fault |

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

Model support is reported honestly in tiers —
`Discovered / Cataloged / Launchable / Parity-verified / Optimized / Serving-verified`
— never as a blanket "supported".

## Measured results so far (1–2× H200, all with on-disk evidence)

| Model | Best plan | Verdict |
|---|---|---|
| video 5B (TI2V) | CFG branch-parallel, 2 GPU | **1.44× E2E, frame-level bit-exact** |
| VLM 8B | static KV cache | **2.75×**, tie-aware exact (top-2 logit gap = 0.0 proven) |
| VLA 7B | native checkpoint precision | **4.59×** vs naive fp32 (the fp32 upcast was the variant, not the oracle) |

## Design principles

1. **Measured or it didn't happen** — plans are ranked by a cost model but
   promoted only by real end-to-end measurement; a receipt without
   performance distributions is void.
2. **Fail closed** — a silent fallback in the logs invalidates the whole
   result; authenticity checks must prove the optimization was active;
   an unknown model gets diagnose-only mode, never a fake win.
3. **Verified contracts only** — an agent hypothesis about state semantics
   never unlocks a transformation; counterfactual probes do.
4. **Always a correct path** — optimized plan → last-known-good →
   reference; the user is never stranded.

## Quality engineering

```bash
sbatch slurm/wllm_ci_cpu.sbatch   # or run the steps directly on any CPU box
```

The CI battery: naming/secret release gate → full syntax sweep → unit +
integration tests (pytest) → gherkin BDD scenarios driving the real CLI
(`tests/features/*.feature`, zero-dependency runner) → coverage gate
(control plane + data-plane pillars ≥ 85%) → mutation smoke (AST mutants
of the fail-closed core; kill rate ≥ 80% required). Independent
adversarial review is part of the process — findings land as pinned
regression tests, and `docs/RISK_REGISTER.md` maps every P0/P1 risk to
its concrete mitigation and its honest residual gap.

## Quick start (developer preview)

```python
from wllm.api import Application

app = Application.from_callable(run, example_inputs={"prompt": "..."})
report = app.baseline(repeats=5)                       # measured evidence
plans  = app.optimize(objective="first-output-latency", num_gpus=4)
print(plans.report())                                   # kept + rejected(why)
```

External engines and model substrates are bound by environment variables,
never imported by name — see `docs/ENGINES.md`.

## Status

Developer preview (alpha → beta). Current coverage: control plane v0
(inspect/plan/verify/apply/rollback + capability registry + receipts),
catalog import for a 258-entry external model substrate, L0/L1
integration, measured exact-plan speedups on three real models (table
above), first end-to-end app Launchable on the unified runtime. APIs
will change.

## License

Apache-2.0
