# Experiment log

Append one block per milestone, newest at the bottom. Do not rewrite
history — negative results are as load-bearing as positive ones.

Start with the IR conversion record (Phase 1–2), then append one
deployment iteration block per variant (Phase 3–4). The first
deployment iteration block must be a benchmark of the user's
reference backend; every subsequent variant's Δ is measured against
that entry. See `wllm/apps/AGENTS.md` for the full spec.

Every PASS / FAIL claim and every numerical result must point at a
live artifact on disk (log file, benchmark JSON, stored hash). An
entry without those pointers is a *proposal*, not a *result*, and
the variant is unmeasured until they exist.

## IR conversion record format:

```markdown
## IR Conversion  (YYYY-MM-DD)
- Worker graph: <N> operators, <M> edges
  - Black-box stages: …
  - Exposed stages: …
  - Streaming edges: …
- Model graph(s): <graph_name>: <N> operators, <M> edges
  - State objects: …
  - Chunk-periodic operators: …
- IR validation: PASS / FAIL
  - fixture: <path>
  - reference_run: <stdout/log path>
  - executor_output_hash: <sha256 or diff summary>
  - reference_output_hash: <sha256 or diff summary>
  - tolerance: <documented numerical tolerance and justification>
- Analysis summary:
  - Pipeline stages: …
  - Streaming overlaps: …
  - Critical path: …
  - Bottleneck stage: …
  - Key independent pairs: …
```

## Deployment iteration record format:

```markdown
## <variant_name>  (YYYY-MM-DD)
- Hypothesis: …
- IR analysis basis: …
- Optimization target(s) attacked: <one or more of the active
  targets for this session — under defaults, "latency-to-first-
  output", "sustainable-output-rate", or "both"; under a per-app
  override, name the user's target plus any defaults the variant
  also moves>
- Change relative to previous baseline: …
- Hardware: GPUs=…, attention_backend=…, other knobs=…
- Correctness: PASS / FAIL (tolerance=…, diff summary=…)
  - correctness_log: <path to the correctness harness log — required>
- t_first_out − t_in: median=…ms, p95=…ms
- Sustainable output rate: …<unit>/s  (omit if the pipeline emits
  exactly one output chunk per input; otherwise include the inter-
  chunk-gap median, p95, and over-budget fraction as the smoothness
  side condition)
- <user-named override target, if any>: …
- benchmark_json: <path under benchmarks/results/ containing the raw
  per-run timings, the steady-rate sweep, and metadata — required>
- Δ vs. previous baseline: <one entry per active target, named
  explicitly>: …
- Verdict: promote / discard / needs-followup
- Follow-ups: …
```
