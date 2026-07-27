# wLLM Alpha Report — Three-Model Measured Optimization (2026-07-24)

Single node, 1× H200 per run. Every number is from a real SLURM job with
on-disk evidence (`benchmarks/results/*.json`, `logs/`). All plans were
produced by wLLM's own machinery (L1 `Application` wrap / plan family
generation / `successive_halving` measurement / verifier gates) — no
hand-tuned deployments.

> **Correction posted 2026-07-27 — the Qwen3-VL row below is WITHDRAWN.**
> This report is left as written, because a dated report that is edited to
> match later knowledge stops being evidence of anything. Read the row,
> then read this.
>
> **UPDATE, job 202503: the claim is now DISPROVEN, not merely unproven.**
> A position census refuses it as a real divergence — 83 of 128 positions
> disagree (65%), deciding gaps 0.25/17.75/22.25 against ε=1e-3. The
> original description ("a single greedy flip at gap 0.0") is contradicted,
> not merely unreproduced.
>
> How it got there, in order. The 2.75× `static KV cache` result was
> accepted as *tie-aware exact* under a rule that adjudicated **one**
> disputed token on the teacher-forced **prefill** path. Job 202244 could
> not re-adjudicate it at all — the adjudicator crashed on this model's
> multimodal path — so the claim came out as *unproven*, on the rule that
> no evidence is not a pass. Job 202503, with the adjudicator repaired
> and ruling on a census, refused it outright as above.
>
> Two further findings explain why the original number looked the way it
> did. On this runtime, requesting a static cache **silently also enables
> a compiled forward**, so a leg labelled "static KV cache" measures a
> composition; and the separation experiment (job 202328) shows the cache
> **alone is 0.87× — slower** — with the entire speedup belonging to the
> compiled forward, which is already classified *bounded* by measurement
> and whose drift signature is exactly one bf16 ULP. The pass was very
> likely misnamed as well as mismeasured.
>
> Consequence for the headline: with this row withdrawn, the "median
> 2.75× across three models" sentence below no longer has three
> substantiated models behind it.

## Scorecard

| Model (archetype) | Naive baseline | Best plan | Speedup | Verdict class | Jobs |
|---|---|---|---|---|---|
| Wan2.2-TI2V-5B (chunk-rollout video diffusion) | 5615 ms /33f@480×832×20st | 4021 ms (`compile max-autotune-no-cudagraphs`) | **1.43×** | bounded-drift (trajectory-divergent; frame drift mean 7.4/255 documented, refused by exact gate, promotable only under a video-quality contract) | 195301 → 195356 → 195374 |
| Qwen3-VL-8B (MLLM AR decode) | 2668 ms /128 tok | 968 ms (`static KV cache`) | **2.75×** | tie-aware exact (single greedy flip at a proven argmax tie: top-2 logit gap = 0.0; probe artifact) | 195433 + tie probe |
| OpenVLA-7B (VLA action prediction, own-env worker) | 133 ms /action (naive fp32 load) | 29 ms (`native bf16 [+sdpa]`) | **4.59×** | native-precision restoration (checkpoint stores bf16 — `torch_dtype: bfloat16`, 13 GB/7B params; fp32 "reference" is an upcast; task-level rollout validation staged for Beta) | 195638 → r2 → 195701 |

Median speedup across the three models: **2.75×** (plan target: ≥1.3×).
Every model individually ≥1.3× (minimum 1.43×).

## What the verifier gates caught (the load-bearing part)

1. **Diffusion**: torch.compile — conventionally "exact" — amplifies bf16
   kernel-fusion reordering across 20 denoise steps into pixel-level
   trajectory divergence. The exact gate refused it; the plan is honestly
   classed bounded-drift. Also: `reduce-overhead` cudagraphs measured
   *slower* (7.1 s) due to re-recording — negative result retained.
2. **AR decode**: the one token that differed under static cache sits at
   a *perfect argmax tie* (top-2 logit gap 0.0). Cross-numerics
   token-equality gates must be tie-aware; a flip at a zero-gap position
   is arbitration, not divergence.
3. **VLA**: the correctness reference itself was mis-specified — the
   checkpoint's declared dtype (bf16) is the semantic reference; an fp32
   upcast is a variant, not the oracle.

Rule extracted for wBench: **strict equality only against like-for-like
numerics at the checkpoint's declared precision; everything else gets a
documented tolerance class and, where trajectories diverge, an
application-level quality contract.**

## Alpha functional checklist

| Item | Status |
|---|---|
| External catalog read (258 manifests, machine inventory) | ✅ |
| Opaque subprocess runner (L0, kill-tree hygiene) | ✅ |
| Python callable integration (L1, <30 lines) | ✅ |
| wGraph v0 (regions/states/streams/quality) | ✅ 11 tests |
| Baseline profiler (median/p95 + JSON evidence) | ✅ |
| Exact placement/pipeline search (rules+constraints+cost model) | ✅ toy-verified |
| Reference fallback | ✅ |
| Unified in-house engines (serving runtime + native engine) | ✅ 2370 files, gated rename |
| Test suite | 34 green |

## Known gaps (honest tier)

- Wan2.2 **exact-only** speedup is still 1.0×; the exact path to ≥1.3× is
  L2 sequence-parallel over 2 GPUs (serving runtime's SP groups) — Beta.
- The native engine (static-graph/C-ABI subsystem) has no sm_90 dispatch;
  runtime validation needs SM89/110/120 hardware or a Hopper-enablement
  pass (Beta wKernels workstream).
- Integrated apps (5) are code-landed but not yet runtime-validated in
  this tree; staged as support-tier work (Cataloged → Launchable).
- OpenVLA bf16 plan needs task-level (LIBERO-class) success-rate
  validation before the precision-restoration verdict is final.
