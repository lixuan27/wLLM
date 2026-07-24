# wLLM Risk Register — P0/P1 Technical Risks vs. Current Implementation

Status: 2026-07-25 (M17). Sources: the internal risk analysis (upstream-a/b/c
references digested in `ref/`), `WLLM_SPEC.md` v2 (incl. §3.6 data-plane
pillars), `docs/BETA_REPORT.md` (verifier laws + support-tier ledger), and the
code under `wllm/`.

Reading rule: **"Mitigation" means code or evidence that exists in this repo
today; "Residual gap" means what is honestly not built.** Nothing in this
register claims coverage that has no file pointer or on-disk evidence behind it.

Role names used: Compiler/IR Lead, GPU Runtime, Distributed Systems, Stateful
Systems, ML Systems/Evaluation, Control Plane/SRE, Security, Model Integration.

---

## 1. P0 risks

| # | Risk | Current mitigation (concrete pointers) | Residual gap | Owning role |
|---|---|---|---|---|
| P0-1 | **Semantic extraction / IR** — a graph that is structurally valid but semantically wrong; the most dangerous failure "still runs". Arbitrary Python (dynamic imports, monkey patches, side effects) cannot be fully auto-understood. | wGraph IR makes loops, state, streams, and quality contracts first-class (`wllm/graph/regions.py`, `states.py`, `streams.py`, `quality.py`), not a pure tensor DAG. The composite runtime executes declared `ComponentGraph` + `Walk` with structural validation, disjoint parallel joins, bounded loops with `until` guards, and pinned-placement enforcement (`wllm/composite/graph.py`, `walk.py`, `executor.py`). Project discovery is evidence-listing, never guessing: everything undetected lands in `unknowns` (`wllm/control/inspect.py`), and `wllm plan` for an unknown model is diagnose-only (`tests/test_control_plane.py::test_cli_plan_unknown_model_diagnose_only`). Spec §5 mandates four-way semantic recovery (static + trace + agent hypothesis + counterfactual-probe adjudication) with contracts on disk; state fields carry a `verified` flag the planner alone may consume. | No general static-analysis + dynamic-tracing extraction pipeline; the inspector is glob/regex heuristics. No reference interpreter, no side-effect modeling, no IR differential validation across multi-chunk / concurrency / cancel / long-run scenarios. `contracts/` is effectively empty — semantic recovery for real user projects is design, not implementation. | Compiler/IR Lead |
| P0-2 | **Correctness + quality oracle** — speedups obtained by silently degrading sampling, precision, or functionality; generative models have no simple equality oracle. | Candidates can never grade themselves: the technique orchestrator owns the exact reference, shared inputs, and the quality budget; crash, shape drift, budget violation, and failure-to-engage are each hard rejections with reasons (`wllm/techniques/orchestrator.py`, `wllm/techniques/base.py` — `QualityBudget.exact()` is the zero budget). Exact/bounded are strictly separated at the spec level: bounded requires an explicit `{metric, max}` budget, exact must not carry one (`wllm/control/spec.py::validate`), and `legal_passes` rejects any non-exact pass under the exact policy (`wllm/control/registry.py`). Level-B numerical comparison detects structural mismatch plus tolerance breaks (`wllm/verify/numerical.py`). Four empirically derived Verifier Laws with disk evidence are already codified (`docs/BETA_REPORT.md`): compiled diffusion diverges by trajectory; ties are arbitration, not divergence; the reference is the checkpoint's declared precision; batching changes results but distribution need not. | No perceptual or task-level oracle is wired in: no LPIPS/temporal-warp/flicker metrics, no public video-quality benchmark suite, no VLM side-by-side judge, no closed-loop simulator success gate, no multi-seed protocol, no per-(model, task) tolerance profiles. wBench levels C–E exist as spec (`WLLM_SPEC.md` §8), not as code. | ML Systems/Evaluation |
| P0-3 | **Session / persistent state** — KV, latents, and action history leaking across sessions; long-horizon drift; wrong robot actions; migration corrupting state. | Hard isolation with provable reset: `SessionStore` keys all state by `(session, component)` under a lock; `reset` provably clears (`wllm/composite/executor.py`); pinned by `tests/test_composite.py::test_session_state_isolation_and_reset`. State taxonomy declares scope / owner / ordered / recomputable / migratable / forkable per state (`wllm/graph/states.py`, `WLLM_SPEC.md` §4). Stream edges are bounded queues with declared backpressure policies (block / drop_oldest / coalesce / reject), overflow behavior tested (`test_stream_backpressure_reject_counts`). "state shared across sessions" is a forbidden-log invariant — a hit invalidates the measurement (`wllm/control/registry_data/wllm_composite.yaml`). | The store is in-process and CPU-side only: no GPU-resident state management, no session lease / generation numbers / idempotency keys / WAL, no snapshot–restore–fork protocol implemented, no migration, no rolling-upgrade schema versioning, no long-run session-churn tests (wBench level D is designed, not built). | Stateful Systems (with Distributed Systems) |
| P0-4 | **Benchmark trustworthiness** — measurement self-deception: cold start mixed in, silent backend fallback, unequal configs; one bad number poisons every conclusion. | No receipt, no claim: promotion requires a real measured distribution (p50/p95 present and positive), *all* authenticity checks passing and non-empty, zero forbidden-log hits, and a quality verdict compatible with the policy (`wllm/control/receipt.py::promote_problems`). A single forbidden-log-pattern hit invalidates the whole measurement regardless of the numbers (`wllm/control/registry.py::scan_log`, patterns shipped per backend in `wllm/control/registry_data/*.yaml`). The deployment fingerprint pins backend+version, source/model revision, hardware, driver, torch, precision, and passes; loading a tampered or schema-drifted receipt raises (`receipt.py::load`). Optimizations must emit machine-checkable engagement counters: `steps_reused` (`wllm/techniques/step_cache.py`), `layers_quantized` (`wllm/techniques/quant_sim.py`), `max_step_batch` (`wllm/omni/engine.py::stats`), `cross_signature_mixes()==0` (`wllm/composite/batching.py`), per-invocation device records (`executor.py::devices_used`). All Beta numbers trace to real SLURM jobs; negative results are retained (`docs/BETA_REPORT.md`). | Receipts do not yet capture a locked measurement environment (GPU clocks, power limits, co-tenant processes, NUMA/affinity); cold/warm separation is a schema convention, not enforced; no statistical noise estimation or repeat-count policy at the receipt level; stateful/load benchmarking (p99 under burst, inter-chunk jitter, 24 h soak) is queued, not built. | Control Plane/SRE |
| P0-5 | **Agent safety boundary** — an agent that can edit the baseline, thresholds, or verifier can "prove its own success"; user repos and checkpoints are untrusted input. | Product axiom enforced in code: the agent's only product is a typed `OptimizeSpec`; the optimizer never reads natural language, validation is fail-closed, and unknown fields are rejected (`wllm/control/spec.py`). The same CLI runs agent-free in CI (`wllm/control/cli.py`; `tests/test_control_plane.py::test_cli_end_to_end_smoke`). Verdicts are owned by the orchestrator/verifier, never the candidate (`wllm/techniques/orchestrator.py`). `apply` raises `PermissionError` on any promote problem, and the rollback chain guarantees the reference path can never be rolled away (`wllm/control/state.py::DeployManager`). Receipts are tamper-evident via fingerprint check on load (`receipt.py`). Agent/template output is confined by contract to `.wllm/generated/` (`WLLM_SPEC.md` §3.3, §6). | The `.wllm/generated/` confinement is a convention, not sandbox-enforced: no container / filesystem jail / network policy / syscall limits / resource quotas around agent-generated code; no independent execution service (verifier runs as the same user/process); no supply-chain scanning of untrusted repos or checkpoints; fingerprints are integrity hashes, not signatures. | Security (with Control Plane/SRE) |

---

## 2. P1 risks

| # | Risk | Current mitigation (concrete pointers) | Residual gap | Owning role |
|---|---|---|---|---|
| P1-1 | **Multi-backend semantic alignment** — the same checkpoint behaves differently across engines (templates, schedulers, sampling defaults, precision, guardrails, RNG order): "API works" ≠ "results equivalent". | Per-backend declarative capability files distinguish `models.exact` from `models.compatible` and declare modalities (`wllm/control/registry.py`, `wllm/control/registry_data/*.yaml`); `rank_backends` prefers exact tiers; `legal_passes` fails closed on any context fact a pass requires but the caller did not state. External engines are never imported by name: env-var late binding fails closed with `OmniEngineNotBound` when unbound (`wllm/engines/omni.py`, `docs/ENGINES.md`). The in-tree omni engine implements the exact app-facing contract, so vendors are interchangeable behind a fixed interface (`wllm/omni/engine.py`). Verifier Law 3 fixes the cross-backend reference point: the checkpoint's declared precision, not an fp32 upcast. | No cross-backend differential-parity harness; per-(model, mode, backend-version) defaults (prompt template, scheduler, sampler, VAE dtype, guardrail state) are not yet explicitly materialized and diffed; only in-tree backends have capability files. | Model Integration |
| P1-2 | **Compile / dynamic shapes / CUDA-graph** — graph breaks, recompilation storms, and capture constraints degrade tail latency or silently change numerics. | Compile is a registry pass, not a boolean: `full_graph_capture` requires `static_shapes: true` and fails closed when the fact is absent (`wllm/control/registry_data/wllm_native.yaml`, `registry.py::_requires_problem`). The measured discipline already caught the trap: the single-GPU compile plan was classified **bounded-drift and refused by the exact gate**, documented rather than shipped (`docs/BETA_REPORT.md` scorecard). Verifier Law 1 requires empirical exactness classification per (model, precision, loop depth). The native engine holds Hopper probe tier (SM90 dispatch verified). | No shape-bucket manager, no compile-artifact cache/eviction, no graph-break or recompilation telemetry, no bucketed capture with safe fallback; the HOT-phase no-recapture/no-allocate contract (`WLLM_SPEC.md` §7) is spec-only. | GPU Runtime |
| P1-3 | **Memory / admission control** — peak VRAM is not just weights; fragmentation and hidden pools cause OOM, and post-OOM allocator state is unreliable. | Planner constraint filtering rejects infeasible plans with explicit reasons (`wllm/planner/constraints.py`); state specs carry `memory_bytes` for accounting (`wllm/graph/states.py`); OOM/crash pressure is a designed wBench level-D case; the field lesson "inference-mode discipline beats memory knobs" (an apparent VAE OOM was autograd retaining ~140 GB) is recorded in `docs/BETA_REPORT.md`. | No memory estimator combining static estimates + historical peaks + warmup measurement + margin; no admission controller in a serving path; no OOM recovery state machine (retry vs. session invalidation vs. worker rebuild). | GPU Runtime (with Distributed Systems) |
| P1-4 | **Multi-GPU topology / communication** — placement mismatched to interconnect makes multi-GPU slower than single; collective ordering bugs hang or corrupt. | Placement is data, not code: every invocation records its device, plans are auditable, and pinned components reject wrong placement (`wllm/composite/executor.py`, `tests/test_composite.py::test_placement_recorded_and_pins_enforced`). The shipped multi-GPU win was chosen for provability: one-CFG-branch-per-GPU has no cross-rank reductions and is frame-level **bit-identical** (Verifier Law 4; measured 1.74× denoise, 1.44× E2E on 2 GPUs, `docs/BETA_REPORT.md`). Scope is deliberately single-node (`WLLM_SPEC.md` §12). | No hardware graph: the inspector lists GPU names only (`wllm/control/inspect.py::_probe_gpus`) — no link/NUMA/PCIe discovery, no measured bandwidth table, no communication cost model, no collective-hang watchdog/abort machinery. | Distributed Systems |
| P1-5 | **Versioned capability registry** — backend upgrades silently invalidate recipes; "supported: true" without evidence rots. | Capability files carry a `version` field and pass-level quality/requires/conflicts plus forbidden-log invariants (`wllm/control/registry_data/`); the support ladder `Discovered → Cataloged → Launchable → Parity-verified → … → Production` requires on-disk evidence per promotion (`WLLM_SPEC.md` §3.5; ledger in `docs/BETA_REPORT.md`). Receipts pin `backend_version`, and any key-field change invalidates the fingerprint (`wllm/control/receipt.py`). Catalog asset readiness is content-level, not presence-level: truncated shards, proxy stubs, and corrupt JSON become explicit blockers (`wllm/backends/catalog/assets.py`, `tests/test_catalog_assets.py`). | No version-range matching, no `commit_tested`/`last_validated` evidence bound to registry entries, no compatibility CI, no profile-retirement process; registry covers in-tree backends only. | Control Plane/SRE |

---

## 3. Accident modes → guards (mirror of the reference §16 table)

| Accident mode | Surface symptom | Our concrete guard | Guard status |
|---|---|---|---|
| Faster but quality degraded | benchmark all green | Quality policy is typed and immutable by the optimizer: exact policy cannot carry a budget, bounded requires an explicit one (`wllm/control/spec.py`); non-exact passes rejected under exact policy (`wllm/control/registry.py::legal_passes`); orchestrator budget enforcement (`wllm/techniques/orchestrator.py`) | Built |
| Session cross-talk | single request fine, multi-user wrong | `(session, component)`-keyed store with provable reset (`wllm/composite/executor.py`); pinned by `tests/test_composite.py::test_session_state_isolation_and_reset`; forbidden pattern "state shared across sessions" (`registry_data/wllm_composite.yaml`) | Built (in-process scope) |
| Multi-GPU slower than single-GPU | GPU util looks high | Promotion demands a measured E2E receipt vs. baseline (`wllm/control/receipt.py`); shipped plan is the provably communication-free one (bit-exact branch-per-GPU, `docs/BETA_REPORT.md`) | Partial — topology-aware planning not built |
| `torch.compile` slower/wrong in production | offline test looked faster | Exact gate refused the compile plan on measured drift; the refusal is documented, not hidden (`docs/BETA_REPORT.md` scorecard; Verifier Law 1) | Partial — no recompile/graph-break telemetry |
| CUDA-graph sporadic errors | most requests fine | `full_graph_capture` pass fails closed unless `static_shapes: true` is stated (`registry_data/wllm_native.yaml` + `registry.py::_requires_problem`) | Gap — bucketed capture + safe fallback not built |
| Persistent anomalies after OOM | first OOM visible, rest silent | wBench level-D OOM/crash cases and runtime fallback chain (optimized → last-known-good → reference) are specified (`WLLM_SPEC.md` §7–8; `wllm/control/state.py` implements the chain at deploy level) | Gap — worker-level recovery not implemented |
| Fake performance data | impressive report | Forbidden-log invariant: one hit invalidates the measurement (`wllm/control/registry.py::scan_log`); receipts with `fallback_hits` cannot be promoted (`receipt.py::promote_problems`); engagement counters mandatory (`wllm/techniques/`, `wllm/omni/engine.py::stats`, `wllm/composite/batching.py`) | Built |
| Wrong action behavior, correct shapes | tensors look fine | Reference-precision law prevents oracle inflation (OpenVLA 4.59× is booked as *native-precision restoration*, not a speedup over an fp32 "oracle") — `docs/BETA_REPORT.md` | Partial — closed-loop simulator gate pending (two-env bridge in next queue) |
| Rolling upgrade loses sessions | stateless traffic unaffected | State specs declare `migratable`/`forkable` (`wllm/graph/states.py`) so the planner cannot assume migratability | Gap — snapshot schema versioning/migration not built |
| Agent "fixes" the failing test | PR looks all green | Optimizer reads typed spec only (`spec.py`); orchestrator owns reference/budget (`orchestrator.py`); receipts tamper-evident on load (`receipt.py`); apply is a hard `PermissionError` gate (`state.py`); CLI runs agent-free in CI (`cli.py`) | Built at protocol level — sandbox enforcement still missing (see P0-5 gap) |

---

## 4. Build-order compliance

The reference build order says: **build the proving system first, then widen
optimization capability** — and explicitly defers multi-node, lossy passes,
automatic CUDA-kernel generation, session migration, and "arbitrary Python
project" auto-IR out of the first phases.

Current stance — compliant:

- **Multi-node: deferred.** All code paths are single-node; multi-node NCCL
  placement is not attempted anywhere (`WLLM_SPEC.md` §12; cluster redlines
  independently forbid unproven multi-node jobs).
- **Lossy passes: deferred.** Quality policy defaults to `exact`; `bounded`
  demands an explicit budget; FP8/NVFP4/KV-quant are scheduled for 1.0 and do
  not exist as promotable passes today. The only quantization in-tree is a
  declarative int8 *simulation* used to exercise the fail-closed orchestrator
  (`wllm/techniques/quant_sim.py`).
- **Auto CUDA kernels: deferred.** No kernel generation; the native engine's
  csrc fast-path build is a queued milestone, and its passes fail closed
  meanwhile.
- **Session migration: deferred.** Snapshot/restore/fork/migration is M18+/1.0;
  the IR already carries the `migratable`/`forkable` flags so plans cannot
  presume it.
- **Arbitrary-repo auto-IR: not claimed.** `wllm inspect` records `unknowns`
  instead of guessing; planning for an unrecognized model is diagnose-only.
- **Proof system first: done in order.** Receipts, forbidden-log invariants,
  drift-gated verdicts, and the verifier laws all landed (M1–M16) *before*
  the data-plane pillars (M17), matching the prescribed sequence.

One honest caveat: §3.6's four data-plane pillars are **CPU-verified
infrastructure**; their GPU-measured wiring (real model runners in the omni
engine, composite replay of the CFG-parallel plan with receipts) is M18+, and
the register above treats them accordingly.

---

## 5. Open gaps (not built — do not cite as coverage)

1. **Real quality oracles**: no LPIPS, temporal-warp/flicker, video-benchmark
   suites, VLM judge, or multi-seed protocol; wBench levels C–E are design.
2. **Closed-loop task gates**: no simulator success-rate gate wired in; the
   LIBERO rollout bridge is queued, not landed.
3. **CUDA-graph bucketing & compile telemetry**: no shape buckets, capture
   cache/eviction, graph-break or recompilation reporting, or safe-fallback
   capture machinery.
4. **Memory estimator + admission control + OOM recovery**: none implemented
   in a serving path.
5. **Interconnect topology awareness**: no NVLink/NUMA/PCIe hardware graph, no
   measured bandwidth model, no collective-hang watchdog.
6. **Session snapshot/restore/fork/migration**: no lease, generation numbers,
   idempotency keys, WAL, or GPU-resident state store; `SessionStore` is
   in-process CPU only.
7. **Sandboxed agent execution**: `.wllm/generated/` confinement is
   convention; no container/filesystem/network/syscall isolation, no
   independent verifier execution service, no signed receipts.
8. **Registry evidence lifecycle**: no version-range matching, per-entry
   validation timestamps, compatibility CI, or profile retirement; external
   backends have no capability files yet.
9. **Measurement environment lockdown**: receipts do not capture GPU clocks,
   power limits, or co-tenant processes; no statistical noise estimation.
10. **Reliability suite**: 24 h streaming soak, fault-injection cases (incl.
    the known IPC no-timeout deadlock), and worker-restart recovery are queued
    Beta gates, not shipped.
