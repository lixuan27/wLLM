# Agent workflow: constrained roles, deterministic core

wLLM development and operation both run as a multi-agent workflow. The
design goal is that **no agent's claim is ever the source of truth**:
agents propose and implement inside typed contracts; the deterministic
core (CLI, gates, CI) executes, measures, and judges. This is the same
boundary the product enforces at runtime, applied to building it.

## Role contracts

Each role gets: typed inputs, an allowed directory set, explicit
success criteria, and machine-readable failure. Roles never grade
their own work.

| Role | Allowed to touch | Success criteria | Never allowed |
|---|---|---|---|
| Coordinator | integration points, docs, CI config | milestone lands with CI PASS + gate clean | bypassing a gate; editing others' in-flight files |
| Implementation agent | one module dir + its tests, `__init__` exports | files compile; tests written (executed by CI, not the agent) | running tests on the login node; touching reference oracles, verifier, thresholds; naming-rule violations |
| Adversarial reviewer | read-only | findings with file:line + concrete failure scenario, CONFIRMED/PLAUSIBLE marked | editing code; grading style instead of correctness |
| Risk/spec writer | one docs file | every claim carries a file pointer or is listed as a gap | claiming coverage without evidence |
| Evidence engineer | control-plane evidence tooling | receipts built only from real logs/benchmarks | fabricating or interpolating measurements |

## Interaction protocol

1. Coordinator scopes non-overlapping directory assignments; two
   agents never write the same file.
2. Implementation agents syntax-check only (`py_compile` loop form);
   the CPU CI job is the sole judge of test outcomes.
3. Every adversarial finding that survives triage lands as a pinned
   regression test before the fix is considered done.
4. The release gate (naming/secrets, staged + full-tree) and the CI
   battery (tests, BDD, coverage >= 85, mutation kill >= 80) run before
   every push; a BLOCKED verdict stops the milestone, not the reverse.

## Why constrained workflows, not stronger prompts

The infrastructure captures the expertise — profiles, IR, transform
templates, validators, structured failure reasons — so an agent's job
reduces to: pick a legal action, implement a local change, repair
against explicit failure output. This is also measurable: candidate
implementation success rate, failed-experiment count, GPU-hours spent,
and human interventions per milestone are tracked per agent tier. If a
smaller open-weight agent approaches a frontier model inside these
contracts, the system has genuinely solidified reusable knowledge.

## Runtime mirror

The same split ships to users: the MCP server (`wllm-mcp`) gives any
coding agent the six control-plane tools, while baselines, benchmark
fixtures, verifiers, quality thresholds, and historical receipts stay
read-only to it. An agent can propose; only evidence promotes.
