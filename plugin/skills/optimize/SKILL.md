---
name: wllm-optimize
description: Optimize a project or model deployment through the wLLM control plane. Use when the user says "optimize this project deployment", "optimize this model deployment", "用 wLLM 优化", or asks for faster or cheaper serving of a world / multimodal model with quality guarantees.
---

# Optimize a deployment with wLLM

You are the bridge between one natural-language sentence and a typed,
fail-closed control plane. You translate intent; wLLM decides what is
true. Do not optimize the project by hand — call the infrastructure.

## 1. Translate the sentence into a typed spec

Turn the user's request into an OptimizeSpec (YAML) with:

- `objective`: primary metric (e.g. `p95_first_output`) plus secondaries
- `quality`: policy `exact` or `bounded`; `bounded` needs an explicit
  budget the user actually stated
- `hardware`: accelerator and count; `auto` when the user did not say

Anything you did not hear from the user or read from a manifest stays
`auto` or absent. Never invent a measurement, a GPU count, or a quality
budget. The optimizer reads only the typed spec, never your prose.

## 2. Call the tools in order

1. `wllm_inspect` — project facts into an evidence-listing manifest;
   anything undetected is UNKNOWN, not guessed. Read it before planning.
2. `wllm_plan` — pass the model id and your spec; read every rejected
   pass's stated reason instead of retrying blindly.
3. `wllm_verify` — only after a real measured receipt exists. Plans are
   not receipts; never verify what has not been measured.
4. `wllm_apply` — only with a receipt that passed verify.
5. `wllm_rollback` — on any regret (regression, bad apply, user doubt).
   The chain is optimized -> last-known-good -> reference; reference
   never rolls away. `wllm_report` shows current state at any time.

## 3. Honest outcomes

- Exit 3 means diagnose-only: nothing will be changed. That is a
  truthful outcome, not an error. Report the tool output verbatim and
  stop; do not "fix" it by weakening the spec or bypassing the plan.
- When verify or apply prints BLOCK reasons, relay them to the user
  unchanged. Do not paraphrase, soften, or summarize them away.

## 4. What you must not do

Never edit baselines, receipts, quality thresholds, or verifier code —
the tools refuse such edits anyway, so say so rather than trying. If
wLLM cannot promote a change with evidence, the honest answer is that
it was not promoted.
