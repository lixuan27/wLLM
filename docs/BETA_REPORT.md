# wLLM Beta Progress Report — Day 1 (2026-07-24)

Eleven milestones on `main`, every number from a real SLURM job with
on-disk evidence. This report is the honest support-tier ledger the
project promised: no blanket "supported".

## Milestone index

| # | Commit | What landed |
|---|---|---|
| 1 | `ed86e17` | wGraph v0, planner v0, L0/L1 integration, catalog importer (258/258), 28 tests |
| 2 | `a34e683` | successive-halving search; substrate L0 adapter; first real baseline (video 5B) |
| 3 | `6f41d77` | drift-gated optimization verdicts; negative results retained |
| 4 | `85e4307` | both in-house engines unified (2370 files, mechanical rename, gate-enforced) |
| 5 | `7f6505e` | VLM 2.75× with tie-aware exactness (argmax-tie proof) — **withdrawn 2026-07-27, see correction below** |
| 6 | `5159846` | Alpha complete: 3 models, median 2.75× — **one of the three is withdrawn** |
| 7 | `28688f7` | integration import health 87%; rename-damage zero |
| 8 | `dd87164` | bit-exact 1.74× denoise via CFG branch parallelism (2 GPU) |
| 9 | `84fe425` | first E2E app launch (Launchable tier, 704 frames) |
| 10 | `1fd44a1` | measured full-pipe **1.44× bit-exact** for the video model |
| 11 | `d9a97b5` | Hopper probe tier for the native engine |

## Model performance scorecard (1–2× H200)

| Model | Baseline | Best plan | Speedup | Verdict class |
|---|---|---|---|---|
| Wan2.2-TI2V-5B | 5762 ms E2E | CFG branch-parallel, 2 GPU | **1.44× E2E, frame-level bit-exact** | exact (measured, job 196293) |
| 〃 (single-GPU) | 5615 ms | compile max-autotune-no-cg | 1.43× | bounded-drift (refused by exact gate, documented) |
| Qwen3-VL-8B | 2668 ms /128 tok | static KV cache | ~~**2.75×**~~ | ~~tie-aware exact~~ **WITHDRAWN 2026-07-27 — see correction** |
| OpenVLA-7B | 133 ms /action (naive fp32) | native bf16 | **4.59×** | native-precision restoration (ckpt declares bf16) |

> **Correction, 2026-07-27 (jobs 202244, 202214).** The Qwen3-VL row is
> withdrawn. It was accepted under an adjudication rule that ruled on the
> teacher-forced **prefill** path alone; the current rule also replays the
> **decode** path and refuses when the two disagree. Re-verification could
> not re-establish the claim, because the adjudicator crashes on this
> model's multimodal path — so the status is **unproven, not disproven**,
> and this project's rule is that no evidence is not a pass.
>
> Two findings from the same week explain why this matters beyond one row.
> First, on a different model the strengthened rule caught a **4.32×** leg
> that the old prefill-only rule would have promoted as exact: prefill
> called it a benign tie, decode called it a real divergence. Second,
> requesting a static cache on this runtime **silently also enables a
> compiled forward**, so every leg labelled "static KV cache" has been
> measuring a composition — and the observed drift is exactly one bf16
> ULP, the signature of the fusion reordering that law 1 below already
> classifies as bounded. The experiment that separates cache from compile
> is running; until it lands, neither component is credited.
>
> The report body is left as originally written. A dated report edited to
> agree with later knowledge is no longer evidence of what was known when.

## Verifier laws (the research core, each with disk evidence)

1. **Compiled diffusion diverges by trajectory** — bf16 fusion reordering
   × 20 denoise steps ⇒ pixel drift (mean 7.4/255); "exact" must be
   classified empirically per (model, precision, loop depth).
2. **Ties are arbitration, not divergence** — the single greedy flip under
   static cache sits at a top-2 logit gap of exactly 0.0; cross-numerics
   token gates must be tie-aware.
3. **The reference is the checkpoint's declared precision** — an fp32
   upcast of bf16-stored weights is a variant, not an oracle.
4. **Batching changes results; distribution need not** — pipeline-native
   batched CFG differs from sequential branches by up to 251/255 per
   pixel (visibly different video), while one-branch-per-GPU is
   bit-identical (no cross-rank reductions). Quantified end-to-end.

## Support-tier ledger

| Component | Tier | Evidence / gap |
|---|---|---|
| longlive app | **Launchable** | job 196229: speech→VAD→ASR (verbatim)→prompt→DiT→VAE→704 frames→clean exit |
| worldplay app | Cataloged | needs HY-WorldPlay 42 GB (HF-proxy only) + converter (in-tree, dogfooded) |
| liveavatar app | Cataloged | needs Wan2.2-S2V-14B + LoRA + the external omni serving engine |
| krea_sam app | Cataloged | needs Krea 14B + SAM3 |
| qwen3_omni app | Cataloged | needs the external omni engine build |
| serving runtime | Import-healthy (87%) + Launchable-proven | portability family landed (SDPA backend, RMSNorm fallback, ASR sdpa ×5) |
| native engine | **Hopper probe tier** | SM90 dispatch + clean import chain on H200; remaining: lerobot processor-JSON stats in loader; csrc build only for FP8/FP4 fast paths |
| Matrix-Game-2.0 | Discovered | weights local (8.4 GB); needs official pipeline code + distilled-weight completion |
| LIBERO rollout validation | Designed | env matrix misaligned (sim in one env, 4.40-era loader in another); two-env bridge = third multi-env worker case |

## Infra lessons digest (from findings F1–F27)

- Mirror quality: a ModelScope mirror shipped without `config.json`,
  tokenizer files, and one weight shard; HF proxy fetches produced both a
  15-byte "Entry not found" stub and a mid-file-corrupted JSON —
  **content-level validation (json.load / tail checks) is mandatory** for
  every fetched file.
- `inference-mode discipline beats memory knobs`: an apparent VAE-decode
  OOM was autograd retaining ~140 GB of graph; tiling was a red herring.
- Triton/inductor caches must be node-local on NFS clusters (atomic-rename
  races); venv copies need `bin/` entrypoints (`python -m
  torch.distributed.run`); `/tmp` is node-local — fixtures live on shared FS.
- IPC: a no-timeout blocking send on an optional buffer with no consumer
  deadlocks the caller — now a planned fault-injection case for the
  verification suite.

## Next queue (priority order)

1. Loader support for lerobot processor-JSON stats (unlocks pi05 on the
   native Hopper path end-to-end).
2. LIBERO two-env bridge (action-server + sim-client) → close the VLA
   rollout-validation gap.
3. App weights acquisition (HY-WorldPlay first) → replicate the
   Launchable harness across the remaining four apps.
4. Native csrc Hopper build (FP8 cuBLASLt fast path).
5. Matrix-Game-2.0 official pipeline integration (interactive WM archetype).
6. 24 h streaming soak + fault-injection suite (Beta reliability gates).
