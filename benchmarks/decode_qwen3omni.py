"""Model #6: Qwen3-Omni-30B-A3B thinker — AR decode optimization sweep.

The onboarding smoke put this MoE thinker-talker at Launchable tier with
an EMPTY optimization list (job 202109: 24.2 tok/s, 65 greedy tokens).
This benchmark asks the next question by measurement, not by assumption:
does any exact-quality pass actually speed up decode on this
30B-A3B **MoE** thinker, and does it stay exact?

Candidates, all applied to ONE loaded model (a 30B load costs ~2 min, so
loading per leg would be the dominant cost and would also stop the legs
from sharing a reference):

  ar_base            reference leg — checkpoint-declared bf16, default
                     attention backend, default expert kernel, dynamic
                     KV cache. The reference precision is the declared
                     precision; fp32 is NOT used as an oracle.
  static_kv_cache    the registry's `static_kv_cache` pass, routed to
                     the thinker as `thinker_cache_implementation`.
  experts_batched_mm MoE-specific kernel selection: the routed-expert
                     GEMM is switched from the default grouped kernel to
                     the batched-gather kernel. Same math, different
                     reduction order and memory traffic.
  experts_eager      MoE-specific kernel selection: the per-expert
                     python loop (touches only hit experts, but syncs on
                     a data-dependent `nonzero`).
  best_combo         the winning experts kernel (if any) + static cache.

Exactness gate (verifier law 2): greedy AR decode is compared by
TOKEN-SEQUENCE EQUALITY against the reference leg. A disagreement is
adjudicated, never tolerated -- and the adjudication is NOT
reimplemented here: it is delegated to `wllm.verify.adjudicate`, the
shared verifier that owns the epsilon-optimal-set rule (both disputed
tokens within eps=1e-3 of the maximum logit) plus the stronger
prefill/decode dual-path consistency rule. Only `benign_tie` counts as
exact; `real_divergence` and `undecidable` are both refusals.

Both generation lengths are FORCED (min_new_tokens == max_new_tokens),
so a leg can never look fast by emitting fewer tokens, and the 256-token
leg genuinely exercises a longer cache than the 64-token one.

Every leg is crash-isolated: a leg that raises produces an honest
`failed` row carrying its exception, and the sweep continues.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_PATH = os.environ.get("WLLM_Q3O_PATH", MODEL_ID)
LENGTHS = (64, 256)          # forced new-token counts, short and long
REPS = 3                     # timed reps per (leg, length); median wins
TIE_GAP_EPS = 1e-3           # bf16 logit resolution (same as model #2)
WIN_THRESHOLD = 1.05         # below this a "speedup" is noise, not a win
BUDGET_WALL_S = float(os.environ.get("WLLM_Q3O_BUDGET_S", "4200"))
_T0 = time.monotonic()

PROMPT = (
    "You are reviewing a deployment plan for a mixture-of-experts "
    "multimodal assistant that must answer streaming user requests. "
    "Explain, step by step and in full sentences, why the key/value "
    "cache layout dominates autoregressive decode cost, how expert "
    "routing changes the memory traffic per generated token, and what "
    "an engineer should measure before claiming an optimization is "
    "both faster and exact."
)


def _elapsed() -> float:
    return time.monotonic() - _T0


def main() -> int:  # noqa: C901 — one linear sweep, kept in one place
    import torch
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    from wllm.verify.adjudicate import (
        BENIGN_TIE, IDENTICAL, TOKEN_MISMATCH, adjudicate_generation,
        first_divergence,
    )

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    path_notes = []
    if hasattr(model, "disable_talker"):
        model.disable_talker()
        path_notes.append("talker disabled via disable_talker()")
    load_s = time.monotonic() - t0

    # The expert kernel and the attention backend both live on the
    # thinker's TEXT config -- the same object the routed-expert module
    # reads at every forward, which is what makes the switch verifiable.
    text_cfg = model.thinker.model.config
    base_experts = getattr(text_cfg, "_experts_implementation", None)
    base_attn = getattr(text_cfg, "_attn_implementation", None)
    print(f"[load] ready in {load_s:.0f}s; {'; '.join(path_notes) or 'no talker toggle'}; "
          f"experts={base_experts!r} attn={base_attn!r}", flush=True)

    messages = [{"role": "user",
                 "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f"[prompt] {prompt_len} tokens", flush=True)
    # The adjudicator replays this prompt through the thinker directly, so
    # it gets exactly the two tensors the text path consumes -- never the
    # generate-only wrapper keys.
    adj_inputs = {"input_ids": inputs["input_ids"],
                  "attention_mask": inputs["attention_mask"]}

    # ------------------------------------------------------------ decode
    def decode_once(n_new: int, cache: str | None):
        """One forced-length greedy generation. Returns (ids, cache_cls).

        Length is pinned on BOTH sides so no leg can win by stopping
        early, and `return_dict_in_generate` is on for every leg (same
        overhead everywhere) so the realized cache class is observable
        -- that class name is the authenticity signal for the static
        cache pass: a pass that silently did not engage must not be
        allowed to report a speedup.
        """
        kw = dict(inputs)
        kw.update(do_sample=False,
                  thinker_max_new_tokens=n_new,
                  thinker_min_new_tokens=n_new,
                  return_dict_in_generate=True)
        if cache:
            kw["thinker_cache_implementation"] = cache
        with torch.inference_mode():
            out = model.generate(**kw)
        torch.cuda.synchronize()
        seq = out.sequences if hasattr(out, "sequences") else out
        cache_cls = type(getattr(out, "past_key_values", None)).__name__
        return seq[0, prompt_len:].tolist(), cache_cls

    # -------------------------------------------------- exactness gate
    def check_exact(ref: list[int], got: list[int]) -> dict:
        """Token-sequence equality, with disagreements sent to the verifier.

        The gate itself decides nothing about numerics: it locates the
        first divergence and hands it to `wllm.verify.adjudicate`, which
        owns the epsilon-optimal-set rule and the prefill/decode
        consistency rule. Only `benign_tie` is exact. The verifier rules
        on the FIRST disagreement, so the count of disagreeing positions
        is reported alongside it rather than being implied away.
        """
        div = first_divergence(ref, got)
        if div.kind == IDENTICAL:
            return {"exact": True, "kind": "identical",
                    "detail": div.reason, "n_mismatch": 0}
        n_mismatch = sum(1 for a, b in zip(ref, got) if a != b)
        if div.kind != TOKEN_MISMATCH:
            # Forced lengths make this structurally impossible; if it
            # happens the leg is refused rather than explained away.
            return {"exact": False, "kind": div.kind, "n_mismatch": n_mismatch,
                    "detail": div.reason, "divergence": div.as_dict()}
        adj = adjudicate_generation(model.thinker, adj_inputs, ref, got,
                                    epsilon=TIE_GAP_EPS)
        return {"exact": adj.verdict == BENIGN_TIE, "kind": adj.verdict,
                "n_mismatch": n_mismatch, "dual_path": adj.dual_path,
                "detail": (f"{adj.reason} [{n_mismatch} of {len(ref)} "
                           f"positions disagree; the verifier rules on the "
                           f"first]"),
                "adjudication": adj.as_dict()}

    # -------------------------------------------------- leg application
    def apply_experts(name: str | None) -> str:
        """Switch the routed-expert kernel and PROVE it took effect."""
        want = base_experts if name is None else name
        model.set_experts_implementation(want)
        got = getattr(text_cfg, "_experts_implementation", None)
        if got != want:
            raise RuntimeError(
                f"experts kernel did not engage: asked {want!r}, config "
                f"reports {got!r} — refusing to time a pass that is not on")
        return got

    legs: list[dict] = [
        {"id": "ar_base", "experts": None, "cache": None,
         "notes": "reference leg: declared bf16, dynamic cache, "
                  "default expert kernel"},
        {"id": "static_kv_cache", "experts": None, "cache": "static",
         "notes": "registry pass static_kv_cache on the thinker"},
        {"id": "experts_batched_mm", "experts": "batched_mm", "cache": None,
         "notes": "routed-expert GEMM: batched gather kernel"},
        {"id": "experts_eager", "experts": "eager", "cache": None,
         "notes": "routed-expert GEMM: per-expert python loop"},
    ]

    rows: list[dict] = []
    ref_tokens: dict[int, list[int]] = {}
    tokens: dict[tuple[str, int], list[int]] = {}

    def run_leg(leg: dict) -> None:
        lid = leg["id"]
        try:
            engaged = apply_experts(leg["experts"])
        except Exception as exc:  # noqa: BLE001 — a failed leg is a result
            for n in LENGTHS:
                rows.append({"leg": lid, "new_tokens": n, "status": "failed",
                             "reason": f"{type(exc).__name__}: {exc}",
                             "notes": leg["notes"]})
            print(f"[leg {lid}] FAILED to apply: {exc}", flush=True)
            return
        try:
            for n in LENGTHS:
                if _elapsed() > BUDGET_WALL_S:
                    rows.append({"leg": lid, "new_tokens": n,
                                 "status": "failed", "notes": leg["notes"],
                                 "reason": "wall budget exceeded before this "
                                           f"cell ({_elapsed():.0f}s)"})
                    continue
                try:
                    torch.cuda.reset_peak_memory_stats()
                    _, cache_cls = decode_once(n, leg["cache"])  # warmup
                    times = []
                    for _ in range(REPS):
                        t = time.monotonic()
                        toks, cache_cls = decode_once(n, leg["cache"])
                        times.append((time.monotonic() - t) * 1e3)
                    med = sorted(times)[len(times) // 2]
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    tokens[(lid, n)] = toks
                    row = {"leg": lid, "new_tokens": n, "status": "measured",
                           "median_ms": round(med, 1),
                           "times_ms": [round(x, 1) for x in times],
                           "tok_per_s": round(len(toks) / (med / 1e3), 2),
                           "n_tokens_returned": len(toks),
                           "peak_vram_gb": round(peak, 1),
                           "experts_impl": engaged, "attn_impl": base_attn,
                           "cache_class": cache_cls, "notes": leg["notes"]}
                    rows.append(row)
                    print(f"[leg {lid} n={n}] {med:.0f} ms "
                          f"{row['tok_per_s']} tok/s cache={cache_cls} "
                          f"experts={engaged} peak={row['peak_vram_gb']}GB "
                          f"t={_elapsed():.0f}s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"leg": lid, "new_tokens": n,
                                 "status": "failed", "notes": leg["notes"],
                                 "reason": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-800:]})
                    print(f"[leg {lid} n={n}] FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                apply_experts(None)          # always restore the reference
            except Exception as exc:         # noqa: BLE001
                print(f"[leg {lid}] WARNING: restore failed: {exc}",
                      flush=True)

    for leg in legs:
        run_leg(leg)
        if leg["id"] == "ar_base":
            for n in LENGTHS:
                if (("ar_base", n)) in tokens:
                    ref_tokens[n] = tokens[("ar_base", n)]

    # A combined leg only earns GPU time if a kernel actually won; the
    # comparison uses the longest length, where cache management has the
    # most room to matter.
    long_n = LENGTHS[-1]
    base_long = next((r for r in rows if r["leg"] == "ar_base"
                      and r["new_tokens"] == long_n
                      and r["status"] == "measured"), None)
    if base_long:
        winners = [r for r in rows
                   if r["status"] == "measured" and r["new_tokens"] == long_n
                   and r["leg"].startswith("experts_")
                   and base_long["median_ms"] / r["median_ms"] >= WIN_THRESHOLD]
        if winners and _elapsed() < BUDGET_WALL_S:
            best = min(winners, key=lambda r: r["median_ms"])
            run_leg({"id": f"combo_{best['leg']}_static", "cache": "static",
                     "experts": best["experts_impl"],
                     "notes": f"{best['experts_impl']} expert kernel + "
                              "static KV cache"})
        else:
            print("[combo] skipped: no expert kernel beat the reference by "
                  f"{WIN_THRESHOLD}x at n={long_n}", flush=True)

    # ------------------------------------------- exactness adjudication
    # Adjudicate only AFTER the reference configuration is restored, so
    # the oracle forward runs under the declared-precision reference
    # numerics rather than under whichever candidate ran last.
    for row in rows:
        if row["status"] != "measured":
            continue
        n = row["new_tokens"]
        ref = ref_tokens.get(n)
        if ref is None:
            row.update(status="failed",
                       reason="no reference tokens: the reference leg did "
                              f"not produce a result at n={n}")
            continue
        if row["leg"] == "ar_base":
            row["exactness"] = {"exact": True, "kind": "reference",
                                "detail": "this leg IS the reference"}
            continue
        try:
            row["exactness"] = check_exact(ref, tokens[(row["leg"], n)])
        except Exception as exc:  # noqa: BLE001
            row["exactness"] = {
                "exact": False, "kind": "adjudication_failed",
                "n_mismatch": None,
                "detail": f"{type(exc).__name__}: {exc}"}
        print(f"[gate {row['leg']} n={n}] "
              f"{'EXACT' if row['exactness']['exact'] else 'REFUSED'} "
              f"({row['exactness']['kind']}) {row['exactness']['detail']}",
              flush=True)

    # ---------------------------------------------------------- verdicts
    base_ms = {r["new_tokens"]: r["median_ms"] for r in rows
               if r["leg"] == "ar_base" and r["status"] == "measured"}
    traces: list[dict] = []
    for row in rows:
        n = row["new_tokens"]
        workload = (f"thinker text decode, {n} forced new tokens, "
                    f"{prompt_len}-token prompt")
        cand = {"pass": row["leg"], "gpus": 1, "new_tokens": n}
        if row["status"] == "failed":
            traces.append({"workload": workload, "candidate": cand,
                           "status": "failed", "reason": row["reason"],
                           "metrics": {}})
            continue
        speedup = (base_ms[n] / row["median_ms"]) if n in base_ms else None
        row["speedup_vs_base"] = round(speedup, 4) if speedup else None
        metrics = {"median_ms": row["median_ms"],
                   "tok_per_s": row["tok_per_s"],
                   "peak_vram_gb": row["peak_vram_gb"],
                   "speedup_vs_base": row["speedup_vs_base"]}
        exact = row["exactness"]["exact"]
        if speedup is None and row["leg"] != "ar_base":
            # No reference number for this length: the candidate is
            # unjudgeable on speed, so it is refused, not guessed at.
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                f"no reference measurement at n={n} (the reference leg did "
                f"not produce one), so this leg's {row['median_ms']:.0f} ms "
                f"cannot be turned into a speedup claim; exactness verdict "
                f"was {row['exactness']['kind']}")
        elif row["leg"] == "ar_base":
            row["verdict"] = "accepted"
            row["verdict_reason"] = (
                "reference leg for this workload: declared-precision bf16, "
                "dynamic KV cache, default expert kernel; no optimization "
                "claimed, these are the numbers every candidate is "
                f"measured against ({row['tok_per_s']} tok/s)")
        elif not exact:
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                "exactness gate refused: " + row["exactness"]["detail"]
                + f" (measured {speedup:.3f}x, which is irrelevant while "
                  "the tokens differ beyond arbitration)")
        elif speedup is None or speedup < WIN_THRESHOLD:
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                f"exact ({row['exactness']['kind']}) but no win: measured "
                f"{speedup:.3f}x vs the reference leg "
                f"({row['median_ms']:.0f} ms vs {base_ms[n]:.0f} ms), below "
                f"the {WIN_THRESHOLD}x bar — not worth a profile claim")
        else:
            row["verdict"] = "accepted"
            row["verdict_reason"] = (
                f"exact ({row['exactness']['kind']}) and faster: "
                f"{speedup:.3f}x ({row['median_ms']:.0f} ms vs "
                f"{base_ms[n]:.0f} ms) at {n} forced new tokens; "
                f"authenticity: cache={row['cache_class']} "
                f"experts={row['experts_impl']}")
        traces.append({"workload": workload, "candidate": cand,
                       "status": row["verdict"],
                       "reason": row["verdict_reason"], "metrics": metrics})

    accepted_opt = [r for r in rows if r.get("verdict") == "accepted"
                    and r["leg"] != "ar_base"]
    summary = {
        "model": MODEL_ID, "task": "thinker-text-decode-sweep",
        "load_s": round(load_s, 1), "prompt_tokens": prompt_len,
        "lengths": list(LENGTHS), "reps": REPS,
        "win_threshold": WIN_THRESHOLD, "tie_gap_eps": TIE_GAP_EPS,
        "base_experts_impl": base_experts, "base_attn_impl": base_attn,
        "path_notes": path_notes,
        "reference_ok": bool(base_ms),
        "any_exact_win": bool(accepted_opt),
        "accepted_optimizations": sorted({r["leg"] for r in accepted_opt}),
        "wall_s": round(_elapsed(), 1),
        "rows": rows, "traces": traces,
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qwen3omni_decode_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))

    print("\n=== decode sweep ===", flush=True)
    for row in rows:
        if row["status"] == "failed":
            print(f"  {row['leg']:<28} n={row['new_tokens']:<4} FAILED  "
                  f"{row['reason'][:90]}", flush=True)
        else:
            print(f"  {row['leg']:<28} n={row['new_tokens']:<4} "
                  f"{row['median_ms']:>8.0f} ms  "
                  f"{row['tok_per_s']:>6.2f} tok/s  "
                  f"{row['speedup_vs_base']:>6}x  "
                  f"{'exact' if row['exactness']['exact'] else 'REFUSED':<8} "
                  f"{row['verdict']}", flush=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("rows", "traces")}, indent=1), flush=True)
    if not base_ms:
        # Without a reference leg nothing in the sweep is judgeable. The
        # summary is still on disk as evidence, but the run is a failure
        # and must not be reported as a clean "no win".
        print("Q3O_DECODE_SWEEP_FAIL: the reference leg produced no "
              "measurement at any length; every candidate here is "
              "unjudgeable", flush=True)
        return 1
    # A measured "no exact win available" is a real result, not a failure:
    # the sweep succeeded either way, and the marker says which happened.
    print("Q3O_DECODE_WIN" if summary["any_exact_win"]
          else "Q3O_DECODE_NO_WIN", flush=True)
    print("Q3O_DECODE_SWEEP_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        traceback.print_exc()
        print(f"Q3O_DECODE_SWEEP_FAIL: {type(exc).__name__}: {exc}",
              flush=True)
        raise SystemExit(1)
