"""Qwen3-Omni thinker: separate the static KV cache from the auto-compile.

Job 202214 measured `cache_implementation="static"` at 4.32x on this MoE
thinker and the token-parity gate REFUSED it. Reading the installed
runtime afterwards showed why the leg was mis-named: `generate()` puts
the model on a `torch.compile`'d forward whenever the cache is
compileable and is not a `DynamicCache` (the auto-compile criterion),
and the default compile config is inductor with `mode="reduce-overhead"`
— i.e. CUDA graphs. So that leg measured a COMPOSITION, and neither its
speed nor its numeric drift could be attributed to the cache.

This benchmark separates them, in one process, against one reference:

  ar_base                   dynamic cache; the auto-compile criterion
                            explicitly excludes DynamicCache, so this leg
                            is uncompiled by construction.
  static_cache_nocompile    static cache with auto-compile SUPPRESSED.
                            This is the cache, alone, for the first time.
  static_cache_autocompile  static cache with auto-compile left on, i.e.
                            a same-process reproduction of the 202214 leg.

Then:
    cache_only        = ar_base / static_cache_nocompile
    compile_increment = static_cache_nocompile / static_cache_autocompile
    composition       = ar_base / static_cache_autocompile
and `cache_only * compile_increment == composition` by construction, so
the decomposition is arithmetically checkable rather than asserted.

SUPPRESSION IS A CLAIM, SO IT IS PROVEN, NOT TRUSTED. Passing a kwarg
named `disable_compile` proves nothing on its own. The no-compile leg is
only allowed to report a cache-only number if two independent signals
agree that no compilation happened:
  (A) the runtime never built a compiled callable — the attribute it
      caches on the model is still absent (checked before/after, and the
      legs are ordered so the suppressed leg runs BEFORE the compiled
      one, because that attribute is sticky once set);
  (B) the compiler's own counters did not move during the leg.
If the signals disagree, or if suppression simply does not work on this
build, the leg is recorded as a FAILURE with that as the finding — an
unsuppressed run is never quietly relabelled as "cache alone".

Exactness is delegated to `wllm.verify.adjudicate` unchanged: token
equality against the reference leg, disagreements settled by the
epsilon-optimal-set rule plus the prefill/decode consistency rule, and
only `benign_tie` counts as exact.

Generation lengths are forced (min_new_tokens == max_new_tokens) so no
leg can look fast by emitting fewer tokens.
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
LENGTHS = (64, 256)          # identical to job 202214, so numbers compare
REPS = 3
TIE_GAP_EPS = 1e-3
WIN_THRESHOLD = 1.05
BUDGET_WALL_S = float(os.environ.get("WLLM_Q3O_BUDGET_S", "3000"))
_T0 = time.monotonic()

# Byte-identical to the job 202214 prompt: reusing it is what makes the
# reference leg here comparable with the reference leg there.
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


def main() -> int:  # noqa: C901 — one linear experiment, kept in one place
    import torch
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    from wllm.verify.adjudicate import (
        BENIGN_TIE, IDENTICAL, TOKEN_MISMATCH, adjudicate_generation,
        first_divergence,
    )

    # Signal (B). Imported defensively: if this build does not expose the
    # compiler counters, the signal is UNAVAILABLE (which weakens the
    # suppression proof) rather than an exception that kills the run.
    try:
        from torch._dynamo.utils import counters as dynamo_counters
    except Exception as exc:  # noqa: BLE001
        dynamo_counters = None
        print(f"[warn] compiler counters unavailable ({exc}); signal B "
              f"cannot be collected", flush=True)

    def compile_stats() -> dict:
        if dynamo_counters is None:
            return {}
        return dict(dynamo_counters["stats"])

    def stats_delta(before: dict, after: dict) -> dict:
        keys = set(before) | set(after)
        return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(keys)
                if after.get(k, 0) != before.get(k, 0)}

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
    thinker = model.thinker
    text_cfg = thinker.model.config
    print(f"[load] ready in {load_s:.0f}s; "
          f"{'; '.join(path_notes) or 'no talker toggle'}; "
          f"experts={getattr(text_cfg, '_experts_implementation', None)!r} "
          f"attn={getattr(text_cfg, '_attn_implementation', None)!r}",
          flush=True)

    messages = [{"role": "user",
                 "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f"[prompt] {prompt_len} tokens", flush=True)
    adj_inputs = {"input_ids": inputs["input_ids"],
                  "attention_mask": inputs["attention_mask"]}

    # ------------------------------------------------------------ decode
    def decode_once(n_new: int, cache: str | None, no_compile: bool):
        # reduce-overhead compilation manages output buffers with CUDA
        # graphs, so a previous run's output can be overwritten by the
        # next one; mark the step and materialise the ids immediately.
        torch.compiler.cudagraph_mark_step_begin()
        kw = dict(inputs)
        kw.update(do_sample=False,
                  thinker_max_new_tokens=n_new,
                  thinker_min_new_tokens=n_new,
                  return_dict_in_generate=True)
        if cache:
            kw["thinker_cache_implementation"] = cache
        if no_compile:
            kw["thinker_disable_compile"] = True
        with torch.inference_mode():
            out = model.generate(**kw)
        torch.cuda.synchronize()
        seq = out.sequences if hasattr(out, "sequences") else out
        ids = seq[0, prompt_len:].tolist()
        return ids, type(getattr(out, "past_key_values", None)).__name__

    # -------------------------------------------------- exactness gate
    def check_exact(ref: list[int], got: list[int]) -> dict:
        div = first_divergence(ref, got)
        if div.kind == IDENTICAL:
            return {"exact": True, "kind": "identical",
                    "detail": div.reason, "n_mismatch": 0}
        n_mismatch = sum(1 for a, b in zip(ref, got) if a != b)
        if div.kind != TOKEN_MISMATCH:
            return {"exact": False, "kind": div.kind, "n_mismatch": n_mismatch,
                    "detail": div.reason, "divergence": div.as_dict()}
        adj = adjudicate_generation(thinker, adj_inputs, ref, got,
                                    epsilon=TIE_GAP_EPS)
        return {"exact": adj.verdict == BENIGN_TIE, "kind": adj.verdict,
                "n_mismatch": n_mismatch, "dual_path": adj.dual_path,
                "detail": (f"{adj.reason} [{n_mismatch} of {len(ref)} "
                           f"positions disagree; the verifier rules on "
                           f"the first]"),
                "adjudication": adj.as_dict()}

    # Ordered deliberately: the compiled-callable attribute is sticky, so
    # the suppressed leg must be measured before the compiled one for
    # signal (A) to mean anything.
    legs = [
        {"id": "ar_base", "cache": None, "no_compile": False,
         "expect_compiled": False,
         "notes": "reference: dynamic cache; the auto-compile criterion "
                  "excludes DynamicCache, so this leg is uncompiled"},
        {"id": "static_cache_nocompile", "cache": "static",
         "no_compile": True, "expect_compiled": False,
         "notes": "static KV cache with auto-compile suppressed: the "
                  "cache on its own"},
        {"id": "static_cache_autocompile", "cache": "static",
         "no_compile": False, "expect_compiled": True,
         "notes": "static KV cache with auto-compile left on: same-process "
                  "reproduction of the job 202214 composition"},
    ]

    rows: list[dict] = []
    ref_tokens: dict[int, list[int]] = {}
    tokens: dict[tuple[str, int], list[int]] = {}

    for leg in legs:
        lid = leg["id"]
        for n in LENGTHS:
            if _elapsed() > BUDGET_WALL_S:
                rows.append({"leg": lid, "new_tokens": n, "status": "failed",
                             "notes": leg["notes"],
                             "reason": "wall budget exceeded before this "
                                       f"cell ({_elapsed():.0f}s)"})
                continue
            try:
                had_compiled = hasattr(thinker, "_compiled_call")
                before = compile_stats()
                torch.cuda.reset_peak_memory_stats()
                _, cache_cls = decode_once(n, leg["cache"], leg["no_compile"])
                after_warm = compile_stats()
                # Compilation happens on the warmup call, so signal (B) is
                # read across the warmup, not across the timed reps.
                delta = stats_delta(before, after_warm)
                has_compiled = hasattr(thinker, "_compiled_call")

                times = []
                for _ in range(REPS):
                    t = time.monotonic()
                    toks, cache_cls = decode_once(
                        n, leg["cache"], leg["no_compile"])
                    times.append((time.monotonic() - t) * 1e3)
                med = sorted(times)[len(times) // 2]
                peak = torch.cuda.max_memory_allocated() / 2**30
                tokens[(lid, n)] = toks

                # ---- suppression adjudication (fail closed) ----
                compiled_now = has_compiled and not had_compiled
                counters_moved = bool(delta)
                signals = {
                    "compiled_call_attr_before": had_compiled,
                    "compiled_call_attr_after": has_compiled,
                    "compiled_callable_built_here": compiled_now,
                    "compiler_counter_delta": delta,
                    "counters_available": dynamo_counters is not None,
                }
                observed_compiled = compiled_now or counters_moved
                row = {"leg": lid, "new_tokens": n, "status": "measured",
                       "median_ms": round(med, 1),
                       "times_ms": [round(x, 1) for x in times],
                       "tok_per_s": round(len(toks) / (med / 1e3), 2),
                       "n_tokens_returned": len(toks),
                       "peak_vram_gb": round(peak, 1),
                       "cache_class": cache_cls,
                       "compile_signals": signals,
                       "observed_compiled": observed_compiled,
                       "expect_compiled": leg["expect_compiled"],
                       "notes": leg["notes"]}

                if leg["no_compile"] and observed_compiled:
                    # THE finding, if it happens: suppression did not work.
                    row["status"] = "failed"
                    row["reason"] = (
                        "auto-compile could NOT be suppressed on this "
                        "build: asked for suppression but observed "
                        f"compilation (compiled callable built here="
                        f"{compiled_now}, compiler counter delta={delta}). "
                        "No cache-only number is reported, because this "
                        "leg did not isolate the cache")
                elif leg["expect_compiled"] and not observed_compiled:
                    row["status"] = "failed"
                    row["reason"] = (
                        "expected the auto-compiled path but observed no "
                        f"compilation (counter delta={delta}, counters "
                        f"available={dynamo_counters is not None}); this "
                        "leg cannot be labelled as the compiled "
                        "composition")
                rows.append(row)
                print(f"[leg {lid} n={n}] "
                      f"{'FAILED ' if row['status'] == 'failed' else ''}"
                      f"{med:.0f} ms {row['tok_per_s']} tok/s "
                      f"cache={cache_cls} compiled={observed_compiled} "
                      f"delta={delta} peak={row['peak_vram_gb']}GB "
                      f"t={_elapsed():.0f}s", flush=True)
                if row["status"] == "failed":
                    print(f"    reason: {row['reason']}", flush=True)
            except Exception as exc:  # noqa: BLE001 — a failed leg is a result
                rows.append({"leg": lid, "new_tokens": n, "status": "failed",
                             "notes": leg["notes"],
                             "reason": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()[-800:]})
                print(f"[leg {lid} n={n}] FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        if lid == "ar_base":
            for n in LENGTHS:
                if ("ar_base", n) in tokens:
                    ref_tokens[n] = tokens[("ar_base", n)]

    # ------------------------------------------- exactness adjudication
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
            row["exactness"] = {"exact": False, "n_mismatch": None,
                                "kind": "adjudication_failed",
                                "detail": f"{type(exc).__name__}: {exc}"}
        print(f"[gate {row['leg']} n={n}] "
              f"{'EXACT' if row['exactness']['exact'] else 'REFUSED'} "
              f"({row['exactness']['kind']}) {row['exactness']['detail']}",
              flush=True)

    # ------------------------------------------------------- attribution
    def ms(leg: str, n: int):
        for r in rows:
            if (r["leg"] == leg and r["new_tokens"] == n
                    and r["status"] == "measured"):
                return r["median_ms"]
        return None

    attribution = {}
    for n in LENGTHS:
        base, nocomp, auto = (ms("ar_base", n), ms("static_cache_nocompile", n),
                              ms("static_cache_autocompile", n))
        entry = {"base_ms": base, "nocompile_ms": nocomp, "autocompile_ms": auto,
                 "cache_only": None, "compile_increment": None,
                 "composition": None}
        if base and nocomp:
            entry["cache_only"] = round(base / nocomp, 4)
        if nocomp and auto:
            entry["compile_increment"] = round(nocomp / auto, 4)
        if base and auto:
            entry["composition"] = round(base / auto, 4)
        if entry["cache_only"] and entry["compile_increment"]:
            # the decomposition must multiply back out; if it does not,
            # something drifted between legs and the split is not trusted
            product = entry["cache_only"] * entry["compile_increment"]
            entry["decomposition_product"] = round(product, 4)
            entry["decomposition_consistent"] = (
                entry["composition"] is not None
                and abs(product - entry["composition"]) <= 0.02 * product)
        attribution[str(n)] = entry

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
        metrics = {"median_ms": row["median_ms"], "tok_per_s": row["tok_per_s"],
                   "peak_vram_gb": row["peak_vram_gb"],
                   "speedup_vs_base": row["speedup_vs_base"],
                   "observed_compiled": row["observed_compiled"]}
        exact = row["exactness"]["exact"]
        if row["leg"] == "ar_base":
            row["verdict"] = "accepted"
            row["verdict_reason"] = (
                "reference leg: declared-precision bf16, dynamic KV cache, "
                "uncompiled by construction; no optimization claimed "
                f"({row['tok_per_s']} tok/s)")
        elif speedup is None:
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                f"no reference measurement at n={n}; this leg's "
                f"{row['median_ms']:.0f} ms cannot become a speedup claim")
        elif not exact:
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                "exactness gate refused: " + row["exactness"]["detail"]
                + f" (measured {speedup:.3f}x, irrelevant while the tokens "
                  "differ beyond arbitration); compilation observed on this "
                  f"leg: {row['observed_compiled']}")
        elif speedup < WIN_THRESHOLD:
            row["verdict"] = "rejected"
            row["verdict_reason"] = (
                f"exact ({row['exactness']['kind']}) but no win: measured "
                f"{speedup:.3f}x ({row['median_ms']:.0f} ms vs "
                f"{base_ms[n]:.0f} ms), below the {WIN_THRESHOLD}x bar; "
                f"compilation observed: {row['observed_compiled']}")
        else:
            row["verdict"] = "accepted"
            row["verdict_reason"] = (
                f"exact ({row['exactness']['kind']}) and faster: "
                f"{speedup:.3f}x ({row['median_ms']:.0f} ms vs "
                f"{base_ms[n]:.0f} ms) at {n} forced new tokens; "
                f"authenticity: cache={row['cache_class']}, compilation "
                f"observed={row['observed_compiled']} "
                f"(signals {row['compile_signals']})")
        traces.append({"workload": workload, "candidate": cand,
                       "status": row["verdict"], "reason": row["verdict_reason"],
                       "metrics": metrics})

    suppression_worked = any(
        r["leg"] == "static_cache_nocompile" and r["status"] == "measured"
        and not r["observed_compiled"] for r in rows)
    accepted_opt = [r for r in rows if r.get("verdict") == "accepted"
                    and r["leg"] != "ar_base"]
    summary = {
        "model": MODEL_ID, "task": "static-cache-vs-autocompile-split",
        "load_s": round(load_s, 1), "prompt_tokens": prompt_len,
        "lengths": list(LENGTHS), "reps": REPS,
        "win_threshold": WIN_THRESHOLD, "tie_gap_eps": TIE_GAP_EPS,
        "path_notes": path_notes,
        "reference_ok": bool(base_ms),
        "suppression_worked": suppression_worked,
        "attribution": attribution,
        "any_exact_win": bool(accepted_opt),
        "accepted_optimizations": sorted({r["leg"] for r in accepted_opt}),
        "wall_s": round(_elapsed(), 1),
        "rows": rows, "traces": traces,
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qwen3omni_cachesplit_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))

    print("\n=== cache vs auto-compile split ===", flush=True)
    for row in rows:
        if row["status"] == "failed":
            print(f"  {row['leg']:<26} n={row['new_tokens']:<4} FAILED  "
                  f"{row['reason'][:100]}", flush=True)
        else:
            print(f"  {row['leg']:<26} n={row['new_tokens']:<4} "
                  f"{row['median_ms']:>8.0f} ms  {row['tok_per_s']:>6.2f} tok/s"
                  f"  {row['speedup_vs_base']:>7}x  compiled="
                  f"{str(row['observed_compiled']):<5} "
                  f"{'exact' if row['exactness']['exact'] else 'REFUSED':<8} "
                  f"{row['verdict']}", flush=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("rows", "traces")}, indent=1), flush=True)
    if not base_ms:
        print("Q3O_SPLIT_FAIL: the reference leg produced no measurement; "
              "nothing here is judgeable", flush=True)
        return 1
    print("Q3O_SPLIT_SUPPRESSED" if suppression_worked
          else "Q3O_SPLIT_UNSUPPRESSED", flush=True)
    print("Q3O_SPLIT_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        traceback.print_exc()
        print(f"Q3O_SPLIT_FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
