"""Model #2: Qwen3-VL-8B-Instruct (MLLM / AR-decode archetype).

Baseline + optimized variants in one process.  Exact gate for AR greedy
decoding is TIE-AWARE token-id equality (verifier law 2: a token
disagreement can be arbitration rather than divergence).

The adjudication rule is NOT inline here — it lives in
``wllm.verify.adjudicate`` so every benchmark shares one audited,
unit-tested implementation.  It censuses every disagreeing position,
measures a bounded deterministic sample of them (first, last, and an
even spread between) on BOTH the teacher-forced prefill path and the
incremental KV-cache decode path, and aggregates conservatively: the
plan is exact only if every examined position has both disputed tokens
inside the epsilon-optimal set AND both measurement paths agree, the
disagreement is not widespread enough to be a trajectory fork, and the
two runs emitted the same number of tokens.  Anything else — a decisive
gap, paths that disagree, a degenerate row — is a refusal carrying the
measured gaps and the count of unexamined positions as evidence.

Levers: static KV cache, torch.compile on the language model.  Metrics:
TTFT-proxy (prefill+first token), full-generation wall, tokens/s.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Qwen3-VL-8B-Instruct"
NEW_TOKENS = 128
REPS = 3
BUDGET_WALL_S = 1500.0
_T0 = time.monotonic()


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from wllm.planner.plan import DeploymentPlan, Stage
    from wllm.planner.search import Measurement, successive_halving
    from wllm.verify.adjudicate import (
        EPSILON_DEFAULT, adjudicate_generation, first_divergence,
    )

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[load] ready t={time.monotonic() - _T0:.0f}s", flush=True)

    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (448, 448, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in detail, then list "
                                 "three plausible uses for it."}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")

    gen_kwargs = dict(max_new_tokens=NEW_TOKENS, do_sample=False)
    state = {"cache": None}

    def decode_once():
        torch.compiler.cudagraph_mark_step_begin()
        kw = dict(gen_kwargs)
        if state["cache"]:
            kw["cache_implementation"] = state["cache"]
        with torch.inference_mode():
            out = model.generate(**inputs, **kw)
        torch.cuda.synchronize()
        return out[0, inputs["input_ids"].shape[1]:].tolist()

    ref_tokens = {}
    TIE_GAP_EPS = EPSILON_DEFAULT   # bf16 logit band; the Alpha flip had 0.0

    def apply(plan_id: str):
        if plan_id == "ar_base":
            return
        if plan_id == "static_cache":
            state["cache"] = "static"
        elif plan_id == "static_cache_compile":
            state["cache"] = "static"
            model.model.language_model.compile(mode="reduce-overhead")

    def measure(plan: DeploymentPlan, duration_s: float) -> Measurement:
        if time.monotonic() - _T0 > BUDGET_WALL_S and plan.id != "ar_base":
            return Measurement(plan_id=plan.id, duration_s=0, ok=False,
                               error="wall budget exceeded")
        apply(plan.id)
        toks = decode_once()  # warmup (compile cost here)
        times = []
        for _ in range(REPS if duration_s > 30 else 1):
            t0 = time.monotonic()
            toks = decode_once()
            times.append((time.monotonic() - t0) * 1000.0)
        med = sorted(times)[len(times) // 2]

        verdict_blob = None
        if "ar_base" not in ref_tokens:
            ref_tokens["ar_base"] = toks
            exact, err = True, ""
        else:
            ref = ref_tokens["ar_base"]
            exact = toks == ref
            err = ""
            if not exact:
                div = first_divergence(ref, toks)
                if not div.adjudicable:
                    # a pure prefix/length difference has no disputed
                    # logit row to judge — EOS timing, refused as before
                    err = (f"length mismatch (EOS timing): "
                           f"{len(toks)} vs {len(ref)} tokens")
                else:
                    pos = div.position
                    try:
                        adj = adjudicate_generation(model, inputs, ref, toks,
                                                    epsilon=TIE_GAP_EPS)
                    except Exception as exc:                   # noqa: BLE001
                        adj = None
                        err = (f"greedy token mismatch from pos {pos}: "
                               f"ref={div.reference_token} "
                               f"got={div.candidate_token}; adjudication "
                               f"could not be measured "
                               f"({type(exc).__name__}: {exc}) — refused, "
                               f"no evidence is not a pass")
                        print(f"[tie-gate] {plan.id}: REFUSED — {err}",
                              flush=True)
                    if adj is not None:
                        verdict_blob = adj.as_dict()
                        for note in adj.notes:
                            print(f"[tie-gate] {plan.id}: note — {note}",
                                  flush=True)
                        if adj.is_benign:
                            exact = True
                            print(f"[tie-gate] {plan.id}: ACCEPTED "
                                  f"{adj.summary()}", flush=True)
                        else:
                            # verdict is real_divergence or undecidable;
                            # both are refusals, with different diagnoses
                            err = (f"greedy token mismatch from pos {pos} "
                                   f"({adj.positions_disagreeing} of "
                                   f"{adj.positions_compared} positions "
                                   f"disagree) -> {adj.verdict}; "
                                   f"{adj.reason}")
                            print(f"[tie-gate] {plan.id}: REFUSED "
                                  f"{adj.summary()}", flush=True)
        n_tok = len(toks)
        extra = {"times_ms": times, "n_tokens": n_tok,
                 "t_elapsed_s": time.monotonic() - _T0}
        if verdict_blob is not None:
            extra["adjudication"] = verdict_blob
        return Measurement(plan_id=plan.id, duration_s=duration_s, ok=exact,
                           latency_ms=med,
                           sustained_rate=n_tok / (med / 1000.0),
                           error=err, extra=extra)

    plans = [
        DeploymentPlan(id="ar_base", stages=[Stage(id="s", node_ids=["lm"],
                       device=0)], notes="dynamic cache eager"),
        DeploymentPlan(id="static_cache", stages=[Stage(id="s",
                       node_ids=["lm"], device=0)],
                       transforms=["cache:static"]),
        DeploymentPlan(id="static_cache_compile", stages=[Stage(id="s",
                       node_ids=["lm"], device=0)],
                       transforms=["cache:static", "compile:reduce-overhead"]),
    ]
    res = successive_halving(plans, measure, probe_s=10.0, growth=6.0,
                             keep_fraction=0.9, min_final=2)
    print(res.report(), flush=True)

    rows = [{"plan": r.plan.id, "culled_at": r.culled_at_round,
             "cull_reason": r.cull_reason,
             "final_latency_ms": r.last.latency_ms if r.last and r.last.ok else None,
             "rounds": [{"latency_ms": m.latency_ms, "ok": m.ok, **m.extra}
                        for m in r.rounds]} for r in res.records]
    base_ms = next(r["final_latency_ms"] for r in rows if r["plan"] == "ar_base")
    best = min((r for r in rows if r["final_latency_ms"]),
               key=lambda r: r["final_latency_ms"])
    summary = {"model": "qwen3-vl-8b", "new_tokens": NEW_TOKENS,
               "base_ms": base_ms, "best_plan": best["plan"],
               "best_ms": best["final_latency_ms"],
               "speedup": base_ms / best["final_latency_ms"],
               "target_met_1p3x": base_ms / best["final_latency_ms"] >= 1.3,
               "rows": rows}
    out = ROOT / "benchmarks/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"qwen3vl_opt_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=1), flush=True)
    print("QWEN3VL_OK" if summary["target_met_1p3x"]
          else "QWEN3VL_DONE_BELOW_TARGET", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
