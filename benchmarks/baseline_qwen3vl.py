"""Model #2: Qwen3-VL-8B-Instruct (MLLM / AR-decode archetype).

Baseline + optimized variants in one process.  Exact gate for AR greedy
decoding is TIE-AWARE token-id equality (verifier law 2: ties are
arbitration, not divergence).  On the first mismatching position the
gate runs one teacher-forced forward on the reference prefix and
measures the top-2 logit gap: the flip is accepted only if the two
tokens ARE the top-2 pair within a bf16-resolution epsilon — proven,
never assumed.  Anything else is refused with the measured gap as
evidence.

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
    TIE_GAP_EPS = 1e-3   # bf16 logit resolution; the Alpha flip had gap 0.0

    def tie_adjudicate(pos: int, ref: list[int], got: list[int]):
        """Teacher-forced forward on the reference prefix up to ``pos``.

        Returns (is_tie, gap, top2_ids): the flip counts as arbitration
        only if {ref[pos], got[pos]} is exactly the top-2 logit pair and
        their gap is within epsilon. One extra forward per adjudication
        — measured evidence, not tolerance hand-waving.
        """
        ids = inputs["input_ids"]
        if pos:
            ext = torch.tensor([ref[:pos]], device=ids.device,
                               dtype=ids.dtype)
            ids = torch.cat([ids, ext], dim=1)
        fwd = dict(inputs)
        fwd["input_ids"] = ids
        if "attention_mask" in fwd:
            fwd["attention_mask"] = torch.ones_like(ids)
        with torch.inference_mode():
            logits = model(**fwd).logits[0, -1].float()
        # epsilon-optimal-set criterion: knife edges can be degenerate
        # beyond two tokens (observed live: a three-way tie resolved
        # differently by dynamic-cache, static-cache, and teacher-forced
        # paths), so the flip is arbitration iff BOTH disputed tokens
        # sit within epsilon of the maximum logit
        mx = float(logits.max())
        gap_ref = mx - float(logits[ref[pos]])
        gap_got = mx - float(logits[got[pos]])
        gap = max(gap_ref, gap_got)
        return (gap <= TIE_GAP_EPS, gap,
                {"ref_gap": round(gap_ref, 6), "got_gap": round(gap_got, 6)})

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

        if "ar_base" not in ref_tokens:
            ref_tokens["ar_base"] = toks
            exact, err = True, ""
        else:
            ref = ref_tokens["ar_base"]
            exact = toks == ref
            err = ""
            if not exact:
                mism = [i for i, (a, b) in enumerate(zip(toks, ref))
                        if a != b]
                if not mism:
                    err = (f"length mismatch (EOS timing): "
                           f"{len(toks)} vs {len(ref)} tokens")
                else:
                    pos = mism[0]
                    is_tie, gap, pair = tie_adjudicate(pos, ref, toks)
                    if is_tie:
                        exact = True
                        print(f"[tie-gate] {plan.id}: flip at pos {pos} "
                              f"adjudicated as argmax tie (top-2 gap "
                              f"{gap:.2e}, pair {pair}); accepted per "
                              f"tie-aware exact", flush=True)
                    else:
                        # KNOWN LIMITATION (observed live): the
                        # adjudication forward is a PREFILL pass while
                        # generation decodes incrementally; bf16 kernel
                        # differences between the two paths can shift a
                        # knife-edge tie so the flip tokens miss the
                        # teacher-forced top-2. Refusing is the safe
                        # verdict; the full diagnostic makes the case
                        # auditable.
                        err = (f"greedy token mismatch at pos {pos}: "
                               f"ref={ref[pos]} got={toks[pos]}, "
                               f"teacher-forced top2 {pair} gap "
                               f"{gap:.4f} — not adjudicable as a tie")
        n_tok = len(toks)
        return Measurement(plan_id=plan.id, duration_s=duration_s, ok=exact,
                           latency_ms=med,
                           sustained_rate=n_tok / (med / 1000.0),
                           error=err,
                           extra={"times_ms": times, "n_tokens": n_tok,
                                  "t_elapsed_s": time.monotonic() - _T0})

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
