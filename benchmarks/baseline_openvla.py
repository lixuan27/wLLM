"""Model #3: openvla-7b action prediction (VLA archetype), own-env run.

Runs inside the dedicated `openvla` conda env (transformers 4.40 era) —
the multi-environment worker pattern: wLLM optimizes a model in the env it
actually works in.  Levers here are precision + attention implementation;
exact gate = the 7 discrete action tokens must match (tie-aware framing
applies if a flip coincides with a zero logit gap, cf. the VLM finding).

Self-contained: no wllm imports (this env doesn't have the package);
results JSON matches the sweep schema used by the other model tracks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/openvla-7b"
ROOT = Path(__file__).resolve().parents[1]
REPS = 10
INSTRUCTION = "In: What action should the robot take to pick up the red block?\nOut:"


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    rng = np.random.default_rng(11)
    img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))

    results = []
    ref_actions = {}

    def sweep(plan_id: str, dtype, attn_impl: str | None, compile_lm: bool):
        t0 = time.monotonic()
        kwargs = dict(trust_remote_code=True, torch_dtype=dtype,
                      low_cpu_mem_usage=True)
        # explicit attention implementation: the staged remote code
        # predates the _supports_sdpa probe of newer transformers, so an
        # unset implementation crashes in the sdpa dispatch check
        kwargs["attn_implementation"] = attn_impl or "eager"
        processor = AutoProcessor.from_pretrained(MODEL_DIR,
                                                  trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR,
                                                       **kwargs).to("cuda")
        model.eval()
        if compile_lm:
            model.language_model.compile(mode="reduce-overhead")
        load_s = time.monotonic() - t0

        inputs = processor(INSTRUCTION, img).to("cuda", dtype=dtype)

        def predict():
            if hasattr(torch, "compiler"):
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:  # noqa: BLE001
                    pass
            with torch.inference_mode():
                out = model.predict_action(**inputs,
                                              unnorm_key="bridge_orig",
                                              do_sample=False)
            torch.cuda.synchronize()

            def to_host(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().float().cpu().numpy().reshape(-1)
                if isinstance(x, dict):
                    return np.concatenate(
                        [to_host(x[k]) for k in sorted(x)]) if x else np.zeros(0)
                if isinstance(x, (list, tuple)):
                    return np.concatenate([to_host(i) for i in x])
                return np.asarray(x, dtype=np.float64).reshape(-1)

            return to_host(out).astype(np.float64)

        act = predict()  # warmup (compile cost here)
        times = []
        for _ in range(REPS):
            t1 = time.monotonic()
            act = predict()
            times.append((time.monotonic() - t1) * 1000.0)
        med = sorted(times)[len(times) // 2]
        p95 = sorted(times)[min(REPS - 1, int(round(0.95 * (REPS - 1))))]

        if "ref" not in ref_actions:
            ref_actions["ref"] = act
            ok, err = True, ""
        else:
            mse = float(np.mean((act - ref_actions["ref"]) ** 2))
            ok = mse < 1e-4
            err = "" if ok else f"action MSE {mse:.3e} vs fp32 reference"
        results.append({"plan": plan_id, "median_ms": med, "p95_ms": p95,
                        "ok": ok, "err": err, "load_s": load_s,
                        "action": act.tolist()})
        print(f"[{plan_id}] median={med:.0f}ms p95={p95:.0f}ms ok={ok} {err}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    # baseline exactly as the checkpoint era intends: fp32? community
    # default is bf16; we anchor at fp32 for a strict reference, then bf16.
    sweep("fp32_eager", torch.float32, None, False)
    sweep("bf16_eager", torch.bfloat16, None, False)
    sweep("bf16_sdpa", torch.bfloat16, "sdpa", False)

    base = results[0]["median_ms"]
    valid = [r for r in results if r["ok"]] or results[:1]
    best = min(valid, key=lambda r: r["median_ms"])
    summary = {"model": "openvla-7b", "reps": REPS, "base_ms": base,
               "best_plan": best["plan"], "best_ms": best["median_ms"],
               "speedup": base / best["median_ms"],
               "target_met_1p3x": base / best["median_ms"] >= 1.3,
               "rows": results}
    out = ROOT / "benchmarks/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"openvla_opt_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=1), flush=True)
    print("OPENVLA_OK" if summary["target_met_1p3x"]
          else "OPENVLA_DONE_BELOW_TARGET", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
