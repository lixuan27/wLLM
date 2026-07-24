"""Direction #8: exact single-GPU optimized plans for Wan2.2-TI2V-5B.

Runs a family of exact-transformation variants in ONE process (loads the
pipeline once), drives them through wllm.planner.search.successive_halving,
verifies the exact contract (fixed seed, identical config; frame drift vs
in-process baseline within a documented tolerance), and persists every
measurement.  Anchor: job 195301 baseline median 5615 ms; target <=4319 ms.

Tolerance note: torch.compile may reorder float reductions; under bf16 we
accept mean|dF| <= 1.5/255 and p99|dF| <= 6/255 on uint8 frames as
"exact family" (documented, not silent).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Wan2.2-TI2V-5B-Diffusers"
PROMPT = ("A red vintage car drives along a coastal road at sunset, "
          "waves crashing on the rocks, cinematic lighting.")
H, W, FRAMES, STEPS, SEED = 480, 832, 33, 20, 1234
ANCHOR_MS = 5615.0
BUDGET_WALL_S = 2700.0   # stay inside the 55-min job
_T0 = time.monotonic()


def elapsed() -> float:
    return time.monotonic() - _T0


def main() -> int:
    import numpy as np
    import torch
    from diffusers import WanPipeline

    from wllm.planner.plan import DeploymentPlan, Stage
    from wllm.planner.search import Measurement, successive_halving

    pipe = WanPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print(f"[load] ready at t={elapsed():.0f}s", flush=True)

    def gen():
        # CUDA-graph replay safety: each pipeline call is one "step"; without
        # this, reduce-overhead graphs overwrite live outputs of the previous
        # replay (RuntimeError observed in job 195356).
        torch.compiler.cudagraph_mark_step_begin()
        out = pipe(prompt=PROMPT, height=H, width=W, num_frames=FRAMES,
                   num_inference_steps=STEPS,
                   generator=torch.Generator("cuda").manual_seed(SEED))
        return np.array(out.frames[0], copy=True)  # detach from graph buffers

    def as_u8(frames):
        arr = np.asarray(frames)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        return arr

    state = {"applied": set()}

    def apply_transform(plan_id: str) -> None:
        if plan_id == "exact_base":
            return
        if plan_id in state["applied"]:
            return
        if plan_id == "compile_reduce_overhead":
            pipe.transformer.compile(mode="reduce-overhead")
        elif plan_id == "compile_max_autotune_no_cg":
            pipe.transformer.compile(mode="max-autotune-no-cudagraphs")
        state["applied"].add(plan_id)

    ref_frames = {}

    def measure(plan: DeploymentPlan, duration_s: float) -> Measurement:
        if plan.id != "exact_base" and elapsed() > BUDGET_WALL_S:
            return Measurement(plan_id=plan.id, duration_s=0, ok=False,
                               error=f"wall budget exceeded at t={elapsed():.0f}s")
        apply_transform(plan.id)
        # warmup (includes any compile cost — reported separately)
        t_w0 = time.monotonic()
        frames = as_u8(gen())
        warm_s = time.monotonic() - t_w0
        reps = 1 if duration_s <= 30 else 2
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            frames = as_u8(gen())
            torch.cuda.synchronize()
            times.append((time.monotonic() - t0) * 1000.0)
        med = sorted(times)[len(times) // 2]

        # exact-family contract vs in-process baseline frames
        drift_ok, mean_d, p99_d = True, 0.0, 0.0
        if "exact_base" in ref_frames:
            d = np.abs(frames.astype(np.int16)
                       - ref_frames["exact_base"].astype(np.int16))
            mean_d = float(d.mean())
            p99_d = float(np.percentile(d, 99))
            drift_ok = mean_d <= 1.5 and p99_d <= 6.0
        else:
            ref_frames["exact_base"] = frames

        return Measurement(
            plan_id=plan.id, duration_s=duration_s, ok=drift_ok,
            latency_ms=med, sustained_rate=FRAMES / (med / 1000.0),
            error="" if drift_ok else
            f"exact-contract drift mean={mean_d:.2f} p99={p99_d:.1f}",
            extra={"warmup_s": warm_s, "reps": reps, "times_ms": times,
                   "frame_mean_drift": mean_d, "frame_p99_drift": p99_d,
                   "t_elapsed_s": elapsed()})

    plans = [
        DeploymentPlan(id="exact_base",
                       stages=[Stage(id="all", node_ids=["pipe"], device=0)],
                       notes="in-process re-anchor of job-195301 baseline"),
        DeploymentPlan(id="compile_reduce_overhead",
                       stages=[Stage(id="all", node_ids=["pipe"], device=0)],
                       transforms=["compile:reduce-overhead"],
                       notes="cudagraph-backed compiled transformer"),
        DeploymentPlan(id="compile_max_autotune_no_cg",
                       stages=[Stage(id="all", node_ids=["pipe"], device=0)],
                       transforms=["compile:max-autotune-no-cudagraphs"],
                       notes="autotuned fusion without cudagraph output reuse"),
    ]

    # ordering note: exact_base MUST run first to set the reference frames;
    # searcher preserves list order within a round.
    res = successive_halving(plans, measure, probe_s=10.0, growth=6.0,
                             keep_fraction=0.7, min_final=2)
    print(res.report(), flush=True)

    rows = []
    for rec in res.records:
        last = rec.last
        rows.append({
            "plan": rec.plan.id, "culled_at": rec.culled_at_round,
            "cull_reason": rec.cull_reason,
            "rounds": [
                {"latency_ms": m.latency_ms, "ok": m.ok, **m.extra}
                for m in rec.rounds],
            "final_latency_ms": last.latency_ms if last and last.ok else None,
        })
    best_ms = min((r["final_latency_ms"] for r in rows
                   if r["final_latency_ms"]), default=None)
    speedup = (ANCHOR_MS / best_ms) if best_ms else 0.0
    summary = {"anchor_ms": ANCHOR_MS, "best_ms": best_ms,
               "speedup_vs_anchor": speedup,
               "target_met_1p3x": bool(best_ms and best_ms <= ANCHOR_MS / 1.3),
               "rows": rows,
               "config": {"h": H, "w": W, "frames": FRAMES, "steps": STEPS,
                          "seed": SEED}}
    out = ROOT / "benchmarks/results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"wan22_optimized_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=1), flush=True)
    print("OPTIMIZED_OK" if summary["target_met_1p3x"]
          else "OPTIMIZED_DONE_BELOW_TARGET", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
