"""Reuse-cache technique on a real diffusion loop (Wan2.2-TI2V-5B).

Until now the reuse-cache technique executor was only exercised over
synthetic vectors.  This benchmark runs it against the actual denoise
loop of a 5B video model on one GPU: the reference leg and every cached
leg execute the *same* vendored loop, differing only in the technique's
threshold, so the comparison isolates the technique.

Design constraint discovered by inspection, not assumed: this
checkpoint's scheduler is a multistep solver (order 2, with a history of
previous model outputs).  Skipping its update would desynchronize that
history, so the cache is applied at the model-evaluation level — the
CFG-combined prediction — while the solver still steps every iteration.

Quality is measured against the reference leg's own frames at pixel
level; nothing here assumes the technique is lossless.  A leg that never
engaged the cache (steps_reused == 0) is rejected regardless of its
timing, because its speed cannot be attributed to this technique.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.techniques.step_cache_tensor import TensorOutputReuseCache

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Wan2.2-TI2V-5B-Diffusers"
PROMPT = ("A red vintage car drives along a coastal road at sunset, "
          "waves crashing on the rocks, cinematic lighting.")
NEG = "low quality, blurry, distorted"
H, W, FRAMES, SEED, CFG = 480, 832, 33, 1234, 5.0
# Schedule length is the variable that decides whether this technique can
# work at all: consecutive model evaluations are only redundant when the
# schedule is fine enough that neighbouring steps ask nearly the same
# question. Rounds 1-2 measured 20 steps and found no operating point;
# the boundary is a schedule-length question, so the length is a knob.
STEPS = int(os.environ.get("WLLM_STEPCACHE_STEPS", "20"))
THRESHOLDS = [float(x) for x in
              os.environ.get("WLLM_STEPCACHE_TAUS", "0.05,0.10,0.20").split(",")]
# (key, threshold, consecutive cap) triples.  Round 1 (job 202206) swept
# the input key alone and every leg was refused on quality: keying on
# latent movement engages early, where the latent is still near-noise
# (so its *relative* move is small) but the velocity field is least
# stable and sets the video's global structure.  Round 2 adds the
# output-stability key, which cannot engage until the function itself
# has been seen to settle, plus a tighter consecutive cap to bound drift.
LEGS = [("input", tau, 4) for tau in THRESHOLDS] + \
       [("output", tau, cap)
        for tau, cap in ((0.02, 2), (0.05, 2), (0.10, 2), (0.05, 1))]
REPS = 2
# declared bounded-quality budget for this technique on this workload;
# a leg outside it is reported as rejected, not quietly accepted
PSNR_BUDGET_DB = 30.0


def main() -> int:
    import torch
    from diffusers import WanPipeline

    device = "cuda:0"
    pipe = WanPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to(device)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()   # bounds the conv3d decode peak
    tr, sch = pipe.transformer, pipe.scheduler
    print(f"[load] scheduler={type(sch).__name__} "
          f"order={getattr(sch.config, 'solver_order', 'n/a')}", flush=True)

    with torch.inference_mode():
        pe, ne = pipe.encode_prompt(
            prompt=PROMPT, negative_prompt=NEG,
            do_classifier_free_guidance=True, device=device)

    z_dim = tr.config.out_channels
    shape = (1, z_dim, (FRAMES - 1) // 4 + 1, H // 16, W // 16)

    def predict(lat: torch.Tensor, k: int) -> torch.Tensor:
        """The expensive evaluation: batched CFG transformer + combine."""
        t = sch.timesteps[k]
        with torch.inference_mode():
            both = torch.cat([lat, lat]).to(torch.bfloat16)
            out = tr(hidden_states=both, timestep=t.expand(2),
                     encoder_hidden_states=torch.cat([pe, ne]),
                     return_dict=False)[0]
            cond, unc = out.chunk(2)
        return unc.float() + CFG * (cond.float() - unc.float())

    def denoise(cache: TensorOutputReuseCache) -> torch.Tensor:
        gen = torch.Generator(device).manual_seed(SEED)
        lat = torch.randn(shape, generator=gen, device=device,
                          dtype=torch.float32)
        sch.set_timesteps(STEPS, device=device)   # also resets solver state
        cache.reset()
        for k, t in enumerate(sch.timesteps):
            pred = cache(lat, k)
            lat = sch.step(pred, t, lat, return_dict=False)[0]
        return lat

    @torch.inference_mode()
    def decode(lat: torch.Tensor) -> torch.Tensor:
        vae = pipe.vae
        mean = (torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1).to(device))
        inv_std = (1.0 / torch.tensor(vae.config.latents_std)
                   .view(1, vae.config.z_dim, 1, 1, 1)).to(device)
        z = lat.to(torch.bfloat16) / inv_std.to(torch.bfloat16) \
            + mean.to(torch.bfloat16)
        video = vae.decode(z, return_dict=False)[0]
        return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)

    def timed(threshold: float, key: str = "input", cap: int = 4):
        cache = TensorOutputReuseCache(step_fn=predict, threshold=threshold,
                                        key=key, max_consecutive_reuses=cap)
        lat = denoise(cache)                     # warmup (also fills caches)
        torch.cuda.synchronize()
        times = []
        for _ in range(REPS):
            t0 = time.monotonic()
            lat = denoise(cache)
            torch.cuda.synchronize()
            times.append((time.monotonic() - t0) * 1e3)
        return lat, sorted(times)[len(times) // 2], times, cache

    legs = []
    ref_lat, ref_ms, ref_times, ref_cache = timed(0.0)
    ref_frames = decode(ref_lat)
    print(f"[ref] {ref_ms:.1f}ms; observed per-step relative deltas: "
          + ", ".join(f"{d:.3f}" for d in ref_cache.deltas), flush=True)
    legs.append({"leg": "reference", "threshold": 0.0, "median_ms": ref_ms,
                 "times_ms": ref_times, "steps_total": ref_cache.steps_total,
                 "steps_reused": ref_cache.steps_reused, "speedup": 1.0,
                 "status": "reference",
                 "observed_deltas": [round(d, 5) for d in ref_cache.deltas]})

    for key, tau, cap in LEGS:
        try:
            lat, ms, times, cache = timed(tau, key, cap)
            frames = decode(lat)
            diff = (frames.float() - ref_frames.float()).abs()
            max_abs = diff.max().item()
            mse = (diff ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * torch.log10(
                torch.tensor(255.0 ** 2 / mse)).item()
            reused = cache.steps_reused
            if reused == 0:
                status, reason = "rejected", (
                    "technique never engaged (0 steps reused): any timing "
                    "difference cannot be attributed to this pass")
            elif max_abs == 0:
                status, reason = "accepted", "exact: frames bit-identical"
            elif psnr >= PSNR_BUDGET_DB:
                status, reason = "accepted", (
                    f"bounded: PSNR {psnr:.1f}dB >= budget "
                    f"{PSNR_BUDGET_DB}dB, max_abs {max_abs:.0f}/255")
            else:
                status, reason = "rejected", (
                    f"quality outside declared budget: PSNR {psnr:.1f}dB < "
                    f"{PSNR_BUDGET_DB}dB, max_abs {max_abs:.0f}/255")
            legs.append({"leg": f"cache_{key}_tau{tau}_cap{cap}", "key": key,
                         "threshold": tau, "consecutive_cap": cap,
                         "median_ms": ms, "times_ms": times,
                         "steps_total": cache.steps_total,
                         "steps_reused": reused,
                         "speedup": round(ref_ms / ms, 4),
                         "max_abs_255": max_abs, "psnr_db": round(psnr, 2),
                         "status": status, "reason": reason})
            print(f"[leg key={key} tau={tau} cap={cap}] {ms:.1f}ms "
                  f"({ref_ms/ms:.2f}x) reused={reused}/{cache.steps_total} "
                  f"psnr={psnr:.1f}dB -> {status}: {reason}", flush=True)
        except Exception as exc:   # one leg must not abort the sweep
            legs.append({"leg": f"cache_{key}_tau{tau}_cap{cap}", "key": key,
                         "threshold": tau, "consecutive_cap": cap,
                         "status": "failed", "reason":
                         f"{type(exc).__name__}: {exc}"})
            print(f"[leg key={key} tau={tau}] FAILED "
                  f"{type(exc).__name__}: {exc}", flush=True)

    summary = {"model": "Wan-AI/Wan2.2-TI2V-5B", "technique": "reuse_cache",
               "cache_site": "model_evaluation", "workload":
               f"{FRAMES}f {H}x{W} {STEPS} steps cfg{CFG} seed{SEED}",
               "scheduler": type(sch).__name__,
               "psnr_budget_db": PSNR_BUDGET_DB,
               "peak_vram_gb": round(torch.cuda.max_memory_allocated()
                                     / 2 ** 30, 1),
               "legs": legs}
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"wan22_stepcache_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    print("STEPCACHE_WAN22_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        print(f"STEPCACHE_WAN22_FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
