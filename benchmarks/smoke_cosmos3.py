"""Cosmos3-Nano onboarding smoke: first measured baseline, honestly tiered.

Loads the locally staged checkpoint through the library omni pipeline,
introspects the call signature (printed as evidence), runs a minimal
fixed-seed text-to-image generation, and reports wall times as JSON.
This is a *Launchable-tier* probe — it certifies "the model runs here
end-to-end and this is its unoptimized latency", nothing more. Any
failure prints an explicit marker instead of a partial success.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Cosmos3-Nano"
PROMPT = ("A robot arm placing a red block on a wooden table, "
          "studio lighting, photorealistic")
SEED = 1234
REPS = 2


def main() -> int:
    import torch
    from diffusers import Cosmos3OmniPipeline

    t0 = time.monotonic()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    load_s = time.monotonic() - t0
    print(f"[load] pipeline ready in {load_s:.0f}s", flush=True)

    sig = inspect.signature(pipe.__call__)
    params = list(sig.parameters)
    print(f"[introspect] __call__ params: {params}", flush=True)

    kwargs = {"prompt": PROMPT,
              "generator": torch.Generator("cuda").manual_seed(SEED)}
    for knob, val in (("num_inference_steps", 20),
                      ("height", 480), ("width", 832),
                      ("output_type", "pil")):
        if knob in params:
            kwargs[knob] = val

    times = []
    result = None
    for i in range(REPS):
        torch.cuda.synchronize()
        t = time.monotonic()
        result = pipe(**kwargs)
        torch.cuda.synchronize()
        times.append((time.monotonic() - t) * 1e3)
        print(f"[run {i}] {times[-1]:.0f} ms", flush=True)

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = getattr(result, "images", None)
    artifact = ""
    if images:
        artifact = str(out_dir / "cosmos3_smoke_t2i.png")
        images[0].save(artifact)
    summary = {
        "model": "nvidia/Cosmos3-Nano", "task": "t2i-smoke",
        "load_s": round(load_s, 1),
        "median_ms": sorted(times)[len(times) // 2],
        "times_ms": times, "seed": SEED,
        "steps": kwargs.get("num_inference_steps"),
        "peak_vram_gb": round(
            torch.cuda.max_memory_allocated() / 2**30, 1),
        "artifact": artifact,
    }
    (out_dir / f"cosmos3_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    print("COSMOS3_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        print(f"COSMOS3_SMOKE_FAIL: {type(exc).__name__}: {exc}",
              flush=True)
        raise SystemExit(1)
