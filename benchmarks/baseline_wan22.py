"""First real-model L1 baseline: Wan2.2-TI2V-5B (local diffusers layout).

Wraps the diffusers text-to-video call as a wLLM Application (L1) and runs
the baseline profiler on 1 GPU.  Evidence lands in benchmarks/results/.
This is the anchor every optimized plan for this model is measured against.
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
HEIGHT, WIDTH, FRAMES, STEPS = 480, 832, 33, 20


def main() -> int:
    import torch
    from diffusers import WanPipeline

    from wllm.api import Application

    t_load0 = time.monotonic()
    pipe = WanPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    load_s = time.monotonic() - t_load0
    print(f"[load] WanPipeline ready in {load_s:.1f}s", flush=True)

    def run(prompt: str):
        out = pipe(prompt=prompt, height=HEIGHT, width=WIDTH,
                   num_frames=FRAMES, num_inference_steps=STEPS,
                   generator=torch.Generator("cuda").manual_seed(1234))
        return out.frames[0]

    app = Application.from_callable(run, example_inputs={"prompt": PROMPT},
                                    name="wan22_ti2v_5b")
    report = app.baseline(repeats=2, warmup=1,
                          save_dir=ROOT / "benchmarks/results")
    report.meta.update({
        "model_dir": MODEL_DIR, "height": HEIGHT, "width": WIDTH,
        "num_frames": FRAMES, "steps": STEPS, "load_seconds": load_s,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    })
    path = report.save(ROOT / "benchmarks/results", tag="baseline_meta")
    print(json.dumps({
        "median_ms": report.median_ms, "p95_ms": report.p95_ms,
        "sec_per_frame": report.median_ms / 1000.0 / FRAMES,
        "gpu_mem_peak_gb": (report.gpu_mem_peak_bytes or 0) / 1e9,
        "evidence": str(path),
    }, indent=1), flush=True)
    print("BASELINE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
