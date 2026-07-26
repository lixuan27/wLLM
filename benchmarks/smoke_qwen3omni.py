"""Qwen3-Omni-30B-A3B thinker smoke: the omni archetype, text path.

Launchable-tier probe of the thinker-talker omni archetype: load the
MoE checkpoint, disable the talker (text-only path — the audio chain
is a later tier), run fixed-seed greedy generation, report TTFT-proxy
and per-token decode rate. Adaptive: talker-disabling and audio-free
generation knobs are introspected, and whichever path actually ran is
recorded in the evidence.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
NEW_TOKENS = 64
REPS = 2


def main() -> int:
    import torch
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    path_notes = []
    if hasattr(model, "disable_talker"):
        model.disable_talker()
        path_notes.append("talker disabled via disable_talker()")
    load_s = time.monotonic() - t0
    print(f"[load] ready in {load_s:.0f}s; {'; '.join(path_notes) or 'no talker toggle found'}",
          flush=True)

    messages = [{"role": "user", "content": [
        {"type": "text",
         "text": "Explain in three sentences why caching a rollout "
                 "context beats re-encoding it every step."}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")

    gen_kwargs = dict(max_new_tokens=NEW_TOKENS, do_sample=False)
    gen_params = set(inspect.signature(model.generate).parameters)
    if "return_audio" in gen_params:
        gen_kwargs["return_audio"] = False
        path_notes.append("generate(return_audio=False)")

    def decode_once():
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        ids = out[0] if isinstance(out, tuple) else out
        return ids[0, inputs["input_ids"].shape[1]:]

    toks = decode_once()                       # warmup
    times = []
    for _ in range(REPS):
        t = time.monotonic()
        toks = decode_once()
        times.append((time.monotonic() - t) * 1e3)
    med = sorted(times)[len(times) // 2]
    n_tok = int(toks.shape[0])
    text = processor.tokenizer.decode(toks, skip_special_tokens=True)
    summary = {
        "model": MODEL_ID, "task": "thinker-text-smoke",
        "load_s": round(load_s, 1), "median_ms": med,
        "times_ms": times, "new_tokens": n_tok,
        "tok_per_s": round(n_tok / (med / 1e3), 1),
        "path_notes": path_notes,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "text_head": text[:160],
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qwen3omni_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    print("QWEN3OMNI_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        print(f"QWEN3OMNI_SMOKE_FAIL: {type(exc).__name__}: {exc}",
              flush=True)
        raise SystemExit(1)
