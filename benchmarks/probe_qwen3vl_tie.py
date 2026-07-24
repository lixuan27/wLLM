"""Diagnostic probe: is the static-cache greedy flip a near-tie numerics
boundary or a real divergence?  Captures per-step top-2 logit gaps for the
dynamic-eager and static-cache paths and reports the gap at the first
mismatching position.  Evidence for verifier design (like-for-like
numerics references vs cross-numerics distribution gates)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Qwen3-VL-8B-Instruct"
N = 48  # enough to cover pos 36


def run():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (448, 448, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in detail, then list "
                                 "three plausible uses for it."}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")

    def gen(cache=None):
        kw = dict(max_new_tokens=N, do_sample=False, output_scores=True,
                  return_dict_in_generate=True)
        if cache:
            kw["cache_implementation"] = cache
        with torch.inference_mode():
            out = model.generate(**inputs, **kw)
        toks = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        gaps = []
        for step_scores in out.scores:
            top2 = torch.topk(step_scores[0].float(), 2).values
            gaps.append(float(top2[0] - top2[1]))
        return toks, gaps

    tok_a, gap_a = gen(None)
    tok_b, gap_b = gen("static")
    mismatch = next((i for i, (a, b) in enumerate(zip(tok_a, tok_b))
                     if a != b), None)
    report = {
        "first_mismatch_pos": mismatch,
        "gap_dynamic_at_pos": gap_a[mismatch] if mismatch is not None else None,
        "gap_static_at_pos": gap_b[mismatch] if mismatch is not None else None,
        "median_gap_dynamic": sorted(gap_a)[len(gap_a) // 2],
        "min_gap_dynamic_first40": min(gap_a[:40]),
        "tokens_dynamic_around": tok_a[max(0, (mismatch or 0) - 2):(mismatch or 0) + 3],
        "tokens_static_around": tok_b[max(0, (mismatch or 0) - 2):(mismatch or 0) + 3],
    }
    out_p = ROOT / "benchmarks/results" / f"qwen3vl_tie_probe_{time.strftime('%H%M%S')}.json"
    out_p.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    run()
