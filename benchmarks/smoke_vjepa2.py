"""V-JEPA 2 (ViT-L) Launchable smoke: world-model predictor loop.

The world-model archetype's serving shape is an iterative latent
predictor driven from an encoded context. This smoke certifies that
shape on the locally staged HF-format checkpoint: encode a fixed-seed
synthetic clip once, then run the masked predictor in a rollout-shaped
loop, reusing the cached encoder features every step — the exact
pattern where naive re-prefill wastes quadratic work.

Honest scope: this is the BASE encoder+predictor (masked prediction),
NOT the action-conditioned rollout task — the AC checkpoint lives only
in the native upstream repository. Tier claimed: Launchable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/vjepa2-vitl-fpc64-256"
FRAMES = 16
SIZE = 256
STEPS = 8          # rollout-shaped predictor iterations
REPS = 3


def main() -> int:
    import torch
    from transformers import AutoModel

    t0 = time.monotonic()
    model = AutoModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.monotonic() - t0
    print(f"[load] ready in {load_s:.0f}s "
          f"({type(model).__name__})", flush=True)

    gen = torch.Generator("cpu").manual_seed(1234)
    clip = torch.rand((1, FRAMES, 3, SIZE, SIZE), generator=gen)
    clip = clip.to("cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        # context encoded ONCE; the predictor loop reuses it every step
        t = time.monotonic()
        out = model(pixel_values_videos=clip)
        torch.cuda.synchronize()
        encode_ms = (time.monotonic() - t) * 1e3
    ctx = out.last_hidden_state
    print(f"[encode] {encode_ms:.0f} ms, context {tuple(ctx.shape)}",
          flush=True)

    import inspect

    n_tokens = ctx.shape[1]
    tail = max(1, n_tokens // FRAMES)          # one frame-slice of tokens
    rollout_path = {"mode": None, "detail": ""}

    def _try_predictor_step(state):
        """Cached-context predictor step; raises if the API disagrees."""
        params = set(inspect.signature(model.predictor.forward).parameters)
        ctx_ids = torch.arange(n_tokens - tail, device="cuda").unsqueeze(0)
        tgt_ids = torch.arange(n_tokens - tail, n_tokens,
                               device="cuda").unsqueeze(0)
        kw = {}
        for name, val in (("hidden_states", state),
                          ("encoder_hidden_states", state),
                          ("context_mask", [ctx_ids]),
                          ("target_mask", [tgt_ids]),
                          ("position_masks", [tgt_ids])):
            if name in params and name not in kw:
                kw[name] = val
        pred = model.predictor(**kw)
        nxt = getattr(pred, "last_hidden_state", None)
        if nxt is None:
            nxt = pred[0]
        return torch.cat([state[:, tail:, :], nxt[:, -tail:, :]], dim=1)

    def rollout() -> float:
        state = ctx
        torch.cuda.synchronize()
        t = time.monotonic()
        with torch.inference_mode():
            for _ in range(STEPS):
                if rollout_path["mode"] in (None, "cached-predictor"):
                    try:
                        state = _try_predictor_step(state)
                        rollout_path["mode"] = "cached-predictor"
                        continue
                    except Exception as exc:  # noqa: BLE001 — fall back loud
                        if rollout_path["mode"] is None:
                            rollout_path["mode"] = "naive-reencode"
                            rollout_path["detail"] = (
                                f"predictor API mismatch "
                                f"({type(exc).__name__}: {str(exc)[:80]}); "
                                f"measuring the naive full re-encode "
                                f"rollout instead — the exact baseline "
                                f"pattern cached serving exists to beat")
                            print(f"[rollout] {rollout_path['detail']}",
                                  flush=True)
                        else:
                            raise
                # naive path: full model forward per step (no caching)
                model(pixel_values_videos=clip)
        torch.cuda.synchronize()
        return (time.monotonic() - t) * 1e3

    rollout()                                   # warmup
    times = [rollout() for _ in range(REPS)]
    med = sorted(times)[len(times) // 2]
    summary = {
        "model": "vjepa2-vitl-fpc64-256 (local)",
        "task": "predictor-rollout-smoke (base masked prediction, NOT AC)",
        "frames": FRAMES, "size": SIZE, "steps": STEPS,
        "load_s": round(load_s, 1), "encode_ms": round(encode_ms, 1),
        "rollout_ms": med, "per_step_ms": round(med / STEPS, 1),
        "rollout_path": rollout_path,
        "times_ms": times,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"vjepa2_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
     ).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    print("VJEPA2_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        print(f"VJEPA2_SMOKE_FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
