"""Cosmos3-Nano safety chain: measure it, then claim it.

The onboarding smoke (`benchmarks/smoke_cosmos3.py`) ran with the
safety checker explicitly disabled and printed a BLOCKER, because the
guardrail weights were gated. That path stays exactly as it was. This
benchmark is the other half: with the weights on disk, run the SAME
fixed prompt / seed / steps twice inside one process — once with the
safety chain off (the recorded baseline) and once with it on — and
report what the chain actually costs.

Design notes that matter for honesty:

* One pipeline, one set of weights, one process. The A/B is the
  pipeline's own per-request knob `enable_safety_check`, so the two
  configurations differ in nothing but the guardrail.
* The checker is constructed standalone (so its load time is isolated,
  not buried inside the pipeline load) and attached afterwards, on
  CPU — which is the steady state the pipeline is written for: it
  moves the checker to the device around each check and back to CPU
  after. The code executed inside `__call__` is the stock path.
* Runs are interleaved off/on/off/... after a discarded warmup, so
  clock and cache drift cannot masquerade as guardrail cost.
* The guardrail's frame postprocessor may rewrite pixels (it pixelates
  detected faces). We diff the guarded frames against the baseline
  frames and record whether it engaged — a guardrail that provably did
  nothing to the output is still evidence about *this* prompt, not
  evidence that the chain is a no-op.
* Nothing is claimed about overlapping the safety check with
  generation. If the cost is significant we record a hypothesis, and
  say in the same breath that it was not measured.

Absent or incomplete weights are a BLOCKER, not a silent skip.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.guardrail import (CacheRequirement, GuardrailTiming,
                                    attribute, check_hf_cache,
                                    overlap_hypothesis)

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Cosmos3-Nano"
# Identical to smoke_cosmos3.py — the comparison is only meaningful if
# the generation workload is bit-for-bit the same request.
PROMPT = ("A robot arm placing a red block on a wooden table, "
          "studio lighting, photorealistic")
SEED = 1234
STEPS = 20
HEIGHT, WIDTH = 480, 832
OUTPUT_TYPE = "pil"
REPS = 2

CACHE_ROOT = str(ROOT / "checkpoints" / "hf_guardrail_cache")
# What CosmosSafetyChecker actually reaches for at construction time:
# a blocklist + its nltk corpus, a face-blur postprocessor checkpoint,
# and a separate small generative guard model for the prompt check.
REQUIREMENTS = [
    # Blocklist() reads these four directories through
    # read_keyword_list_from_dir + nltk; RetinaFaceFilter() loads the
    # single face-blur checkpoint. Both go through snapshot_download.
    CacheRequirement("nvidia/Cosmos-1.0-Guardrail",
                     ("blocklist/custom", "blocklist/exact_match",
                      "blocklist/whitelist", "blocklist/nltk_data",
                      "face_blur_filter/Resnet50_Final.pth"),
                     note="blocklist + face-blur postprocessor"),
    # Qwen3Guard() is a SEPARATE repo resolved by transformers, not by
    # the guardrail checkpoint id — the prompt-side check cannot
    # construct without it.
    CacheRequirement("Qwen/Qwen3Guard-Gen-0.6B",
                     ("config.json", "generation_config.json",
                      "model.safetensors", "tokenizer.json",
                      "tokenizer_config.json"),
                     note="prompt-side generative guard"),
]


class _Probe:
    """Times the guardrail segments the pipeline runs per request.

    Wraps bound methods on the live objects, so what is timed is the
    stock call path rather than a re-implementation of it.
    """

    def __init__(self) -> None:
        self.text_check_ms: list[float] = []
        self.video_stage_ms: list[float] = []
        self.video_check_ms: list[float] = []
        self.transfer_ms: list[float] = []
        self.transfers: list[tuple[str, float]] = []
        self.text_verdicts: list[bool] = []

    def reset_request(self) -> None:
        self.transfers = []

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def install(self, pipe, checker) -> None:
        raw_text = checker.check_text_safety
        raw_video_check = checker.check_video_safety
        raw_to = checker.to
        raw_stage = pipe._apply_video_safety_check

        def text(prompt):
            t = self._now()
            out = raw_text(prompt)
            self.text_check_ms.append((self._now() - t) * 1e3)
            self.text_verdicts.append(bool(out))
            return out

        def video_check(frames):
            t = self._now()
            out = raw_video_check(frames)
            self.video_check_ms.append((self._now() - t) * 1e3)
            return out

        def to(device=None, dtype=None):
            t = self._now()
            out = raw_to(device=device, dtype=dtype)
            dt = (self._now() - t) * 1e3
            self.transfer_ms.append(dt)
            self.transfers.append((str(device), dt))
            return out

        def stage(video, output_type, device):
            t = self._now()
            out = raw_stage(video, output_type=output_type, device=device)
            self.video_stage_ms.append((self._now() - t) * 1e3)
            return out

        object.__setattr__(checker, "check_text_safety", text)
        object.__setattr__(checker, "check_video_safety", video_check)
        object.__setattr__(checker, "to", to)
        object.__setattr__(pipe, "_apply_video_safety_check", stage)


def _frames_to_array(video, np):
    """Guarded/baseline frames as a comparable uint8 array."""
    if isinstance(video, list):
        return np.stack([np.asarray(f) for f in video], axis=0)
    return np.asarray(video)


def main() -> int:
    # --------------------------------------------- 0. fail closed first
    readiness = check_hf_cache(CACHE_ROOT, REQUIREMENTS)
    print(f"[readiness] ready={readiness.ready} "
          f"checked={readiness.checked}", flush=True)
    for path in sorted(readiness.resolved.values()):
        print(f"[readiness] resolved snapshot: {path}", flush=True)
    for b in readiness.blockers:
        print(f"[readiness] BLOCKER: {b}", flush=True)
    if not readiness.ready:
        print("COSMOS3_GUARDRAIL_BLOCKED: safety-checker weights are not "
              "loadable from the cache tree; refusing to report a "
              "guardrail-enabled serving number", flush=True)
        return 2

    import numpy as np
    import torch
    from cosmos_guardrail import CosmosSafetyChecker
    from diffusers import Cosmos3OmniPipeline

    # ------------------------------------------------- 1. pipeline load
    t0 = time.monotonic()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        enable_safety_checker=False)
    pipe.to("cuda")
    pipe_load_s = time.monotonic() - t0
    print(f"[load] pipeline ready in {pipe_load_s:.1f}s", flush=True)

    kwargs = {"prompt": PROMPT, "num_inference_steps": STEPS,
              "height": HEIGHT, "width": WIDTH,
              "output_type": OUTPUT_TYPE}

    def run(enable: bool):
        torch.cuda.synchronize()
        t = time.monotonic()
        out = pipe(generator=torch.Generator("cuda").manual_seed(SEED),
                   enable_safety_check=enable, **kwargs)
        torch.cuda.synchronize()
        return out, (time.monotonic() - t) * 1e3

    # Discarded: the first request pays one-off autotune/allocator cost
    # that would otherwise be charged to whichever arm ran first.
    _, warm_ms = run(False)
    print(f"[warmup] discarded first request: {warm_ms:.0f} ms",
          flush=True)

    # ------------------------------------------------ 2. guardrail load
    t0 = time.monotonic()
    checker = CosmosSafetyChecker()
    guardrail_load_s = time.monotonic() - t0
    models = [type(m).__name__ for m in checker.models]
    print(f"[guardrail] safety chain loaded in {guardrail_load_s:.1f}s; "
          f"components={models}", flush=True)

    object.__setattr__(pipe, "safety_checker", checker)
    if not isinstance(pipe.safety_checker, CosmosSafetyChecker):
        print("COSMOS3_GUARDRAIL_BLOCKED: checker did not attach to the "
              "pipeline; the guarded arm would silently be the baseline",
              flush=True)
        return 1

    # ------------------------------------ 2b. can the chain say "no"?
    # A guardrail that passes everything is indistinguishable from a
    # disabled one. Two negative controls, run before instrumentation:
    # the empty-input branch (proves the verdict propagates) and a term
    # taken from the guardrail's OWN loaded blocklist (proves the
    # shipped weights are consulted). The term is never printed and
    # never stored in this repository.
    controls = {"empty_input_refused": checker.check_text_safety("") is False}
    by_name = {type(m).__name__: m for m in checker.models}
    blocklist = by_name.get("Blocklist")
    words: list[str] = []
    if blocklist is not None:
        for attr in ("exact_match_words", "blocklist_words"):
            words += sorted(getattr(blocklist, attr, None) or [])
    controls["blocklist_terms_loaded"] = len(words)
    # Probed against the blocklist component directly, not the whole
    # chain: a term that fails to trip the wordlist would otherwise
    # fall through to the CPU-resident guard model and cost minutes for
    # a control. Several candidates, because any single term could be
    # whitelisted or too short to trip fuzzy matching; one refusal is
    # enough to prove the shipped list is consulted.
    sample = words[:5]
    refused = sum(1 for w in sample
                  if blocklist is not None and blocklist.is_safe(w)[0] is False)
    controls["blocklist_terms_probed"] = len(sample)
    controls["blocklist_terms_refused"] = refused
    controls["blocklist_term_refused"] = refused > 0
    # The prompt-side guard model swallows its own exceptions and
    # returns "safe" — a fail-OPEN we must be able to see rather than
    # inherit silently.
    guard_model = by_name.get("Qwen3Guard")
    if guard_model is not None:
        checker.to("cuda")
        try:
            g_safe, g_msg = guard_model.is_safe(PROMPT)
        finally:
            checker.to("cpu")
        controls["prompt_guard_verdict"] = bool(g_safe)
        controls["prompt_guard_failed_open"] = "Unexpected error" in g_msg
    for k, v in controls.items():
        print(f"[control] {k} = {v}", flush=True)
    if not controls["empty_input_refused"] or \
            not controls.get("blocklist_term_refused", False):
        print("COSMOS3_GUARDRAIL_BLOCKED: the safety chain never refuses "
              "anything; an always-pass chain is not evidence of a "
              "restored guardrail", flush=True)
        return 1

    probe = _Probe()
    probe.install(pipe, checker)

    # ------------------------------------------------- 3. interleaved AB
    base_ms: list[float] = []
    guard_ms: list[float] = []
    base_frames = guard_frames = None
    per_request: list[dict] = []
    for i in range(REPS):
        _out, ms = run(False)
        base_ms.append(ms)
        print(f"[baseline {i}] {ms:.0f} ms", flush=True)
        if base_frames is None:
            base_frames = _frames_to_array(_out.video, np)

        probe.reset_request()
        n_text, n_stage = len(probe.text_check_ms), len(probe.video_stage_ms)
        _out, ms = run(True)
        guard_ms.append(ms)
        text_ms = probe.text_check_ms[n_text] if len(
            probe.text_check_ms) > n_text else 0.0
        stage_ms = probe.video_stage_ms[n_stage] if len(
            probe.video_stage_ms) > n_stage else 0.0
        # The pipeline moves the checker to the device and back around
        # the prompt check; those two transfers are part of the
        # prompt-side stage cost. Transfers inside the frame stage are
        # already inside `stage_ms`.
        text_transfer = sum(dt for _d, dt in probe.transfers[:2])
        per_request.append({
            "rep": i,
            "text_check_ms": text_ms,
            "text_transfer_ms": text_transfer,
            "text_stage_ms": text_ms + text_transfer,
            "video_stage_ms": stage_ms,
            "transfers_ms": [round(dt, 1) for _d, dt in probe.transfers],
        })
        print(f"[guarded  {i}] {ms:.0f} ms "
              f"(prompt check {text_ms:.0f} ms + transfer "
              f"{text_transfer:.0f} ms, frame check {stage_ms:.0f} ms)",
              flush=True)
        if guard_frames is None:
            guard_frames = _frames_to_array(_out.video, np)

    if not probe.text_verdicts:
        print("COSMOS3_GUARDRAIL_BLOCKED: the prompt check never ran in "
              "the guarded arm", flush=True)
        return 1
    benign_pass = all(probe.text_verdicts)
    print(f"[verdict] benign prompt passed the text guardrail: "
          f"{benign_pass} (verdicts={probe.text_verdicts})", flush=True)

    # ----------------------------------------- 4. did it touch output?
    frames_equal = None
    max_abs_diff = None
    changed_frames = None
    if base_frames is not None and guard_frames is not None and \
            base_frames.shape == guard_frames.shape:
        diff = np.abs(base_frames.astype(np.int16)
                      - guard_frames.astype(np.int16))
        max_abs_diff = int(diff.max())
        changed_frames = int((diff.reshape(diff.shape[0], -1).max(axis=1)
                              > 0).sum())
        frames_equal = max_abs_diff == 0
    print(f"[postprocess] frames identical to baseline={frames_equal} "
          f"max_abs_diff={max_abs_diff} changed_frames={changed_frames}",
          flush=True)

    # --------------------------------------------------- 5. accounting
    def med(xs):
        return sorted(xs)[len(xs) // 2]

    timing = GuardrailTiming(
        baseline_ms=med(base_ms),
        guarded_ms=med(guard_ms),
        text_stage_ms=med([r["text_stage_ms"] for r in per_request]),
        video_stage_ms=med([r["video_stage_ms"] for r in per_request]),
        transfer_ms=med([r["text_transfer_ms"] for r in per_request]),
    )
    acct = attribute(timing)
    hypothesis = overlap_hypothesis(acct)
    print(f"[overhead] +{acct['overhead_ms']:.0f} ms "
          f"({acct['overhead_pct']:.1f}%) — attributed "
          f"{acct['attributed_ms']:.0f} ms, unattributed "
          f"{acct['unattributed_ms']:.0f} ms, coherent="
          f"{acct['attribution_coherent']}", flush=True)
    if hypothesis:
        print(f"[hypothesis] {hypothesis}", flush=True)

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "nvidia/Cosmos3-Nano",
        "task": "text2world guardrail A/B",
        "workload": (f"text2world 189f {HEIGHT}x{WIDTH}, {STEPS} steps"),
        "prompt": PROMPT, "seed": SEED, "steps": STEPS,
        "reps": REPS,
        "pipeline_load_s": round(pipe_load_s, 1),
        "guardrail_load_s": round(guardrail_load_s, 1),
        "guardrail_components": models,
        "attach_mode": "constructed standalone, attached post-load (CPU "
                       "resident); per-request path is the stock "
                       "enable_safety_check branch",
        "baseline_times_ms": base_ms,
        "guarded_times_ms": guard_ms,
        "per_request": per_request,
        "video_check_ms": probe.video_check_ms,
        "benign_prompt_passed": benign_pass,
        "text_verdicts": probe.text_verdicts,
        "negative_controls": controls,
        "frames_identical_to_baseline": frames_equal,
        "frames_max_abs_diff": max_abs_diff,
        "frames_changed": changed_frames,
        "accounting": acct,
        "hypothesis": hypothesis,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "readiness": readiness.to_dict(),
    }
    path = out_dir / f"cosmos3_guardrail_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("readiness", "hypothesis")}), flush=True)
    print(f"[artifact] {path}", flush=True)

    if not benign_pass:
        print("COSMOS3_GUARDRAIL_BLOCKED: the benign control prompt was "
              "refused by the text guardrail; the chain is not usable as "
              "a serving default", flush=True)
        return 1
    print("COSMOS3_GUARDRAIL_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — explicit failure marker
        # The one-line marker is for the sbatch grep; the traceback is
        # for whoever has to diagnose it. A previous run lost the
        # identity of a missing cache artifact to a bare exception
        # string, so print both — never just the summary.
        import traceback
        print(f"COSMOS3_GUARDRAIL_FAIL: {type(exc).__name__}: {exc}",
              flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise SystemExit(1)
