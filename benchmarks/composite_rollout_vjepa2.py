"""K concurrent world-model rollout sessions through the composite runtime.

The composite graph runtime (:mod:`wllm.composite`) has only ever been
exercised over synthetic components, so its two load-bearing claims —
per-session state isolation, and step batching that preserves per-request
results — were contract-level, not measured. This benchmark drives them
with a real model on a real GPU.

Workload. A V-JEPA 2 ViT-L checkpoint staged locally. Encoding a clip
costs ~1.2 s cold; one predictor step over the cached encoder features
costs ~16 ms (job 201778). A rollout is therefore a loop whose expensive
prefix must be computed once and reused — exactly the shape the runtime
exists to express:

    graph     context_encoder --> rollout_step
    walks     "ground"  = Seq(context_encoder)
              "rollout" = Loop(Seq(rollout_step), carry="latent", iterations=STEPS)
    request   state machine: ground -> rollout -> done

The encoder caches its features in SESSION state, so the second request
on a session re-grounds for free — measured here as cold vs warm encode.

Three arms, one GPU:

    solo         each session on its OWN executor and SessionStore, run
                 one at a time, every step a batch of BRANCH. This is the
                 reference for both verdicts and the sequential
                 throughput arm.
    concurrent   all K sessions on ONE executor and ONE SessionStore, K
                 threads, still one step per call (gate width 1). Same
                 arithmetic as solo, arbitrary interleaving. Any
                 difference from solo is state leaking between sessions,
                 so the isolation verdict demands BIT-IDENTICAL outputs.
    batched      all K sessions on one executor, K threads, gate width K,
                 so the runtime's batching layer fuses K sessions' steps
                 into one call of K*BRANCH. Batching may legally change
                 numerics (house law: quantify, do not assume), so its
                 parity is measured against solo and classified against a
                 budget declared below, before any measurement.

Every session gets a DIFFERENT clip, so a leaked cache would show up as
an O(1) difference, not as rounding.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.composite import (Component, ComponentGraph, Edge, Loop, Seq,
                            StepBatcher, StepGate, Walk, WalkExecutor,
                            WalkSet, current_session, lower_plan, require,
                            run_request)
from wllm.graph.regions import NodeOp
from wllm.graph.states import StateKind, StateScope, StateSpec
from wllm.planner.plan import DeploymentPlan, Hardware, Stage


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


MODEL_DIR = os.environ.get(
    "WLLM_CR_MODEL",
    "/public/home/lixuan/lixuan/pretrained-model/vjepa2-vitl-fpc64-256")
SESSIONS = _env_int("WLLM_CR_SESSIONS", 64)   # K concurrent rollout sessions
BRANCH = _env_int("WLLM_CR_BRANCH", 8)        # per-session batch shape
STEPS = _env_int("WLLM_CR_STEPS", 8)          # rollout steps per session
FRAMES = _env_int("WLLM_CR_FRAMES", 16)
SIZE = _env_int("WLLM_CR_SIZE", 256)
GATE_WAIT_S = 90.0        # peers this far apart still share a round
GATE_TIMEOUT_S = 600.0    # past this a request raises instead of hanging

# Declared BEFORE any measurement. Batching here is a pure scheduling
# decision: the predictor has no cross-sample interaction, so a fused call
# computes the same function per sample. What can still move is bf16
# reduction order and kernel selection at a different batch size. bf16
# carries an 8-bit mantissa (eps = 2^-8 = 3.9e-3) and STEPS predictor
# passes are chained, so a few eps of relative drift is expected and legal;
# anything past this budget is not "batching noise" and gets rejected.
PARITY_BUDGET = {
    "rel_l2": 2.0e-2,       # ||batched - solo||_2 / ||solo||_2
    "rel_max_abs": 5.0e-2,  # max|batched - solo| / max|solo|
}
# The utilization redline this benchmark must live inside: a job holding a
# GPU at ~0.5 GB is auto-killed. K and BRANCH are chosen for the realistic
# concurrent-serving regime, which also clears the floor.
VRAM_FLOOR_GB = 12.0


# --------------------------------------------------------------- graph
def build_graph() -> ComponentGraph:
    """The composite model: an encoder and an iterative predictor."""
    return ComponentGraph(
        name="world-model-rollout",
        components=[
            Component(
                "context_encoder", NodeOp.ENCODER, batchable=False,
                states=[StateSpec(
                    id="context_cache",
                    kind=StateKind.RECOMPUTABLE_FEATURE,
                    scope=StateScope.SESSION, recomputable=True,
                    owner="context_encoder",
                    description="encoder features for this session's clip: "
                                "computed once, reused by every rollout "
                                "step and by later requests")]),
            Component(
                "rollout_step", NodeOp.TRANSFORMER, batchable=True,
                states=[StateSpec(
                    id="rollout_trace", kind=StateKind.ROLLING_CONTEXT,
                    scope=StateScope.SESSION, owner="rollout_step",
                    description="how far this session's rollout has run")]),
        ],
        edges=[Edge("context_encoder", "rollout_step")])


def build_walkset(steps: int) -> WalkSet:
    return WalkSet(walks={
        "ground": Walk([Seq("context_encoder")]),
        "rollout": Walk([Loop(body=Walk([Seq("rollout_step")]),
                              carry="latent", iterations=steps)]),
    })


def build_placement(graph: ComponentGraph) -> dict[str, str]:
    """Placement is data: derive it from a plan, fail closed."""
    plan = DeploymentPlan(
        id="composite-rollout-1gpu",
        stages=[Stage(id="grounding", node_ids=["context_encoder"],
                      device=0),
                Stage(id="rollout", node_ids=["rollout_step"], device=0)],
        notes="single-GPU concurrent rollout serving")
    hardware = Hardware(num_gpus=1, hbm_bytes_per_gpu=144 * 2 ** 30)
    return require(lower_plan(graph, plan, hardware))


def ground_only(request_ctx, last_walk, last_output):
    """Request 1: ground the session (cold encode), nothing else."""
    return "ground" if last_walk is None else None


def ground_then_rollout(request_ctx, last_walk, last_output):
    """Request 2: re-ground (cache hit) and roll the world forward."""
    if last_walk is None:
        return "ground"
    if last_walk == "ground":
        return "rollout"
    return None


# ------------------------------------------------------- implementations
def make_clip_bank(torch, sessions, branch, frames, size, seed0=20260727):
    """One distinct fixed-seed clip batch per session, held on CPU."""
    bank = {}
    for i, sid in enumerate(sessions):
        gen = torch.Generator("cpu").manual_seed(seed0 + i)
        bank[sid] = torch.rand((branch, frames, 3, size, size),
                               generator=gen).to(torch.bfloat16)
    return bank


def make_predictor_step(torch, model, device):
    """(fused step fn, batched-fn for StepBatcher).

    The step consumes a cached context ``[M, N, D]`` and returns the
    context advanced by one predicted frame-slice. Fusing M requests is a
    scheduling decision only: the predictor never mixes samples.
    """

    def forward(state):
        m, n_tokens, _ = state.shape
        tail = max(1, n_tokens // FRAMES)
        ctx_ids = torch.arange(n_tokens - tail, device=device
                               ).unsqueeze(0).expand(m, -1)
        tgt_ids = torch.arange(n_tokens - tail, n_tokens, device=device
                               ).unsqueeze(0).expand(m, -1)
        with torch.inference_mode():
            pred = model.predictor(encoder_hidden_states=state,
                                   context_mask=[ctx_ids],
                                   target_mask=[tgt_ids])
            nxt = pred.last_hidden_state
            return torch.cat([state[:, tail:, :], nxt[:, -tail:, :]], dim=1)

    def batched(payloads):
        if len(payloads) == 1:
            # a fuse of one is the identity; skipping the cat keeps the
            # unbatched arm's arithmetic literally the solo arm's
            return [forward(payloads[0])]
        sizes = [int(p.shape[0]) for p in payloads]
        fused = forward(torch.cat(payloads, dim=0))
        return list(torch.split(fused, sizes, dim=0))

    return forward, batched


def make_impls(torch, model, clips, gate, device):
    """Component implementations bound to one gate (i.e. one arm)."""

    def context_encoder(ctx, state):
        session = current_session()
        if "context" in state:
            state["cache_hits"] = state.get("cache_hits", 0) + 1
            return {"latent": state["context"], "grounded_by": "cache"}
        started = time.perf_counter()
        with torch.inference_mode():
            out = model(pixel_values_videos=clips[session].to(device))
            features = out.last_hidden_state
        # NOTE on `cold_encode_ms`: this is a device-wide sync, so in the
        # THREADED arms every encoder waits out all 64 concurrent encodes
        # and the recorded per-encode figure is inflated by the fleet, not
        # by this session. Only the solo arm's number is a per-encode
        # latency; the phase-level `ground_wall_s` is the honest
        # cross-arm comparison.
        torch.cuda.synchronize()
        state["context"] = features
        state["cold_encode_ms"] = (time.perf_counter() - started) * 1e3
        state["cold_encodes"] = state.get("cold_encodes", 0) + 1
        return {"latent": features, "grounded_by": "encode"}

    def rollout_step(ctx, state):
        session = current_session()
        latent = ctx["latent"]
        # what makes a fused call legal: same component, same shape, same
        # dtype. The step index does not enter the predictor call, so it
        # is not part of the compatibility key.
        signature = (tuple(int(d) for d in latent.shape), str(latent.dtype))
        nxt = gate.submit(session, "rollout_step", latent, signature)
        state["steps"] = state.get("steps", 0) + 1
        return {"latent": nxt}

    return {"context_encoder": context_encoder, "rollout_step": rollout_step}


# ------------------------------------------------------------- driving
def run_phase(sessions, executor_for, walkset, chooser, gate, threaded,
              settle):
    """One request per session; returns ({session: result}, wall seconds).

    ``settle`` runs inside the timed region after the last walk returns:
    component calls hand back device tensors without synchronizing, so a
    wall time taken before the device drains would be fiction.

    Every session registers with the gate BEFORE any thread starts. A
    round is ``min(width, live)`` wide, so registering from inside the
    workers would let an early submitter fire a round of two while the
    rest of the fleet was still spawning — the batched arm would then be
    measuring a batch size nobody asked for.
    """
    results, errors = {}, {}

    def one(session):
        results[session] = run_request(executor_for(session), walkset,
                                       chooser, session=session)

    def guarded(session):
        def body():
            try:
                one(session)
            except BaseException as exc:      # noqa: BLE001 — surfaced below
                errors[session] = f"{type(exc).__name__}: {exc}"
        return body

    for _ in sessions:
        gate.join()
    try:
        started = time.perf_counter()
        if threaded:
            threads = [threading.Thread(target=guarded(s), name=f"walk-{s}")
                       for s in sessions]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for session in sessions:
                guarded(session)()
        settle()
        wall = time.perf_counter() - started
    finally:
        for _ in sessions:
            gate.leave()

    if errors:
        first = sorted(errors.items())[:3]
        raise RuntimeError(f"{len(errors)}/{len(sessions)} sessions failed; "
                           f"first: {first}")
    return results, wall


def run_arm(torch, name, sessions, graph, walkset, placement, model, clips,
            device, width, threaded):
    """Ground every session, then roll out; return outputs and timings."""
    _, batched = make_predictor_step(torch, model, device)
    batcher = StepBatcher(batched_fns={"rollout_step": batched},
                          max_batch=width)
    gate = StepGate(batcher, width=width, wait_s=GATE_WAIT_S,
                    timeout_s=GATE_TIMEOUT_S)
    impls = make_impls(torch, model, clips, gate, device)

    if name == "solo":
        # "alone" is literal: a private executor AND a private SessionStore
        # per session, so there is no shared object to leak through.
        stores = {s: WalkExecutor(graph, impls, placement=dict(placement))
                  for s in sessions}
        executors = list(stores.values())

        def executor_for(session):
            return stores[session]
    else:
        shared = WalkExecutor(graph, impls, placement=dict(placement))
        executors = [shared]

        def executor_for(session):
            return shared

    _, ground_wall = run_phase(sessions, executor_for, walkset, ground_only,
                               gate, threaded, torch.cuda.synchronize)
    results, rollout_wall = run_phase(sessions, executor_for, walkset,
                                      ground_then_rollout, gate, threaded,
                                      torch.cuda.synchronize)

    with torch.inference_mode():
        outputs = {s: r.outputs["latent"].to("cpu").clone()
                   for s, r in results.items()}

    trail = sorted({tuple(r.walk_trail) for r in results.values()})
    invocations: dict[str, int] = {}
    devices: dict[str, list[str]] = {}
    cold_ms, warm_hits = [], 0
    for ex in executors:
        for inv in ex.invocations:
            invocations[inv.component] = invocations.get(inv.component, 0) + 1
        for comp, devs in ex.devices_used().items():
            devices.setdefault(comp, [])
            devices[comp] = sorted(set(devices[comp]) | devs)
        for session in ex.store.sessions():
            enc = ex.store.state(session, "context_encoder")
            if "cold_encode_ms" in enc:
                cold_ms.append(enc["cold_encode_ms"])
            warm_hits += enc.get("cache_hits", 0)

    stats = gate.stats()
    report = {
        "arm": name,
        "gate_width": width,
        "threaded": threaded,
        "ground_wall_s": round(ground_wall, 3),
        "rollout_wall_s": round(rollout_wall, 3),
        "steps": len(sessions) * STEPS,
        "steps_per_s": round(len(sessions) * STEPS / rollout_wall, 2),
        "walk_trails": [list(t) for t in trail],
        "invocations": invocations,
        "devices_used": devices,
        "batcher_max_group": batcher.max_group_size(),
        "batcher_calls": len(batcher.records),
        "cross_signature_mixes": batcher.cross_signature_mixes(),
        "gate_rounds": stats.rounds,
        "gate_fused_rounds": stats.fused_rounds,
        "gate_max_round": stats.max_round,
        "gate_partial_rounds": stats.partial_rounds,
        "cold_encode_ms_median": (round(sorted(cold_ms)[len(cold_ms) // 2], 1)
                                  if cold_ms else None),
        "cold_encodes": len(cold_ms),
        "warm_ground_cache_hits": warm_hits,
    }
    return outputs, report


# ---------------------------------------------------------- comparison
def compare(torch, ref, cand):
    with torch.inference_mode():
        a, b = ref.float(), cand.float()
        diff = a - b
        max_abs = float(diff.abs().max())
        ref_max = float(a.abs().max())
        ref_l2 = float(a.norm())
        return {
            "max_abs": max_abs,
            "rel_max_abs": max_abs / ref_max if ref_max else float("inf"),
            "rel_l2": float(diff.norm()) / ref_l2 if ref_l2 else float("inf"),
            "ref_mean": float(a.mean()), "cand_mean": float(b.mean()),
            "ref_std": float(a.std()), "cand_std": float(b.std()),
        }


def summarize_diffs(torch, ref_out, cand_out):
    per_session = {s: compare(torch, ref_out[s], cand_out[s])
                   for s in sorted(ref_out)}
    worst_key = max(per_session, key=lambda s: per_session[s]["rel_l2"])
    return {
        "sessions": len(per_session),
        "max_abs_over_sessions": max(d["max_abs"]
                                     for d in per_session.values()),
        "max_rel_l2": max(d["rel_l2"] for d in per_session.values()),
        "max_rel_max_abs": max(d["rel_max_abs"]
                               for d in per_session.values()),
        "worst_session": worst_key,
        "worst": per_session[worst_key],
        # Verifier law 4: pointwise change and distribution change are two
        # different questions and are reported separately.
        "max_mean_shift": max(abs(d["cand_mean"] - d["ref_mean"])
                              for d in per_session.values()),
        "max_std_shift": max(abs(d["cand_std"] - d["ref_std"])
                             for d in per_session.values()),
        "per_session": {s: {k: d[k] for k in
                            ("max_abs", "rel_max_abs", "rel_l2")}
                        for s, d in per_session.items()},
    }


def classify_parity(diffs, budget):
    if diffs["max_abs_over_sessions"] == 0.0:
        return "exact"
    if (diffs["max_rel_l2"] <= budget["rel_l2"]
            and diffs["max_rel_max_abs"] <= budget["rel_max_abs"]):
        return "bounded"
    return "over_budget"


# --------------------------------------------------------------- main
def _session_ids(n, prefix="s"):
    return [f"{prefix}{i:03d}" for i in range(n)]


def _reclaim(torch):
    gc.collect()
    torch.cuda.empty_cache()


def _run_all_arms(torch, sessions, graph, walkset, placement, model, clips,
                  device, width):
    solo_out, solo_rep = run_arm(torch, "solo", sessions, graph, walkset,
                                 placement, model, clips, device,
                                 width=1, threaded=False)
    _reclaim(torch)
    conc_out, conc_rep = run_arm(torch, "concurrent", sessions, graph,
                                 walkset, placement, model, clips, device,
                                 width=1, threaded=True)
    _reclaim(torch)
    isolation = summarize_diffs(torch, solo_out, conc_out)
    del conc_out
    _reclaim(torch)
    batch_out, batch_rep = run_arm(torch, "batched", sessions, graph,
                                   walkset, placement, model, clips, device,
                                   width=width, threaded=True)
    parity = summarize_diffs(torch, solo_out, batch_out)
    del solo_out, batch_out
    _reclaim(torch)
    return {"solo": solo_rep, "concurrent": conc_rep, "batched": batch_rep}, \
        isolation, parity


def main() -> int:
    import torch
    from transformers import AutoModel

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("composite rollout benchmark needs a GPU")

    t0 = time.monotonic()
    model = AutoModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to(device).eval()
    load_s = time.monotonic() - t0
    print(f"[load] {type(model).__name__} ready in {load_s:.0f}s", flush=True)

    graph = build_graph()
    problems = graph.validate()
    if problems:
        raise RuntimeError(f"component graph invalid: {problems}")
    walkset = build_walkset(STEPS)
    problems = walkset.validate(graph)
    if problems:
        raise RuntimeError(f"walk set invalid: {problems}")
    placement = build_placement(graph)
    print(f"[graph] {[c.id for c in graph.components]} "
          f"walks={sorted(walkset.walks)} placement={placement}", flush=True)

    # Preflight: the whole pipeline at toy scale. Cheap, and it turns a
    # plumbing bug into a 20-second failure instead of a wasted GPU hour.
    pre_sessions = _session_ids(3, prefix="pre")
    pre_clips = make_clip_bank(torch, pre_sessions, 1, FRAMES, SIZE,
                               seed0=7000)
    pre_arms, pre_iso, pre_par = _run_all_arms(
        torch, pre_sessions, graph, walkset, placement, model, pre_clips,
        device, width=len(pre_sessions))
    print(f"[preflight] isolation max_abs="
          f"{pre_iso['max_abs_over_sessions']:.3e} "
          f"parity={classify_parity(pre_par, PARITY_BUDGET)} "
          f"batched_group={pre_arms['batched']['batcher_max_group']}",
          flush=True)
    if pre_iso["max_abs_over_sessions"] != 0.0:
        # Loud, but not fatal: the full-scale run is the measurement worth
        # reporting, and a runtime defect deserves its numbers published,
        # not an early exit that hides them.
        print("PREFLIGHT ISOLATION WARNING: concurrent execution already "
              "changed outputs at toy scale; the full-scale verdict below "
              "is the one that counts", flush=True)
    del pre_clips
    gc.collect()
    torch.cuda.empty_cache()

    sessions = _session_ids(SESSIONS)
    t = time.monotonic()
    clips = make_clip_bank(torch, sessions, BRANCH, FRAMES, SIZE)
    print(f"[data] {SESSIONS} distinct clip batches "
          f"({BRANCH}x{FRAMES}x3x{SIZE}x{SIZE} bf16) in "
          f"{time.monotonic() - t:.0f}s", flush=True)

    # Warm the exact shapes both arms will use, so the first arm measured
    # does not pay one-time kernel selection for the others.
    forward, _ = make_predictor_step(torch, model, device)
    with torch.inference_mode():
        probe = model(pixel_values_videos=clips[sessions[0]].to(device)
                      ).last_hidden_state
        n_tokens = int(probe.shape[1])
        forward(probe)
        forward(torch.cat([probe] * SESSIONS, dim=0))
    torch.cuda.synchronize()
    del probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"[warmup] context tokens={n_tokens} "
          f"tail={max(1, n_tokens // FRAMES)} "
          f"fused batch={SESSIONS * BRANCH}", flush=True)

    arms, isolation, parity = _run_all_arms(
        torch, sessions, graph, walkset, placement, model, clips, device,
        width=SESSIONS)

    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    reserved_gb = torch.cuda.max_memory_reserved() / 2 ** 30
    parity_class = classify_parity(parity, PARITY_BUDGET)
    speedup = (arms["batched"]["steps_per_s"]
               / arms["solo"]["steps_per_s"])
    isolation_ok = isolation["max_abs_over_sessions"] == 0.0

    if parity_class == "exact":
        batching_status, batching_reason = "accepted", (
            "fused K sessions' steps into one call and every session's "
            "output stayed bit-identical to running alone")
    elif parity_class == "bounded" and speedup > 1.0:
        batching_status, batching_reason = "accepted", (
            f"parity bounded within the declared budget "
            f"(rel_l2 {parity['max_rel_l2']:.2e} <= "
            f"{PARITY_BUDGET['rel_l2']:.0e}, rel_max_abs "
            f"{parity['max_rel_max_abs']:.2e} <= "
            f"{PARITY_BUDGET['rel_max_abs']:.0e}); bf16 reduction order "
            f"changes with batch size, the per-sample function does not")
    elif parity_class == "bounded":
        batching_status, batching_reason = "rejected", (
            f"parity is within budget but throughput did not improve "
            f"({arms['batched']['steps_per_s']} vs "
            f"{arms['solo']['steps_per_s']} steps/s, {speedup:.2f}x); "
            f"paying a numerics change for no speedup is not a win")
    else:
        batching_status, batching_reason = "rejected", (
            f"parity over the declared budget: rel_l2 "
            f"{parity['max_rel_l2']:.2e} > {PARITY_BUDGET['rel_l2']:.0e} "
            f"or rel_max_abs {parity['max_rel_max_abs']:.2e} > "
            f"{PARITY_BUDGET['rel_max_abs']:.0e} on session "
            f"{parity['worst_session']}; a scheduling decision moved "
            f"results further than bf16 batch-size noise explains")

    summary = {
        "model": "vjepa2-vitl-fpc64-256 (local)",
        "runtime": "wllm-composite",
        "task": "K concurrent world-model rollout sessions, one GPU",
        "config": {"sessions": SESSIONS, "branch": BRANCH, "steps": STEPS,
                   "frames": FRAMES, "size": SIZE, "tokens": n_tokens,
                   "fused_batch": SESSIONS * BRANCH, "load_s": round(load_s, 1)},
        "graph": {
            "components": [c.id for c in graph.components],
            "edges": [[e.source, e.target] for e in graph.edges],
            "walks": sorted(walkset.walks),
            "placement": placement,
        },
        "arms": arms,
        "isolation": {
            "verdict": "isolated" if isolation_ok else "LEAK",
            "requirement": "bit-identical outputs, concurrent vs alone",
            **{k: v for k, v in isolation.items() if k != "per_session"},
            "per_session": isolation["per_session"],
        },
        "batching": {
            "parity_class": parity_class,
            "budget": PARITY_BUDGET,
            "status": batching_status,
            "reason": batching_reason,
            "speedup_vs_sequential": round(speedup, 3),
            "sequential_steps_per_s": arms["solo"]["steps_per_s"],
            "batched_steps_per_s": arms["batched"]["steps_per_s"],
            "concurrent_unbatched_steps_per_s":
                arms["concurrent"]["steps_per_s"],
            **{k: v for k, v in parity.items() if k != "per_session"},
            "per_session": parity["per_session"],
        },
        "peak_vram_gb": round(peak_gb, 2),
        "peak_vram_reserved_gb": round(reserved_gb, 2),
        "vram_floor_gb": VRAM_FLOOR_GB,
        "vram_floor_ok": peak_gb >= VRAM_FLOOR_GB,
        "preflight": {"isolation_max_abs": pre_iso["max_abs_over_sessions"],
                      "parity_class": classify_parity(pre_par,
                                                      PARITY_BUDGET)},
    }

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"composite_rollout_{stamp}.json"
    path.write_text(json.dumps(summary, indent=1))

    print(flush=True)
    print(f"[isolation] {summary['isolation']['verdict']}: "
          f"max |concurrent - alone| over {isolation['sessions']} sessions "
          f"= {isolation['max_abs_over_sessions']:.6e} "
          f"(worst {isolation['worst_session']})", flush=True)
    if not isolation_ok:
        print("ISOLATION FAILURE: running sessions concurrently through one "
              "executor changed their outputs. This is a runtime defect "
              "(state crossing session boundaries), not benchmark noise.",
              flush=True)
    print(f"[throughput] sequential {arms['solo']['steps_per_s']} steps/s "
          f"| concurrent-unbatched {arms['concurrent']['steps_per_s']} "
          f"| batched {arms['batched']['steps_per_s']} "
          f"({speedup:.2f}x, fused call = "
          f"{arms['batched']['batcher_max_group']} requests)", flush=True)
    print(f"[parity] {parity_class} -> {batching_status}: {batching_reason}",
          flush=True)
    print(f"[vram] peak allocated {peak_gb:.2f} GB "
          f"(reserved {reserved_gb:.2f} GB, floor {VRAM_FLOOR_GB} GB, "
          f"{'OK' if summary['vram_floor_ok'] else 'LOW'})", flush=True)
    print(json.dumps(summary), flush=True)
    print(f"[results] {path}", flush=True)

    if not isolation_ok:
        print("COMPOSITE_ROLLOUT_FAIL: session isolation not measured exact",
              flush=True)
        return 1
    print("COMPOSITE_ROLLOUT_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:      # noqa: BLE001 — explicit failure marker
        print(f"COMPOSITE_ROLLOUT_FAIL: {type(exc).__name__}: {exc}",
              flush=True)
        raise SystemExit(1)
