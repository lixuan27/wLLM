"""M1 toy end-to-end: program -> rules -> constraints -> cost model ->
reference execution -> numerical verification.

The toy pipeline mimics a chunked video generator: a "dit" node whose KV
state accumulates across chunks, and a "vae" node with its own recurrent
cache.  The two nodes share no state, so the rules must discover both the
co-located family (latency) and the disaggregated family (throughput), and
the analytic cost model must reproduce the fundamental tradeoff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wllm.graph import (
    Modality, Node, NodeOp, Program, QualityContract, Region, RegionKind,
    StateKind, StateSpec, StreamSpec,
)
from wllm.planner.constraints import filter_plans
from wllm.planner.plan import Hardware
from wllm.planner.rules import generate_candidates
from wllm.runtime.reference_executor import ImplRegistry, ReferenceExecutor
from wllm.verify.numerical import compare

GB = 10 ** 9


def build_program() -> Program:
    root = Region(
        id="app", kind=RegionKind.CHUNK_ROLLOUT, attrs={"chunk_size": 4},
        nodes=[
            Node(id="dit", op=NodeOp.TRANSFORMER, reads=["kv"], writes=["kv"],
                 cost_hint_ms=60.0),
            Node(id="vae", op=NodeOp.CODEC, reads=["vcache"],
                 writes=["vcache", "frames_total"], cost_hint_ms=30.0),
        ])
    states = {
        "kv": StateSpec(id="kv", kind=StateKind.KV, ordered=True,
                        migratable=False, memory_bytes=4 * GB,
                        verified=True, evidence="toy://probe/kv"),
        "vcache": StateSpec(id="vcache", kind=StateKind.RECURRENT,
                            ordered=True, migratable=False,
                            memory_bytes=1 * GB,
                            verified=True, evidence="toy://probe/vcache"),
        "frames_total": StateSpec(id="frames_total",
                                  kind=StateKind.RECOMPUTABLE_FEATURE,
                                  ordered=False, recomputable=True,
                                  verified=True, evidence="toy://probe/ft"),
    }
    streams = {
        "latents": StreamSpec(id="latents", modality=Modality.LATENT,
                              producer="dit", consumer="vae", chunk_size=1),
    }
    return Program(name="toy_chunk_video", root=root, states=states,
                   streams=streams, quality=QualityContract())


def make_registry() -> ImplRegistry:
    reg = ImplRegistry()

    def dit(data, state, ctx):
        kv = state.get("kv") or []
        kv = kv + [ctx["chunk_index"]]
        state.set("kv", kv)
        return {"latent": float(sum(kv)) + data.get("cond", 0.0)}

    def vae(data, state, ctx):
        cache = state.get("vcache") or 0.0
        frame = data["latent"] * 0.5 + cache * 0.25
        state.set("vcache", frame)
        total = (state.get("frames_total") or 0.0) + frame
        state.set("frames_total", total)
        return {"frame": frame}

    reg.register("dit", dit)
    reg.register("vae", vae)
    return reg


def run_reference(num_chunks=3):
    ex = ReferenceExecutor(build_program(), make_registry())
    out = ex.run({"cond": 1.0}, num_chunks=num_chunks)
    return out, dict(ex.state)


def test_rules_discover_both_families():
    plans = generate_candidates(build_program(), Hardware(4, 141 * GB))
    ids = {p.id for p in plans}
    assert "baseline_1gpu" in ids
    assert any(i.startswith("colocated_deg") for i in ids), ids
    assert any(i.startswith("disagg_") for i in ids), ids
    assert len(plans) >= 4


def test_cost_model_reproduces_tradeoff():
    prog = build_program()
    hw = Hardware(4, 141 * GB)
    plans = {p.id: p for p in generate_candidates(prog, hw)}
    costs = {n.id: n.cost_hint_ms for n in prog.root.iter_nodes()}

    lat_base, per_base = plans["baseline_1gpu"].estimate(costs)
    assert lat_base == 90.0 and per_base == 90.0

    colocated = next(p for i, p in plans.items() if i.startswith("colocated"))
    lat_co, per_co = colocated.estimate(costs)
    disagg = plans["disagg_2stage"]
    lat_dis, per_dis = disagg.estimate(costs, transfer_ms=2.0)

    # co-location wins latency; disaggregation wins sustainable period
    assert lat_co < lat_dis, (lat_co, lat_dis)
    assert per_dis < per_base, (per_dis, per_base)
    assert per_dis == 60.0  # max(dit, vae) instead of sum


def test_constraints_reject_oom_and_split_state():
    prog = build_program()
    hw_small = Hardware(4, 3 * GB)  # kv alone (4GB) cannot fit
    plans = generate_candidates(prog, hw_small)
    kept, rejected = filter_plans(plans, prog, hw_small)
    assert not kept
    assert all("OOM" in r for _, r in rejected)

    hw_ok = Hardware(4, 141 * GB)
    kept, rejected = filter_plans(generate_candidates(prog, hw_ok), prog, hw_ok)
    assert {p.id for p in kept} >= {"baseline_1gpu", "disagg_2stage"}
    assert not rejected


def test_reference_executor_and_verifier():
    out1, state1 = run_reference()
    out2, state2 = run_reference()
    res = compare({"out": out1, "state": state1},
                  {"out": out2, "state": state2},
                  QualityContract())
    assert res.passed, res.report()

    # deterministic values: kv=[0,1,2] -> latent=sum(kv)+cond=4;
    # frame chain: 0.5 -> 1.125 -> 4*0.5+1.125*0.25 = 2.28125
    assert out1["latent"] == 4.0
    assert abs(state1["vcache"] - 2.28125) < 1e-9, state1["vcache"]

    broken = dict(out2)
    broken["latent"] = out2["latent"] + 1.0
    res_bad = compare({"out": out1}, {"out": broken}, QualityContract())
    assert not res_bad.passed


def test_state_access_enforced():
    prog = build_program()
    reg = make_registry()

    def rogue(data, state, ctx):
        state.set("kv", [])  # undeclared write for node 'vae' -> error
        return {}

    reg.register("vae", rogue)
    ex = ReferenceExecutor(prog, reg)
    try:
        ex.run({"cond": 0.0}, num_chunks=1)
        raise AssertionError("expected StateAccessError")
    except Exception as exc:  # noqa: BLE001
        assert "undeclared state" in str(exc)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
