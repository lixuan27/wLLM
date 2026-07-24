"""Composite graph runtime: structure, walks, isolation, batching parity."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.composite import (
    Component, ComponentGraph, Edge, Loop, Par, Seq, SessionStore, Stream,
    StepBatcher, Walk, WalkExecutor,
)
from wllm.composite.batching import StepRequest
from wllm.graph.regions import NodeOp
from wllm.graph.states import StateKind, StateSpec
from wllm.graph.streams import Backpressure, Modality, StreamSpec


def _graph() -> ComponentGraph:
    return ComponentGraph(
        name="demo",
        components=[
            Component("text_encoder", NodeOp.ENCODER),
            Component("dit", NodeOp.TRANSFORMER, batchable=True,
                      states=[StateSpec(id="latent", kind=StateKind.KV)]),
            Component("vae", NodeOp.CODEC),
            Component("action_head", NodeOp.POLICY_HEAD,
                      placement_domain="fixed:gpu0"),
        ],
        edges=[
            Edge("text_encoder", "dit"),
            Edge("dit", "vae", stream=StreamSpec(
                id="latents", modality=Modality.LATENT, producer="dit",
                consumer="vae", bounded_queue=2,
                backpressure=Backpressure.REJECT)),
            Edge("vae", "action_head"),
        ])


def _impls():
    def text_encoder(ctx, state):
        return {"emb": f"emb({ctx['prompt']})"}

    def dit(ctx, state):
        state["calls"] = state.get("calls", 0) + 1
        return {"latent": ctx.get("latent", 0) + 1, "state_calls": state["calls"]}

    def vae(ctx, state):
        return {"frames": ctx["latent"] * 10}

    def action_head(ctx, state):
        return {"action": [0.1, 0.2]}

    return {"text_encoder": text_encoder, "dit": dit, "vae": vae,
            "action_head": action_head}


# ------------------------------------------------------------------ graph

def test_graph_validation_catches_structural_errors():
    g = _graph()
    assert g.validate() == []
    g.components.append(Component("dit", NodeOp.TRANSFORMER))
    g.edges.append(Edge("ghost", "dit"))
    g.components[0].states.append(StateSpec(id="latent", kind=StateKind.KV))
    errs = g.validate()
    joined = " ".join(errs)
    assert "duplicate component ids" in joined
    assert "unknown component 'ghost'" in joined
    assert "owned by both" in joined


def test_state_owner_lookup():
    g = _graph()
    assert g.state_owner("latent") == "dit"
    assert g.state_owner("nope") is None


# ------------------------------------------------------------------- walks

def test_walk_validation():
    g = _graph()
    ok = Walk([Seq("text_encoder"),
               Loop(Walk([Seq("dit")]), carry="latent", iterations=3),
               Stream("dit", "vae")])
    assert ok.validate(g) == []
    bad = Walk([Seq("mystery"),
                Loop(Walk([Seq("dit")]), carry="x"),
                Stream("text_encoder", "vae")])
    errs = bad.validate(g)
    joined = " ".join(errs)
    assert "unknown components" in joined
    assert "exactly one of iterations/until" in joined
    assert "no stream edge" in joined


def test_fixed_loop_and_until_loop():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls)
    out = ex.run(Walk([Seq("text_encoder"),
                       Loop(Walk([Seq("dit")]), carry="latent", iterations=5),
                       Seq("vae")]), session="s1", ctx={"prompt": "hi"})
    assert out["latent"] == 5 and out["frames"] == 50
    assert out["loop_iterations_run"] == 5

    def dit_until(ctx, state):
        nxt = ctx.get("latent", 0) + 1
        return {"latent": nxt, "done": nxt >= 3}
    ex2 = WalkExecutor(g, {**impls, "dit": dit_until})
    out2 = ex2.run(Walk([Loop(Walk([Seq("dit")]), carry="latent",
                              until="done", max_iterations=10)]), "s1")
    assert out2["latent"] == 3 and out2["loop_iterations_run"] == 3

    ex3 = WalkExecutor(g, impls)   # never sets 'done'
    try:
        ex3.run(Walk([Loop(Walk([Seq("dit")]), carry="latent",
                           until="done", max_iterations=4)]), "s1")
    except RuntimeError as exc:
        assert "max_iterations" in str(exc)
    else:
        raise AssertionError("runaway until-loop must fail closed")


def test_parallel_branches_merge_and_conflict():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls, placement={"action_head": "gpu0"})
    out = ex.run(Walk([Seq("text_encoder"),
                       Par([Walk([Seq("dit")]), Walk([Seq("action_head")])])]),
                 "s1", {"prompt": "p", "latent": 0})
    assert out["latent"] == 1 and out["action"] == [0.1, 0.2]
    try:
        ex.run(Walk([Par([Walk([Seq("dit")]), Walk([Seq("dit")])])]),
               "s1", {"latent": 0})
    except ValueError as exc:
        assert "both produced key" in str(exc)
    else:
        raise AssertionError("conflicting parallel writes must be rejected")


# --------------------------------------------------------------- isolation

def test_session_state_isolation_and_reset():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls)
    walk = Walk([Loop(Walk([Seq("dit")]), carry="latent", iterations=2)])
    a = ex.run(walk, session="A")
    b = ex.run(walk, session="B")
    assert a["state_calls"] == 2 and b["state_calls"] == 2  # B saw no A state
    a2 = ex.run(walk, session="A")
    assert a2["state_calls"] == 4                            # A accumulated
    ex.store.reset("A")
    a3 = ex.run(walk, session="A")
    assert a3["state_calls"] == 2                            # provably cleared
    assert "A" in ex.store.sessions() and "B" in ex.store.sessions()


def test_placement_recorded_and_pins_enforced():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls, placement={"dit": "gpu1",
                                           "action_head": "gpu0"})
    ex.run(Walk([Seq("dit"), Seq("action_head")]), "s1", {"latent": 0})
    used = ex.devices_used()
    assert used["dit"] == {"gpu1"} and used["action_head"] == {"gpu0"}
    ex_bad = WalkExecutor(g, impls, placement={"action_head": "gpu3"})
    try:
        ex_bad.run(Walk([Seq("action_head")]), "s1")
    except ValueError as exc:
        assert "pinned" in str(exc)
    else:
        raise AssertionError("placement pin violation must be rejected")


def test_stream_backpressure_reject_counts():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls)
    walk = Walk([Stream("dit", "vae")])
    for i in range(4):
        ex.run(walk, "s1", {"stream_item": i})
    chan = ex.channel("s1", "dit", "vae")
    assert chan.rejected == 2 and len(chan.drain()) == 2   # capacity 2, reject


def test_stream_channels_are_session_isolated_and_reset():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls)
    walk = Walk([Stream("dit", "vae")])
    ex.run(walk, "A", {"stream_item": "frameA"})
    ex.run(walk, "B", {"stream_item": "frameB"})
    # B never sees A's items, and A's overflow can never evict B's
    assert ex.channel("A", "dit", "vae").drain() == ["frameA"]
    assert ex.channel("B", "dit", "vae").drain() == ["frameB"]
    ex.run(walk, "A", {"stream_item": "stale"})
    ex.reset_session("A")
    # reset clears A's channels along with its state; B untouched
    assert ex.channel("A", "dit", "vae").drain() == []
    ex.run(walk, "B", {"stream_item": "b2"})
    assert ex.channel("B", "dit", "vae").drain() == ["b2"]


def test_par_unknown_join_and_inplace_safety():
    g, impls = _graph(), _impls()
    ex = WalkExecutor(g, impls, placement={"action_head": "gpu0"})
    try:
        ex.run(Walk([Par([Walk([Seq("dit")])], join="concat")]),
               "s1", {"latent": 0})
    except NotImplementedError as exc:
        assert "join" in str(exc)
    else:
        raise AssertionError("unknown join mode must be rejected")
    # a Stream inside a Par branch must not mutate the parent ctx list
    parent_ctx = {"stream_item": "x", "stream_accepted": [True]}
    ex.run(Walk([Par([Walk([Stream("dit", "vae")]),
                      Walk([Seq("action_head")])])]), "s1", parent_ctx)
    assert parent_ctx["stream_accepted"] == [True]   # parent list untouched


def test_nested_and_sequential_loops_keep_clean_state():
    g, impls = _graph(), _impls()

    def dit_until(ctx, state):
        nxt = ctx.get("latent", 0) + 1
        return {"latent": nxt, "done": nxt >= 2}

    ex = WalkExecutor(g, {**impls, "dit": dit_until})
    # two until-loops in sequence: the second must NOT exit early on the
    # first loop's stale flag
    two = Walk([
        Loop(Walk([Seq("dit")]), carry="latent", until="done",
             max_iterations=10),
        Loop(Walk([Seq("dit")]), carry="latent", until="done",
             max_iterations=10),
    ])
    out = ex.run(two, "s1")
    assert out["latent"] == 3   # loop1: 0->2 (2 iters); loop2: one more
    # nested fixed loops: inner must not clobber the outer loop_index
    seen = []

    def probe(ctx, state):
        seen.append(ctx["loop_index"])
        return {}

    g2 = ComponentGraph("g2", [Component("probe", NodeOp.PROBE),
                               Component("inner", NodeOp.CUSTOM)])
    ex2 = WalkExecutor(g2, {"probe": probe, "inner": lambda c, s: {}})
    ex2.run(Walk([Loop(Walk([
        Loop(Walk([Seq("inner")]), carry="x", iterations=3),
        Seq("probe"),                      # reads OUTER loop_index
    ]), carry="x", iterations=2)]), "s1")
    assert seen == [0, 1]


# ---------------------------------------------------------------- batching

def test_step_batching_parity_and_grouping():
    def batched_dit(payloads):
        return [p * 2 for p in payloads]

    b = StepBatcher({"dit": batched_dit}, max_batch=3)
    reqs = [StepRequest(f"r{i}", "dit", i, signature=("f32", i % 2))
            for i in range(7)]
    got = b.run(reqs)
    # parity: identical to sequential application
    assert got == {f"r{i}": i * 2 for i in range(7)}
    # grouping: same-signature only, capped at max_batch
    assert b.max_group_size() <= 3
    for rec in b.records:
        assert len(set(rec.member_signatures)) == 1
    assert b.cross_signature_mixes() == 0
    # the mix detector is falsifiable: a corrupted record trips it
    b.records[0].member_signatures = [("f32", 0), ("f32", 1)]
    assert b.cross_signature_mixes() == 1


def test_step_batching_fail_closed():
    b = StepBatcher({"dit": lambda ps: ps[:-1]})   # wrong length
    try:
        b.run([StepRequest("a", "dit", 1), StepRequest("b", "dit", 2)])
    except RuntimeError as exc:
        assert "refusing to guess" in str(exc)
    else:
        raise AssertionError("length mismatch must fail closed")
    b2 = StepBatcher({"dit": lambda ps: ps})
    try:
        b2.run([StepRequest("a", "dit", 1), StepRequest("a", "dit", 2)])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate ids must be rejected")
    try:
        b2.run([StepRequest("a", "mystery", 1)])
    except KeyError:
        pass
    else:
        raise AssertionError("unknown component must be rejected")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:200]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
