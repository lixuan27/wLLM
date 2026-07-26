"""Named walk sets, request state machines, and chunk policies."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.composite import (
    ChunkedChannel, Component, ComponentGraph, Edge, FixedChunk,
    LeftContext, Loop, Seq, SlidingWindow, Walk, WalkExecutor, WalkSet,
    run_request,
)
from wllm.graph.regions import NodeOp
from wllm.graph.streams import Backpressure


def _graph() -> ComponentGraph:
    return ComponentGraph(
        name="composite",
        components=[
            Component("encoder", NodeOp.ENCODER),
            Component("reasoner", NodeOp.TRANSFORMER),
            Component("generator", NodeOp.TRANSFORMER),
            Component("decoder", NodeOp.CODEC),
        ],
        edges=[
            Edge("encoder", "reasoner"),
            Edge("reasoner", "generator"),
            Edge("generator", "decoder"),
        ])


def _impls():
    def encoder(ctx, state):
        return {"emb": "emb"}

    def reasoner(ctx, state):
        return {"plan": f"plan({ctx['emb']})"}

    def generator(ctx, state):
        return {"latent": ctx.get("latent", 0) + 1}

    def decoder(ctx, state):
        return {"frames": ctx["latent"] * 10}

    return {"encoder": encoder, "reasoner": reasoner,
            "generator": generator, "decoder": decoder}


def _walkset() -> WalkSet:
    return WalkSet(walks={
        "understand": Walk([Seq("encoder"), Seq("reasoner")]),
        "generate": Walk([Loop(Walk([Seq("generator")]),
                               carry="latent", iterations=3)]),
        "decode_only": Walk([Seq("decoder")]),
    })


def _chooser(request_ctx, last_walk, last_output):
    if last_walk is None:
        return "understand"
    if request_ctx["task"] == "plan":
        return None                # planning stops after understanding
    if last_walk == "understand":
        return "generate"
    if last_walk == "generate" and last_output["latent"] >= 3:
        return "decode_only"
    return None


# ------------------------------------------------------ walk state machine

def test_request_runs_series_of_walks():
    ex = WalkExecutor(_graph(), _impls())
    res = run_request(ex, _walkset(), _chooser, session="r1",
                      ctx={"task": "generate"})
    assert res.walk_trail == ["understand", "generate", "decode_only"]
    assert res.outputs["plan"] == "plan(emb)"
    assert res.outputs["latent"] == 3
    assert res.outputs["frames"] == 30
    # routing intent stays out of the execution context
    assert "task" not in res.outputs
    assert res.ctx == {"task": "generate"}


def test_minimum_components_property():
    ex = WalkExecutor(_graph(), _impls())
    res = run_request(ex, _walkset(), _chooser, session="r2",
                      ctx={"task": "plan"})
    assert res.walk_trail == ["understand"]
    invoked = {inv.component for inv in ex.invocations}
    # only the chosen walk's components ever ran
    assert invoked == {"encoder", "reasoner"}
    assert "generator" not in invoked and "decoder" not in invoked
    assert "frames" not in res.outputs


def test_chooser_sees_copies_not_live_state():
    def evil(request_ctx, last_walk, last_output):
        request_ctx["task"] = "hacked"     # must not stick
        last_output["latent"] = 999        # must not leak into walks
        return "understand" if last_walk is None else None

    ex = WalkExecutor(_graph(), _impls())
    res = run_request(ex, _walkset(), evil, "r3", {"task": "generate"})
    assert res.ctx == {"task": "generate"}
    assert "latent" not in res.outputs


def test_unknown_walk_name_fails_closed():
    ex = WalkExecutor(_graph(), _impls())
    try:
        run_request(ex, _walkset(), lambda rc, lw, lo: "mystery", "r4")
    except KeyError as exc:
        msg = str(exc)
        assert "mystery" in msg
        assert "decode_only" in msg and "understand" in msg
    else:
        raise AssertionError("unknown walk name must be rejected")


def test_non_terminating_machine_fails_closed():
    ex = WalkExecutor(_graph(), _impls())
    try:
        run_request(ex, _walkset(), lambda rc, lw, lo: "understand",
                    "r5", max_walks=5)
    except RuntimeError as exc:
        assert "did not terminate" in str(exc)
    else:
        raise AssertionError("runaway state machine must be rejected")
    # exactly max_walks walks ran before the refusal (2 components each)
    assert len(ex.invocations) == 10


def test_walkset_validation():
    g = _graph()
    assert _walkset().validate(g) == []
    empty = WalkSet(walks={})
    assert any("empty" in e for e in empty.validate(g))
    unnamed = WalkSet(walks={"": Walk([Seq("encoder")])})
    assert any("non-empty" in e for e in unnamed.validate(g))
    broken = WalkSet(walks={"w": Walk([Seq("ghost")])})
    errs = broken.validate(g)
    assert any("walk 'w'" in e and "unknown components" in e
               for e in errs)
    ex = WalkExecutor(g, _impls())
    try:
        run_request(ex, empty, lambda rc, lw, lo: None, "r6")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty walk set must be rejected")


# ----------------------------------------------------------- chunk policies

def test_fixed_chunk_sequence_and_flush():
    ch = ChunkedChannel(FixedChunk(3), capacity=8,
                        backpressure=Backpressure.BLOCK)
    fired = []
    for i in range(1, 8):
        fired.extend(ch.put(i))
    assert fired == [[1, 2, 3], [4, 5, 6]]
    assert ch.flush() == [7]
    assert ch.flush() is None          # tail delivered exactly once
    assert ch.items_in == 7 and ch.chunks_out == 3
    assert ch.dropped == 0 and ch.rejected == 0


def test_sliding_window_sequence_and_flush():
    ch = ChunkedChannel(SlidingWindow(window=4, stride=2), capacity=8,
                        backpressure=Backpressure.BLOCK)
    fired = []
    for i in range(1, 7):
        fired.extend(ch.put(i))
    assert fired == [[1, 2, 3, 4], [3, 4, 5, 6]]
    assert ch.flush() is None          # every item already delivered
    ch2 = ChunkedChannel(SlidingWindow(window=4, stride=2), capacity=8,
                         backpressure=Backpressure.BLOCK)
    fired2 = []
    for i in range(1, 8):
        fired2.extend(ch2.put(i))
    assert fired2 == [[1, 2, 3, 4], [3, 4, 5, 6]]
    assert ch2.flush() == [5, 6, 7]    # final partial window + context
    ch3 = ChunkedChannel(SlidingWindow(window=4, stride=2), capacity=8,
                         backpressure=Backpressure.BLOCK)
    ch3.put(1)
    ch3.put(2)
    assert ch3.flush() == [1, 2]       # never fired -> all undelivered


def test_left_context_sequence_and_flush():
    ch = ChunkedChannel(LeftContext(chunk=2, left=1), capacity=8,
                        backpressure=Backpressure.BLOCK)
    fired = []
    for i in range(1, 6):
        fired.extend(ch.put(i))
    assert fired == [[1, 2], [2, 3, 4]]   # first chunk has no left yet
    assert ch.flush() == [4, 5]           # tail keeps its left context
    assert ch.flush() is None
    assert ch.items_in == 5 and ch.chunks_out == 3


def test_chunk_policy_parameter_rejection():
    bad = [lambda: FixedChunk(0),
           lambda: SlidingWindow(window=0, stride=1),
           lambda: SlidingWindow(window=4, stride=0),
           lambda: SlidingWindow(window=4, stride=5),  # skips items
           lambda: LeftContext(chunk=0, left=1),
           lambda: LeftContext(chunk=2, left=-1)]
    for mk in bad:
        try:
            mk()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk policy must be "
                                 "rejected")
    try:
        ChunkedChannel(FixedChunk(2), capacity=0,
                       backpressure=Backpressure.BLOCK)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity < 1 must be rejected")


def test_raw_buffer_backpressure_reject_and_block():
    # capacity below the fill requirement: the channel can never fire,
    # and REJECT refuses (and counts) every overflowing item
    ch = ChunkedChannel(FixedChunk(5), capacity=3,
                        backpressure=Backpressure.REJECT)
    fired = []
    for i in range(1, 7):
        fired.extend(ch.put(i))
    assert fired == []
    assert ch.rejected == 3 and ch.items_in == 3 and ch.chunks_out == 0
    assert ch.flush() == [1, 2, 3]     # bounded tail still delivered
    ch_block = ChunkedChannel(FixedChunk(5), capacity=2,
                              backpressure=Backpressure.BLOCK)
    ch_block.put("a")
    ch_block.put("b")
    try:
        ch_block.put("c")
    except BufferError:
        pass
    else:
        raise AssertionError("block overflow must raise, not drop")


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
