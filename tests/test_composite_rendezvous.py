"""Runtime session binding and the concurrent-walk step gate.

These cover the two pieces the composite runtime needed before it could
drive real concurrent sessions: ``current_session`` (component code that
must key an external resource by request) and ``StepGate`` (K blocked
walks -> one batched call -> results routed back per request). Everything
here is synthetic and CPU-only; the GPU counterpart is
``benchmarks/composite_rollout_vjepa2.py``.

Round size is ``min(width, live)``, so a test that wants a round of N
either registers exactly N participants up front or registers none and
lets ``width`` decide — registering participants from inside the worker
threads would race the first submit and is never done here.
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.composite import (Component, ComponentGraph, Edge, Loop, Seq,
                            StepBatcher, StepGate, Walk, WalkExecutor,
                            WalkSet, current_session, run_request)
from wllm.graph.regions import NodeOp
from wllm.graph.states import StateKind, StateSpec


# --------------------------------------------------------------- helpers
def _graph() -> ComponentGraph:
    return ComponentGraph(
        name="rollout",
        components=[
            Component("encoder", NodeOp.ENCODER,
                      states=[StateSpec(
                          id="ctx", kind=StateKind.RECOMPUTABLE_FEATURE)]),
            Component("stepper", NodeOp.TRANSFORMER, batchable=True,
                      states=[StateSpec(
                          id="trace", kind=StateKind.ROLLING_CONTEXT)]),
        ],
        edges=[Edge("encoder", "stepper")])


def _walkset(steps: int) -> WalkSet:
    return WalkSet(walks={
        "ground": Walk([Seq("encoder")]),
        "rollout": Walk([Loop(body=Walk([Seq("stepper")]), carry="latent",
                              iterations=steps)]),
    })


def _chooser(request_ctx, last_walk, last_output):
    if last_walk is None:
        return "ground"
    if last_walk == "ground":
        return "rollout"
    return None


def _run_threads(fn, args_list, timeout=30.0):
    """Run fn(arg) on one thread each; return {arg: failure message}."""
    errors = {}

    def guarded(arg):
        def body():
            try:
                fn(arg)
            except BaseException as exc:      # noqa: BLE001 — reported below
                errors[arg] = f"{type(exc).__name__}: {exc}"
        return body

    threads = [threading.Thread(target=guarded(a)) for a in args_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        if t.is_alive():
            raise AssertionError("a worker never finished; the gate hung")
    return errors


class _StubBatcher:
    """Batcher stand-in so pathological returns are testable."""

    def __init__(self, fn):
        self.fn = fn

    def run(self, requests):
        return self.fn(requests)


# -------------------------------------------------------- current_session
def test_current_session_visible_to_components():
    seen = []

    def encoder(ctx, state):
        seen.append(current_session())
        return {"latent": 1}

    ex = WalkExecutor(_graph(), {"encoder": encoder,
                                 "stepper": lambda c, s: {}})
    ex.run(Walk([Seq("encoder")]), "alice")
    ex.run(Walk([Seq("encoder")]), "bob")
    assert seen == ["alice", "bob"], seen


def test_current_session_outside_a_walk_raises():
    try:
        current_session()
    except RuntimeError as exc:
        assert "outside a walk" in str(exc)
    else:
        raise AssertionError("there is no default session to guess")


def test_current_session_is_restored_after_a_nested_run():
    holder = {}
    seen = []

    def encoder(ctx, state):
        seen.append(("outer-before", current_session()))
        holder["ex"].run(Walk([Seq("stepper")]), "inner")
        seen.append(("outer-after", current_session()))
        return {"latent": 1}

    def stepper(ctx, state):
        seen.append(("inner", current_session()))
        return {}

    ex = WalkExecutor(_graph(), {"encoder": encoder, "stepper": stepper})
    holder["ex"] = ex
    ex.run(Walk([Seq("encoder")]), "outer")
    assert seen == [("outer-before", "outer"), ("inner", "inner"),
                    ("outer-after", "outer")], seen
    try:
        current_session()
    except RuntimeError:
        pass
    else:
        raise AssertionError("binding must not survive the outermost run")


def test_current_session_is_per_thread():
    seen = {}
    overlap = threading.Barrier(4)

    def encoder(ctx, state):
        overlap.wait(timeout=20)          # force real overlap
        seen[current_session()] = threading.current_thread().name
        return {"latent": 1}

    ex = WalkExecutor(_graph(), {"encoder": encoder,
                                 "stepper": lambda c, s: {}})
    sessions = [f"s{i}" for i in range(4)]
    errs = _run_threads(lambda s: ex.run(Walk([Seq("encoder")]), s), sessions)
    assert not errs, errs
    assert sorted(seen) == sessions, seen
    assert len(set(seen.values())) == 4, seen


# ---------------------------------------------------------------- gate
def test_gate_fuses_concurrent_submissions_into_one_call():
    calls = []

    def batched(payloads):
        calls.append(len(payloads))
        time.sleep(0.01)
        return [p * 10 for p in payloads]

    batcher = StepBatcher({"stepper": batched}, max_batch=4)
    gate = StepGate(batcher, width=4, wait_s=20.0, timeout_s=40.0)
    got = {}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i, ("f32",))

    assert not _run_threads(submit, list(range(4)))
    assert got == {i: i * 10 for i in range(4)}, got
    assert calls == [4], calls
    assert batcher.max_group_size() == 4
    stats = gate.stats()
    assert (stats.rounds, stats.fused_rounds, stats.max_round) == (1, 1, 4)
    assert stats.partial_rounds == 0
    assert stats.submissions == 4


def test_gate_width_one_never_fuses_a_call():
    batcher = StepBatcher({"stepper": lambda ps: [p * 10 for p in ps]},
                          max_batch=1)
    gate = StepGate(batcher, width=1, wait_s=10.0, timeout_s=30.0)
    got = {}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i, ("f32",))

    assert not _run_threads(submit, list(range(6)))
    assert got == {i: i * 10 for i in range(6)}, got
    # a round may sweep up several tickets, but max_batch=1 means every
    # actual CALL carried exactly one request — that is the claim that
    # matters, and the batcher, not the gate, is its witness
    assert batcher.max_group_size() == 1
    assert len(batcher.records) == 6


def test_gate_routes_each_result_to_its_own_request():
    def batched(payloads):
        # order-sensitive on purpose: a mis-route shows up as a swap
        return [f"out:{p}" for p in payloads]

    gate = StepGate(StepBatcher({"stepper": batched}, max_batch=8), width=8,
                    wait_s=20.0, timeout_s=40.0)
    got = {}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", f"p{i}", ("f32",))

    assert not _run_threads(submit, list(range(8)))
    assert got == {i: f"out:p{i}" for i in range(8)}, got


def test_gate_rejects_a_second_step_for_the_same_request():
    entered, release = threading.Event(), threading.Event()

    def batched(payloads):
        entered.set()
        release.wait(timeout=20)
        return [p * 10 for p in payloads]

    gate = StepGate(StepBatcher({"stepper": batched}, max_batch=1), width=1,
                    wait_s=5.0, timeout_s=30.0)
    first = {}
    worker = threading.Thread(
        target=lambda: first.update(value=gate.submit("same", "stepper", 1)))
    worker.start()
    assert entered.wait(timeout=20), "the first round never started"
    try:
        gate.submit("same", "stepper", 2)
        duplicated = True
    except ValueError as exc:
        duplicated = False
        assert "already has a step in flight" in str(exc)
    finally:
        release.set()
        worker.join(20)
    assert not worker.is_alive()
    assert duplicated is False, "two steps under one request id must be refused"
    assert first["value"] == 10


def test_a_leader_of_someone_elses_round_still_gets_its_own_result():
    """Rounds are capped at `width`, so tickets pile up behind one.

    A caller that becomes leader for a round it is not a member of must
    keep waiting for its own ticket. Returning after leading a round
    would hand that caller ``None`` — a wrong answer, silently.
    """
    first_in, release = threading.Event(), threading.Event()

    def batched(payloads):
        if not first_in.is_set():
            first_in.set()
            release.wait(timeout=20)       # hold the queue open
        return [p * 10 for p in payloads]

    batcher = StepBatcher({"stepper": batched}, max_batch=1)
    gate = StepGate(batcher, width=1, wait_s=10.0, timeout_s=30.0,
                    poll_s=0.01)
    got = {}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i, ("f32",))

    blocker = threading.Thread(target=lambda: submit(0))
    blocker.start()
    assert first_in.wait(20), "the blocking round never started"
    pile = [threading.Thread(target=(lambda j: lambda: submit(j))(j))
            for j in range(1, 5)]
    for t in pile:
        t.start()
    time.sleep(0.4)                        # let all four queue up behind it
    release.set()
    for t in [blocker] + pile:
        t.join(30)
        assert not t.is_alive(), "the gate hung after a piled-up round"
    assert got == {i: i * 10 for i in range(5)}, got
    assert batcher.max_group_size() == 1
    assert len(batcher.records) == 5


def test_gate_fires_a_partial_round_when_peers_never_arrive():
    batcher = StepBatcher({"stepper": lambda ps: [p + 1 for p in ps]},
                          max_batch=4)
    gate = StepGate(batcher, width=4, wait_s=0.15, timeout_s=20.0)
    got = {}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i)

    assert not _run_threads(submit, [0, 1])
    assert got == {0: 1, 1: 2}, got
    assert gate.stats().partial_rounds >= 1
    assert batcher.max_group_size() <= 2


def test_live_participants_size_the_round_without_waiting():
    batcher = StepBatcher({"stepper": lambda ps: list(ps)}, max_batch=8)
    gate = StepGate(batcher, width=8, wait_s=60.0, timeout_s=120.0)
    got = {}
    for _ in range(3):
        gate.join()

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i)

    started = time.monotonic()
    assert not _run_threads(submit, [0, 1, 2])
    elapsed = time.monotonic() - started
    for _ in range(3):
        gate.leave()
    assert got == {0: 0, 1: 1, 2: 2}, got
    # 3 joined participants => the round is full at 3; sitting out the
    # 60s window instead would mean `live` is not being honored
    assert elapsed < 20.0, elapsed
    assert batcher.max_group_size() == 3
    assert gate.stats().partial_rounds == 0
    assert gate.stats().live == 0


def test_a_failing_batched_call_fails_its_whole_round():
    def boom(payloads):
        raise ValueError(f"kernel refused {len(payloads)} payloads")

    gate = StepGate(StepBatcher({"stepper": boom}, max_batch=3), width=3,
                    wait_s=20.0, timeout_s=40.0)
    seen = {}

    def submit(i):
        try:
            gate.submit(f"r{i}", "stepper", i)
            seen[i] = "completed"
        except ValueError as exc:
            seen[i] = str(exc)

    assert not _run_threads(submit, [0, 1, 2])
    assert set(seen.values()) == {"kernel refused 3 payloads"}, seen


def test_a_missing_result_is_an_error_not_another_requests_value():
    gate = StepGate(_StubBatcher(lambda reqs: {}), width=2, wait_s=20.0,
                    timeout_s=40.0)
    seen = {}

    def submit(i):
        try:
            gate.submit(f"r{i}", "stepper", i)
            seen[i] = "completed"
        except RuntimeError as exc:
            seen[i] = str(exc)

    assert not _run_threads(submit, [0, 1])
    assert len(seen) == 2, seen
    for msg in seen.values():
        assert "refusing to hand back another request" in msg, seen


def test_gate_times_out_instead_of_hanging():
    gate = StepGate(StepBatcher({"stepper": lambda ps: list(ps)},
                                max_batch=4),
                    width=4, wait_s=60.0, timeout_s=0.3, poll_s=0.02)
    for _ in range(4):
        gate.join()               # a round needs 4; only one will arrive
    try:
        gate.submit("lonely", "stepper", 1)
    except TimeoutError as exc:
        assert "refusing to hang" in str(exc)
    else:
        raise AssertionError("a stuck round must raise, not block forever")


def test_gate_never_mixes_signatures_in_one_call():
    seen_groups = []

    def batched(payloads):
        seen_groups.append(list(payloads))
        return [p * 2 for p in payloads]

    batcher = StepBatcher({"stepper": batched}, max_batch=4)
    gate = StepGate(batcher, width=4, wait_s=20.0, timeout_s=40.0)
    got = {}
    sigs = {0: ("a",), 1: ("b",), 2: ("a",), 3: ("b",)}

    def submit(i):
        got[i] = gate.submit(f"r{i}", "stepper", i, sigs[i])

    assert not _run_threads(submit, list(range(4)))
    assert got == {i: i * 2 for i in range(4)}, got
    assert batcher.cross_signature_mixes() == 0
    assert len(batcher.records) == 2, batcher.records
    assert sorted(len(g) for g in seen_groups) == [2, 2], seen_groups


def test_gate_construction_is_fail_closed():
    b = StepBatcher({"stepper": lambda ps: list(ps)})
    for kwargs in ({"width": 0}, {"width": 2, "wait_s": -1.0},
                   {"width": 2, "timeout_s": 0.0},
                   {"width": 2, "poll_s": 0.0}):
        try:
            StepGate(b, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"StepGate accepted {kwargs}")
    gate = StepGate(b, width=2)
    try:
        gate.submit("", "stepper", 1)
    except ValueError as exc:
        assert "non-empty string" in str(exc)
    else:
        raise AssertionError("an empty request id must be rejected")


# ------------------------------------------------------------ integration
def _rollout_impls(gate):
    def encoder(ctx, state):
        session = current_session()
        if "ctx" in state:
            state["hits"] = state.get("hits", 0) + 1
            return {"latent": state["ctx"], "grounded_by": "cache"}
        state["ctx"] = float(int(session[1:]) + 1)
        return {"latent": state["ctx"], "grounded_by": "encode"}

    def stepper(ctx, state):
        session = current_session()
        nxt = gate.submit(session, "stepper", ctx["latent"], ("f32",))
        state["steps"] = state.get("steps", 0) + 1
        return {"latent": nxt}

    return {"encoder": encoder, "stepper": stepper}


def _drive(sessions, executor_for, walkset, gate, threaded):
    """Two requests per session: cold grounding, then a warm re-grounding."""
    out = {}

    def one(session):
        run_request(executor_for(session), walkset, _chooser, session=session)
        out[session] = run_request(executor_for(session), walkset, _chooser,
                                   session=session)

    if not threaded:
        for session in sessions:
            one(session)
        return out
    for _ in sessions:
        gate.join()
    try:
        errs = _run_threads(one, sessions)
    finally:
        for _ in sessions:
            gate.leave()
    assert not errs, errs
    return out


def test_concurrent_batched_sessions_match_sessions_run_alone():
    steps = 5
    sessions = [f"s{i}" for i in range(6)]
    graph, walkset = _graph(), _walkset(steps)
    step_fn = (lambda ps: [p * 2.0 + 1.0 for p in ps])

    solo_batcher = StepBatcher({"stepper": step_fn}, max_batch=1)
    solo_gate = StepGate(solo_batcher, width=1, wait_s=10.0, timeout_s=30.0)
    solo_impls = _rollout_impls(solo_gate)
    solo_execs = {s: WalkExecutor(graph, solo_impls) for s in sessions}
    solo = _drive(sessions, lambda s: solo_execs[s], walkset, solo_gate,
                  threaded=False)

    batch_batcher = StepBatcher({"stepper": step_fn}, max_batch=len(sessions))
    batch_gate = StepGate(batch_batcher, width=len(sessions), wait_s=30.0,
                          timeout_s=60.0)
    shared = WalkExecutor(graph, _rollout_impls(batch_gate))
    batched = _drive(sessions, lambda s: shared, walkset, batch_gate,
                     threaded=True)

    for session in sessions:
        alone = solo[session].outputs["latent"]
        fused = batched[session].outputs["latent"]
        assert alone == fused, (session, alone, fused)
    # sessions started from DIFFERENT values, so a leak would not be subtle
    assert len({r.outputs["latent"] for r in solo.values()}) == len(sessions)
    for session in sessions:
        enc = shared.store.state(session, "encoder")
        stp = shared.store.state(session, "stepper")
        assert enc["hits"] == 1, (session, enc)          # second request
        assert stp["steps"] == 2 * steps, (session, stp)
    assert solo_batcher.max_group_size() == 1
    assert batch_batcher.max_group_size() == len(sessions)
    assert batch_batcher.cross_signature_mixes() == 0
    trails = {tuple(r.walk_trail) for r in batched.values()}
    assert trails == {("ground", "rollout")}, trails


def test_session_reset_clears_state_the_gate_cannot_see():
    gate = StepGate(StepBatcher({"stepper": lambda ps: [p + 1 for p in ps]},
                                max_batch=1), width=1, wait_s=5.0,
                    timeout_s=20.0)
    ex = WalkExecutor(_graph(), _rollout_impls(gate))
    walkset = _walkset(2)
    first = run_request(ex, walkset, _chooser, session="s3")
    ex.reset_session("s3")
    again = run_request(ex, walkset, _chooser, session="s3")
    assert first.outputs["latent"] == again.outputs["latent"]
    assert ex.store.state("s3", "encoder").get("hits") is None


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
