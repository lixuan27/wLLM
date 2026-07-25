"""Trace-store unit tests: round-trip, fail-closed validation, dedup,
query filters, failure patterns, known-bad lookup, corrupt-line
tolerance, and the beta seed (six real traces, idempotent).

The trace store is the project's append-only experiment memory —
failures persist with their reasons so a planner never re-explores a
known-dead configuration blind. Every fail-closed rule gets a negative
test.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.tracestore import (
    Trace, TraceStore, beta_seed_traces, seed_beta_traces,
)


def _trace(**over) -> Trace:
    base = dict(
        model="org/model-a", hardware="1xH200", runtime="wllm-serving",
        workload="decode 128 tok",
        candidate={"pass": "static_kv_cache", "gpus": 1},
        status="accepted", reason="measured exact",
        metrics={"speedup": 2.0}, evidence="job 1",
        recorded="2026-07-24",
    )
    base.update(over)
    return Trace(**base)


# ------------------------------------------------------------ round-trip

def test_roundtrip_append_load():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / "traces.jsonl"   # parent auto-mkdir
        st = TraceStore(path)
        t1 = _trace()
        t2 = _trace(status="rejected", reason="drifted",
                    candidate={"pass": "compile", "gpus": 1})
        id1, id2 = st.append(t1), st.append(t2)
        assert id1 != id2
        st2 = TraceStore(path)
        assert st2.corrupt_lines == 0
        assert [t.trace_id for t in st2.all()] == [id1, id2]
        assert st2.all()[0] == t1              # full field round-trip
        assert st2.all()[1].reason == "drifted"


# ------------------------------------------------------------ validation

def test_validation_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        st = TraceStore(Path(td) / "t.jsonl")
        checks = [
            (dict(status="vibes"), "unknown status"),
            (dict(status="rejected", reason="  "), "non-empty"),
            (dict(status="failed", reason=""), "non-empty"),
            (dict(recorded="24-07-2026"), "YYYY-MM-DD"),
            (dict(recorded=""), "YYYY-MM-DD"),
            (dict(model=""), "model is empty"),
            (dict(candidate={}), "candidate is empty"),
        ]
        for over, frag in checks:
            bad = _trace(**over)
            assert any(frag in e for e in bad.validate()), (over, frag)
            try:
                st.append(bad)
            except ValueError as exc:
                assert frag in str(exc), (over, str(exc))
            else:
                raise AssertionError(f"append must fail closed: {over}")
        # fail-closed means nothing was written at all
        assert st.all() == []
        assert not st.path.is_file()


# ----------------------------------------------------------------- dedup

def test_dedup_same_trace_one_line():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.jsonl"
        st = TraceStore(path)
        id1 = st.append(_trace())
        # same identity key: different prose/metrics, reordered knobs
        again = _trace(metrics={"speedup": 2.1}, evidence="job 2",
                       candidate={"gpus": 1, "pass": "static_kv_cache"})
        id2 = st.append(again)
        assert id2 == id1
        assert st.deduped == 1
        assert len(st.all()) == 1
        lines = [x for x in path.read_text().splitlines() if x.strip()]
        assert len(lines) == 1
        # a changed knob is a different trace
        id3 = st.append(_trace(candidate={"pass": "static_kv_cache",
                                          "gpus": 2}))
        assert id3 != id1 and len(st.all()) == 2


def test_trace_id_over_identity_fields_only():
    base = _trace()
    same = _trace(metrics={}, evidence="elsewhere",
                  recorded="2026-07-25")
    assert same.trace_id == base.trace_id
    assert _trace(status="rejected", reason="r").trace_id \
        != base.trace_id
    assert _trace(hardware="2xH200").trace_id != base.trace_id
    assert _trace(runtime="wllm-native").trace_id != base.trace_id


# ----------------------------------------------------------------- query

def test_query_filters():
    with tempfile.TemporaryDirectory() as td:
        st = TraceStore(Path(td) / "t.jsonl")
        st.append(_trace())
        st.append(_trace(model="org/model-b", hardware="2xH200",
                         runtime="wllm-native",
                         candidate={"pass": "native_bf16", "gpus": 1},
                         status="rejected", reason="slower"))
        assert len(st.query()) == 2
        assert len(st.query(model="org/model-a")) == 1
        assert len(st.query(hardware="2xH200")) == 1
        assert len(st.query(runtime="wllm-native")) == 1
        assert len(st.query(status="rejected")) == 1
        hits = st.query(pass_name="native_bf16")
        assert len(hits) == 1 and hits[0].model == "org/model-b"
        assert st.query(model="org/model-a", status="rejected") == []
        assert st.query(pass_name="no_such_pass") == []


# ------------------------------------------------------ failure patterns

def test_failure_patterns_aggregate():
    with tempfile.TemporaryDirectory() as td:
        st = TraceStore(Path(td) / "t.jsonl")
        st.append(_trace())                 # accepted -> excluded
        st.append(_trace(status="rejected", reason="drift 7.4/255",
                         candidate={"pass": "compile", "mode": "a"}))
        st.append(_trace(status="failed", reason="oom",
                         candidate={"pass": "compile", "mode": "b"}))
        # duplicate reason under the same pass stays unique
        st.append(_trace(status="rejected", reason="drift 7.4/255",
                         candidate={"pass": "compile", "mode": "c"}))
        pat = st.failure_patterns()
        assert set(pat) == {"compile"}
        assert pat["compile"] == ["drift 7.4/255", "oom"]


# --------------------------------------------------------------- known_bad

def test_known_bad_hit_and_miss():
    with tempfile.TemporaryDirectory() as td:
        st = TraceStore(Path(td) / "t.jsonl")
        cand = {"pass": "cfg_batched", "gpus": 1}
        st.append(_trace(status="rejected", reason="not bit-exact",
                         candidate=cand))
        st.append(_trace())                 # accepted, other candidate
        hit = st.known_bad("org/model-a", "1xH200",
                           {"gpus": 1, "pass": "cfg_batched"})
        assert hit is not None and hit.reason == "not bit-exact"
        # any key change is a miss (exact-key semantics)
        assert st.known_bad("org/model-a", "2xH200", cand) is None
        assert st.known_bad("org/other", "1xH200", cand) is None
        assert st.known_bad("org/model-a", "1xH200",
                            {"pass": "cfg_batched", "gpus": 2}) is None
        # an accepted config is never "known bad"
        ok_cand = {"pass": "static_kv_cache", "gpus": 1}
        assert st.known_bad("org/model-a", "1xH200", ok_cand) is None


# --------------------------------------------------------- corrupt lines

def test_corrupt_line_skipped_and_counted():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.jsonl"
        st = TraceStore(path)
        good_id = st.append(_trace())
        with path.open("a") as fh:
            fh.write("{{{ this is not json\n")
        st2 = TraceStore(path)
        assert st2.corrupt_lines == 1
        assert [t.trace_id for t in st2.all()] == [good_id]
        # wrong-schema JSON object also counts as corrupt, not a crash
        with path.open("a") as fh:
            fh.write(json.dumps({"model": "x", "bogus": True}) + "\n")
        st3 = TraceStore(path)
        assert st3.corrupt_lines == 2
        assert len(st3.all()) == 1


# ------------------------------------------------------------- beta seed

def test_seed_beta_traces_idempotent():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "beta.jsonl"
        st = TraceStore(path)
        assert seed_beta_traces(st) == 6
        assert len(st.all()) == 6
        assert seed_beta_traces(st) == 0    # second run adds nothing
        assert st.deduped == 6
        st2 = TraceStore(path)              # fresh load, same result
        assert st2.corrupt_lines == 0 and len(st2.all()) == 6
        assert seed_beta_traces(st2) == 0
        lines = [x for x in path.read_text().splitlines() if x.strip()]
        assert len(lines) == 6


def test_seed_content_matches_reports():
    seeds = beta_seed_traces()
    assert len(seeds) == 6
    for t in seeds:
        assert t.validate() == [], t
        assert t.evidence, "every seed must point at real evidence"
        assert t.recorded in ("2026-07-24", "2026-07-25")
    accepted = [t for t in seeds if t.status == "accepted"]
    rejected = [t for t in seeds if t.status == "rejected"]
    assert len(accepted) == 3 and len(rejected) == 3
    with tempfile.TemporaryDirectory() as td:
        st = TraceStore(Path(td) / "b.jsonl")
        seed_beta_traces(st)
        # accepted branch-parallel: 1.44x bit-exact, job 196293
        ok = st.query(pass_name="cfg_branch_parallel",
                      status="accepted")
        assert len(ok) == 1 and ok[0].hardware == "2xH200"
        assert ok[0].metrics["speedup"] == 1.44
        assert "196293" in ok[0].evidence
        # accepted static KV 2.75x and native bf16 4.59x
        kv = st.query(pass_name="static_kv_cache")[0]
        assert kv.metrics["speedup"] == 2.75
        assert kv.model == "Qwen/Qwen3-VL-8B-Instruct"
        vla = st.query(pass_name="native_bf16")[0]
        assert vla.metrics["speedup"] == 4.59
        assert vla.runtime == "wllm-native"
        # planner can skip the known-dead batched-CFG config
        bad = st.known_bad("Wan-AI/Wan2.2-TI2V-5B", "1xH200",
                           {"pass": "cfg_batched", "gpus": 1})
        assert bad is not None and "251/255" in bad.reason
        # failure patterns cover exactly the three rejected passes
        pat = st.failure_patterns()
        assert set(pat) == {"cfg_batched",
                            "torch_compile_max_autotune",
                            "torch_compile_reduce_overhead"}
        for reasons in pat.values():
            assert all(r.strip() for r in reasons)


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
