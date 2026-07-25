"""Three-source candidate planning: registry x profile x traces x SLO."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.candidates import plan_candidates
from wllm.control.cli import main as cli_main
from wllm.control.registry import default_registry, parse_capability
from wllm.control.slo import SLOSpec
from wllm.control.tracestore import TraceStore, seed_beta_traces
from wllm.profiles import ModelProfile, load_profiles

WAN = "Wan-AI/Wan2.2-TI2V-5B"


def _seeded_store(td: Path) -> TraceStore:
    store = TraceStore(td / "traces.jsonl")
    seed_beta_traces(store)
    return store


def test_full_pipeline_on_known_model():
    with tempfile.TemporaryDirectory() as td:
        rep = plan_candidates(
            WAN, hardware="2xH200",
            context={"num_gpus": 2, "model_uses_cfg": True},
            registry=default_registry(), profiles=load_profiles(),
            trace_store=_seeded_store(Path(td)), today="2026-07-26")
        assert rep.mode == "plan"
        serving = next(c for c in rep.candidates
                       if c.backend == "wllm-serving")
        assert "cfg_branch_parallel" in serving.passes
        # profile-backed vs registry-only claims are distinguished
        assert any("lack profile evidence" in p for p in serving.provenance)
        # bounded compile pass is rejected under exact policy (registry)
        assert any("torch_compile_max_autotune" in k
                   for k in rep.rejected)


def test_known_bad_pass_is_dropped_with_trace_provenance():
    cap = parse_capability({
        "backend": "toy", "models": {"compatible": ["Wan-AI/*"]},
        "passes": {"cfg_batched": {"quality": "exact"},
                   "safe_pass": {"quality": "exact"}},
    })
    with tempfile.TemporaryDirectory() as td:
        rep = plan_candidates(
            WAN, hardware="1xH200", context={"num_gpus": 1},
            registry={"toy": cap},
            trace_store=_seeded_store(Path(td)))
        toy = next(c for c in rep.candidates if c.backend == "toy")
        # the seeded 251/255 divergence kills cfg_batched, with cause
        assert "cfg_batched" not in toy.passes
        assert "safe_pass" in toy.passes
        assert any("known-bad from trace" in p and "251" in p
                   for p in toy.provenance)


def test_whole_candidate_known_bad_rejects_backend():
    cap = parse_capability({
        "backend": "toy", "models": {"compatible": ["Wan-AI/*"]},
        "passes": {"p1": {"quality": "exact"}},
    })
    with tempfile.TemporaryDirectory() as td:
        store = TraceStore(Path(td) / "t.jsonl")
        from wllm.control.tracestore import Trace
        store.append(Trace(
            model=WAN, hardware="1xH200", runtime="toy",
            workload="w", candidate={"backend": "toy", "passes": ["p1"]},
            status="failed", reason="OOM at load",
            recorded="2026-07-25"))
        rep = plan_candidates(
            WAN, hardware="1xH200", context={"num_gpus": 1},
            registry={"toy": cap}, trace_store=store)
        assert rep.mode == "diagnose-only"
        assert any("OOM at load" in v for v in rep.rejected.values())


def test_stale_profile_downgrades_and_incompat_prunes():
    profile = ModelProfile.from_dict({
        "model_family": "toy", "detection_ids": ["org/toy"],
        "optimizations_lossless": [
            {"name": "a", "quality_class": "exact"},
            {"name": "b", "quality_class": "exact"}],
        "incompatibilities": ["a + b"],
        "binding": {"last_validated": "2026-01-01",
                    "evidence_level": "declared"}})
    assert profile.validate() == []
    cap = parse_capability({
        "backend": "toy", "models": {"exact": ["org/toy"]},
        "passes": {"a": {"quality": "exact"}, "b": {"quality": "exact"}}})
    rep = plan_candidates(
        "org/toy", hardware="1xTEST", context={"num_gpus": 1},
        registry={"toy": cap}, profiles={"toy": profile},
        today="2026-07-26")
    assert any("STALE" in n for n in rep.notes)
    toy = rep.candidates[0]
    assert toy.passes == ["a"]           # b dropped by incompatibility
    assert any("incompatible" in p for p in toy.provenance)


def test_slo_gates_declared_and_invalid_slo_rejected():
    reg = default_registry()
    slo = SLOSpec.from_dict({
        "hard_constraints": {"p99_latency_ms": 2000,
                             "api_compatibility": "strict"}})
    rep = plan_candidates(WAN, hardware="2xH200",
                          context={"num_gpus": 2, "model_uses_cfg": True},
                          registry=reg, slo=slo)
    assert "p99_latency_ms <= 2000" in rep.pending_gates
    assert "api_compatibility == strict" in rep.pending_gates
    bad = SLOSpec.from_dict({"preferences": {"latency": 0.5}})
    try:
        plan_candidates(WAN, hardware="2xH200", context={},
                        registry=reg, slo=bad)
    except ValueError as exc:
        assert "invalid SLO" in str(exc)
    else:
        raise AssertionError("invalid SLO must be rejected")


def test_unknown_model_is_diagnose_only():
    rep = plan_candidates("nobody/mystery", hardware="1xTEST",
                          context={"num_gpus": 1},
                          registry=default_registry(),
                          profiles=load_profiles(), today="2026-07-26")
    assert rep.mode == "diagnose-only"
    assert any("no verified profile matches" in n for n in rep.notes)


def test_cli_candidates_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        store_path = Path(td) / "traces.jsonl"
        seed_beta_traces(TraceStore(store_path))
        rc = cli_main(["candidates", td, "--model", WAN,
                       "--hardware", "2xH200", "--num-gpus", "2",
                       "--context", '{"model_uses_cfg": true}',
                       "--traces", str(store_path),
                       "--today", "2026-07-26"])
        assert rc == 0
        plan = (Path(td) / ".wllm" / "plans" /
                f"candidates-{WAN.replace('/', '_')}.json")
        doc = json.loads(plan.read_text())
        assert doc["mode"] == "plan" and doc["candidates"]
        rc2 = cli_main(["candidates", td, "--model", "nobody/mystery"])
        assert rc2 == 3


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
