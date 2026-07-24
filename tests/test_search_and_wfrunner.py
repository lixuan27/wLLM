"""Tests: successive-halving searcher + substrate L0 launch adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wllm.backends.worldfoundry.importer import CatalogEntry
from wllm.backends.worldfoundry.runner import (
    SubstrateInstall, build_infer_spec, resolve_env_name,
)
from wllm.planner.plan import DeploymentPlan, Stage
from wllm.planner.search import Measurement, successive_halving


def _plans(n=6):
    return [DeploymentPlan(id=f"p{i}",
                           stages=[Stage(id="s", node_ids=["n"], device=0)])
            for i in range(n)]


def _measure_factory(latency_by_plan, fail_ids=()):
    calls = []

    def measure(plan, duration_s):
        calls.append((plan.id, duration_s))
        if plan.id in fail_ids:
            return Measurement(plan_id=plan.id, duration_s=duration_s,
                               ok=False, error="boom")
        return Measurement(plan_id=plan.id, duration_s=duration_s, ok=True,
                           latency_ms=latency_by_plan[plan.id],
                           sustained_rate=1000.0 / latency_by_plan[plan.id])
    return measure, calls


def test_halving_converges_to_best():
    plans = _plans(6)
    lat = {f"p{i}": 100.0 + 10 * i for i in range(6)}   # p0 best
    measure, calls = _measure_factory(lat)
    res = successive_halving(plans, measure, probe_s=1.0, growth=2.0,
                             min_final=2)
    best = res.best()
    assert best is not None and best.plan.id == "p0"
    assert len(res.survivors()) <= 3
    # later rounds run longer than the probe round
    durations = sorted({d for _, d in calls})
    assert durations[0] == 1.0 and durations[-1] >= 2.0
    # culled entries carry reasons
    culled = [r for r in res.records if r.culled_at_round is not None]
    assert culled and all(r.cull_reason for r in culled)


def test_halving_culls_failures_and_survives_exceptions():
    plans = _plans(4)
    lat = {f"p{i}": 100.0 + i for i in range(4)}
    measure, _ = _measure_factory(lat, fail_ids={"p1"})
    res = successive_halving(plans, measure, probe_s=1.0, min_final=1)
    rec = {r.plan.id: r for r in res.records}["p1"]
    assert rec.culled_at_round == 1 and "boom" in rec.cull_reason

    def explosive(plan, duration_s):
        raise RuntimeError("kaboom")

    res2 = successive_halving(_plans(2), explosive, probe_s=1.0, min_final=1)
    assert not res2.survivors()
    assert all("kaboom" in r.cull_reason for r in res2.records)


def test_halving_respects_budget():
    plans = _plans(8)
    lat = {f"p{i}": 100.0 + i for i in range(8)}
    measure, calls = _measure_factory(lat)
    res = successive_halving(plans, measure, probe_s=1.0, budget_s=0.0,
                             min_final=2)
    # zero budget: at most one partial round attempted
    assert res.spent_s >= 0.0 and len(calls) <= 1


def test_objective_sustained_rate():
    plans = _plans(4)
    lat = {f"p{i}": 100.0 + 10 * i for i in range(4)}
    measure, _ = _measure_factory(lat)
    res = successive_halving(plans, measure, probe_s=1.0, min_final=1,
                             objective="sustained-rate")
    assert res.best().plan.id == "p0"   # lowest latency = highest rate here
    assert "objective=sustained-rate" in res.report()


def _entry(env=""):
    return CatalogEntry(id="demo-model", category="video", path="x.yaml",
                        environment=env)


def test_env_resolution():
    inst = SubstrateInstall(repo_root="/tmp/wf")
    assert resolve_env_name(_entry(""), inst) == inst.unified_env
    assert resolve_env_name(_entry("_unified"), inst) == inst.unified_env
    assert resolve_env_name(_entry("special-env"), inst) == "special-env"


def test_build_infer_spec_shape(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        inst = SubstrateInstall(repo_root=td, ckpt_dir=f"{td}/ckpts")
        spec = build_infer_spec(_entry(), inst, output_dir=f"{td}/out",
                                prompt="hello", gpu_indices=[2],
                                extra_args=["--frames", "17"])
        argv = spec.argv
        assert argv[:4] == ["conda", "run", "--no-capture-output", "-n"]
        assert "worldfoundry.studio.workspace_job" in argv
        assert "--model-id" in argv and "demo-model" in argv
        assert "--prompt" in argv and "hello" in argv
        assert argv[-2:] == ["--frames", "17"]
        assert spec.gpu_indices == [2]
        assert spec.env["WORLDFOUNDRY_CKPT_DIR"] == f"{td}/ckpts"
        assert Path(f"{td}/out").is_dir()   # created eagerly
        assert spec.artifacts[0].kind == "output_dir"


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
    sys.exit(1 if fails else 0)
