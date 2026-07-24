"""Step definitions binding tests/features/*.feature to real control code."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from bdd_runner import StepRegistry, run_feature  # noqa: E402

from wllm.control.cli import main as cli_main  # noqa: E402
from wllm.control.receipt import Receipt  # noqa: E402
from wllm.control.state import DeployManager  # noqa: E402

steps = StepRegistry()


def _project(ctx) -> Path:
    if "project" not in ctx:
        td = Path(tempfile.mkdtemp(prefix="wllm_bdd_"))
        ctx["project"] = td
        ctx["_teardown"] = lambda: shutil.rmtree(td, ignore_errors=True)
    return ctx["project"]


def _receipt(**over) -> Receipt:
    base = dict(
        plan_id="bdd-plan", backend="wllm-serving", source_revision="abc",
        model_revision="r1", hardware="1xH200", passes=["static_kv_cache"],
        perf={"p50_ms": 100.0, "p95_ms": 130.0},
        baseline_perf={"p50_ms": 250.0},
        quality={"verdict": "exact"},
        authenticity={"cache_active": True},
    )
    base.update(over)
    return Receipt(**base)


# ------------------------------------------------------------------- given

@steps.given(r"a project directory with a runnable entrypoint and a model config")
def given_project(ctx):
    td = _project(ctx)
    (td / "inference.py").write_text("print('run')\n")
    (td / "config.json").write_text(json.dumps(
        {"architectures": ["DemoDiT"], "_name_or_path": "org/demo"}))


@steps.given(r"a measured receipt with passing checks and a real speedup")
def given_good_receipt(ctx):
    ctx["receipt"] = _receipt()
    assert ctx["receipt"].speedup() and ctx["receipt"].speedup() > 1.0


@steps.given(r"a measured receipt whose log matched a forbidden fallback pattern")
def given_fallback_receipt(ctx):
    ctx["receipt"] = _receipt(plan_id="bdd-fallback",
                              fallback_hits=["falling back to eager"])


@steps.given(r"a receipt claiming success but carrying no performance numbers")
def given_unmeasured_receipt(ctx):
    ctx["receipt"] = _receipt(plan_id="bdd-unmeasured", perf={})


# -------------------------------------------------------------------- when

@steps.when(r"the agent runs wllm inspect")
def when_inspect(ctx):
    rc = cli_main(["inspect", str(_project(ctx)), "--no-gpu-probe"])
    assert rc == 0, f"inspect rc={rc}"


@steps.when(r'the agent plans for model "([^"]+)" with (\d+) GPUs and CFG enabled')
def when_plan(ctx, model, gpus):
    rc = cli_main(["plan", str(_project(ctx)), "--model", model,
                   "--num-gpus", gpus,
                   "--context", '{"model_uses_cfg": true}'])
    assert rc == 0, f"plan rc={rc}"
    ctx["plan"] = _load_plan(ctx, model)


@steps.when(r'the agent plans for unknown model "([^"]+)"')
def when_plan_unknown(ctx, model):
    ctx["plan_rc"] = cli_main(["plan", str(_project(ctx)), "--model", model])
    ctx["plan"] = _load_plan(ctx, model)


@steps.when(r"the agent (?:applies|tries to apply) the receipt")
def when_apply(ctx):
    mgr = DeployManager(_project(ctx) / ".wllm")
    ctx["mgr"] = mgr
    try:
        mgr.apply(ctx["receipt"])
        ctx["apply_error"] = ""
    except PermissionError as exc:
        ctx["apply_error"] = str(exc)


@steps.when(r"the agent rolls back")
def when_rollback(ctx):
    ctx["mgr"].rollback()


def _load_plan(ctx, model):
    plan = (_project(ctx) / ".wllm" / "plans" /
            f"plan-{model.replace('/', '_')}.json")
    return json.loads(plan.read_text()) if plan.exists() else None


# -------------------------------------------------------------------- then

@steps.then(r"a project manifest exists with at least (\d+) entrypoint")
def then_manifest(ctx, n):
    man = _project(ctx) / ".wllm" / "manifests" / "project-manifest.json"
    doc = json.loads(man.read_text())
    assert len(doc["entrypoints"]) >= int(n), doc["entrypoints"]


@steps.then(r'the plan keeps pass "([^"]+)"')
def then_plan_keeps(ctx, name):
    kept = {p for b in ctx["plan"]["backends"] for p in b["passes"]}
    assert name in kept, f"{name} not in kept passes {sorted(kept)}"


@steps.then(r'the plan rejects pass "([^"]+)" citing the quality policy')
def then_plan_rejects(ctx, name):
    reasons = {n: why for b in ctx["plan"]["backends"]
               for n, why in b["rejected_passes"].items()}
    assert name in reasons, f"{name} was not rejected: {reasons}"
    assert "policy" in reasons[name], reasons[name]


@steps.then(r"the active plan is the receipt's plan")
def then_active_is_receipt(ctx):
    assert ctx["apply_error"] == "", ctx["apply_error"]
    assert ctx["mgr"].state().active == ctx["receipt"].plan_id


@steps.then(r'the active plan is "([^"]+)"')
def then_active_is(ctx, name):
    mgr = ctx.get("mgr") or DeployManager(_project(ctx) / ".wllm")
    assert mgr.state().active == name, mgr.state().active


@steps.then(r'the apply is refused citing "([^"]+)"')
def then_apply_refused(ctx, frag):
    assert ctx["apply_error"], "apply unexpectedly succeeded"
    assert frag in ctx["apply_error"], ctx["apply_error"]


@steps.then(r"planning ends in diagnose-only mode with the reference path intact")
def then_diagnose_only(ctx):
    assert ctx["plan_rc"] == 3, f"expected diagnose-only rc 3, got {ctx['plan_rc']}"
    assert ctx["plan"] is not None, "diagnose-only must still record evidence"
    mgr = DeployManager(_project(ctx) / ".wllm")
    assert mgr.state().active == "reference"


# ------------------------------------------------- technique orchestration

from wllm.techniques import (  # noqa: E402
    QualityBudget, StepResidualCache, TechniqueOrchestrator, TechniqueSpec,
)
from wllm.techniques.step_cache import run_loop  # noqa: E402

_X0 = [1.0, -2.0, 3.0, 0.5]


def _smooth(x, k):
    return [v * 0.99 + 0.01 for v in x]


def _jumpy(x, k):
    return [v * (2.0 if k % 2 == 0 else 0.4) + 1.0 for v in x]


def _cache_candidate(step_fn, threshold):
    def run():
        c = StepResidualCache(step_fn, threshold=threshold)
        out = run_loop(c, _X0, 20)
        return out, c.authenticity()
    return run


def _tech_spec():
    return TechniqueSpec(name="step_cache", family="cache",
                         authenticity_signals=["steps_reused"])


@steps.given(r"a smooth iterative workload and a step cache candidate")
def given_smooth_cache(ctx):
    ctx["orch"] = TechniqueOrchestrator(
        lambda: run_loop(_smooth, _X0, 20),
        QualityBudget(max_rel_deviation=0.05))
    ctx["candidates"] = [(_tech_spec(), _cache_candidate(_smooth, 0.05))]


@steps.given(r"a jumpy iterative workload and a step cache candidate")
def given_jumpy_cache(ctx):
    ctx["orch"] = TechniqueOrchestrator(
        lambda: run_loop(_jumpy, _X0, 20),
        QualityBudget(max_rel_deviation=0.5))
    ctx["candidates"] = [(_tech_spec(), _cache_candidate(_jumpy, 0.05))]


@steps.given(r"a smooth iterative workload and an over-aggressive cache "
             r"candidate under a strict budget")
def given_aggressive_cache(ctx):
    ctx["orch"] = TechniqueOrchestrator(
        lambda: run_loop(_smooth, _X0, 20),
        QualityBudget(max_rel_deviation=1e-6))
    ctx["candidates"] = [(_tech_spec(), _cache_candidate(_smooth, 0.9))]


@steps.when(r"the technique orchestrator evaluates the candidates")
def when_orchestrate(ctx):
    ctx["verdicts"] = ctx["orch"].evaluate(ctx["candidates"])


@steps.then(r"the cache candidate is accepted with nonzero reuse evidence")
def then_cache_accepted(ctx):
    v = ctx["verdicts"][0]
    assert v.accepted, v.reason
    assert v.authenticity.get("steps_reused", 0) > 0


@steps.then(r"its receipt reports a bounded quality verdict")
def then_receipt_bounded(ctx):
    rec = ctx["verdicts"][0].receipt_fields()
    assert rec["quality"]["verdict"] == "bounded", rec


@steps.then(r"the cache candidate is rejected because it never engaged")
def then_cache_never_engaged(ctx):
    v = ctx["verdicts"][0]
    assert not v.accepted and "never engaged" in v.reason, v.reason


@steps.then(r"the cache candidate is rejected for exceeding the budget")
def then_cache_over_budget(ctx):
    v = ctx["verdicts"][0]
    assert not v.accepted and "quality budget exceeded" in v.reason, v.reason


# ------------------------------------------------------------------ driver

FEATURES = sorted((Path(__file__).parent / "features").glob("*.feature"))


def test_all_features():
    assert FEATURES, "no feature files found"
    failures = []
    for feature in FEATURES:
        for res in run_feature(feature, steps):
            tag = "PASS" if res.passed else "FAIL"
            print(f"{tag} [{feature.name}] {res.scenario}"
                  + ("" if res.passed
                     else f"\n      at: {res.failed_step}\n      {res.error}"))
            if not res.passed:
                failures.append(res)
    assert not failures, f"{len(failures)} scenario(s) failed"


if __name__ == "__main__":
    try:
        test_all_features()
        print("ALL PASS")
    except AssertionError as exc:
        print(f"FAILURES: {exc}")
        raise SystemExit(1)
