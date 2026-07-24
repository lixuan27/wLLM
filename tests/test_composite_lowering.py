"""Plan lowering: DeploymentPlan stages -> executor placement, fail-closed."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.composite import (
    Component, ComponentGraph, Edge, LoweringReport, Seq, Walk, WalkExecutor,
    lower_plan, require,
)
from wllm.graph.regions import NodeOp
from wllm.planner.plan import DeploymentPlan, Hardware, OverlapMode, Stage

GB = 10 ** 9


def _graph() -> ComponentGraph:
    return ComponentGraph(
        name="demo",
        components=[
            Component("text_encoder", NodeOp.ENCODER),
            Component("dit", NodeOp.TRANSFORMER, batchable=True),
            Component("vae", NodeOp.CODEC),
            Component("action_head", NodeOp.POLICY_HEAD,
                      placement_domain="fixed:gpu0"),
        ],
        edges=[
            Edge("text_encoder", "dit"),
            Edge("dit", "vae"),
            Edge("vae", "action_head"),
        ])


def _plan() -> DeploymentPlan:
    return DeploymentPlan(id="p0", stages=[
        Stage(id="s0", node_ids=["text_encoder", "action_head"], device=0),
        Stage(id="s1", node_ids=["dit", "vae"], device=1),
    ])


def _impls():
    return {
        "text_encoder": lambda ctx, state: {"emb": 1},
        "dit": lambda ctx, state: {"latent": 2},
        "vae": lambda ctx, state: {"frames": 3},
        "action_head": lambda ctx, state: {"action": [0.1]},
    }


# ------------------------------------------------------------- happy path

def test_happy_path_two_stages():
    rep = lower_plan(_graph(), _plan())
    assert isinstance(rep, LoweringReport)
    assert rep.ok and rep.problems == []
    assert rep.placement == {"text_encoder": "gpu0", "action_head": "gpu0",
                             "dit": "gpu1", "vae": "gpu1"}
    assert rep.stage_of == {"text_encoder": "s0", "action_head": "s0",
                            "dit": "s1", "vae": "s1"}
    assert rep.overlapped == []


def test_cross_chunk_overlap_is_recorded_not_claimed():
    plan = _plan()
    plan.stages[1].overlap = OverlapMode.CROSS_CHUNK
    rep = lower_plan(_graph(), plan)
    assert rep.ok                       # overlap is allowed
    assert rep.overlapped == ["s1"]     # ...and recorded, nothing more


# ----------------------------------------------------- graph/plan mismatch

def test_unknown_node_id_is_a_problem():
    plan = _plan()
    plan.stages[1].node_ids.append("ghost")
    rep = lower_plan(_graph(), plan)
    assert not rep.ok
    joined = " ".join(rep.problems)
    assert "'ghost'" in joined and "not a component" in joined
    assert "ghost" not in rep.placement and "ghost" not in rep.stage_of


def test_unassigned_and_doubly_assigned_components():
    plan = DeploymentPlan(id="p_bad", stages=[
        Stage(id="s0", node_ids=["text_encoder", "dit", "action_head"],
              device=0),
        Stage(id="s1", node_ids=["dit"], device=1),   # dit again; vae missing
    ])
    rep = lower_plan(_graph(), plan)
    assert not rep.ok
    joined = " ".join(rep.problems)
    assert "assigned by both stage 's0' and stage 's1'" in joined
    assert "'vae' is not assigned by any stage" in joined
    assert rep.stage_of["dit"] == "s0"          # first assignment kept
    assert rep.placement["dit"] == "gpu0"       # ...as diagnostic evidence


# -------------------------------------------------------- placement domains

def test_fixed_pin_honored_and_violated():
    rep_ok = lower_plan(_graph(), _plan())      # action_head on device 0
    assert rep_ok.ok and rep_ok.placement["action_head"] == "gpu0"

    plan = DeploymentPlan(id="p_pin", stages=[
        Stage(id="s0", node_ids=["text_encoder", "dit", "vae"], device=0),
        Stage(id="s1", node_ids=["action_head"], device=1),   # pin broken
    ])
    rep = lower_plan(_graph(), plan)
    assert not rep.ok
    joined = " ".join(rep.problems)
    assert "pinned to 'gpu0'" in joined and "'gpu1'" in joined
    assert "action_head" not in rep.placement   # never place a broken pin


def test_cpu_domain_component_in_gpu_stage_is_a_problem():
    g = _graph()
    g.components.append(Component("tok", NodeOp.TOKENIZER,
                                  placement_domain="cpu"))
    plan = DeploymentPlan(id="p_cpu", stages=[
        Stage(id="s0", node_ids=["text_encoder", "action_head", "tok"],
              device=0),
        Stage(id="s1", node_ids=["dit", "vae"], device=1),
    ])
    rep = lower_plan(g, plan)
    assert not rep.ok
    joined = " ".join(rep.problems)
    assert "placement_domain 'cpu'" in joined
    assert "cannot place cpu-domain" in joined
    assert "tok" not in rep.placement


def test_unknown_placement_domain_refused():
    g = _graph()
    g.components.append(Component("odd", NodeOp.CUSTOM,
                                  placement_domain="quantum"))
    plan = _plan()
    plan.stages[0].node_ids.append("odd")
    rep = lower_plan(g, plan)
    assert not rep.ok
    assert any("unknown placement_domain 'quantum'" in p
               for p in rep.problems)
    assert "odd" not in rep.placement


# --------------------------------------------- unlowerable / out of bounds

def test_parallel_degree_rejected_honestly():
    plan = _plan()
    plan.stages[1].parallel_degree = 2
    rep = lower_plan(_graph(), plan)
    assert not rep.ok
    joined = " ".join(rep.problems)
    assert "parallel_degree 2 not yet lowerable" in joined
    assert "refusing to pretend a parallel group exists" in joined


def test_hardware_bound_violation():
    hw = Hardware(num_gpus=2, hbm_bytes_per_gpu=140 * GB)
    ok = lower_plan(_graph(), _plan(), hardware=hw)
    assert ok.ok                                # devices 0,1 fit in 2 GPUs

    plan = _plan()
    plan.stages[1].device = 2                   # device index 2 of 2
    rep = lower_plan(_graph(), plan, hardware=hw)
    assert not rep.ok
    assert any("needs device 2 but hardware has 2 GPUs" in p
               for p in rep.problems)

    # parallel_degree widens the device range checked against hardware
    plan2 = _plan()
    plan2.stages[1].parallel_degree = 2         # devices 1..2 of 2
    rep2 = lower_plan(_graph(), plan2, hardware=hw)
    joined = " ".join(rep2.problems)
    assert "needs device 2 but hardware has 2 GPUs" in joined
    assert "not yet lowerable" in joined        # both problems reported


# ----------------------------------------------------------------- require

def test_require_raises_with_all_problems_or_returns_placement():
    plan = DeploymentPlan(id="p_bad", stages=[
        Stage(id="s0", node_ids=["text_encoder", "ghost"], device=0,
              parallel_degree=2),
    ])
    rep = lower_plan(_graph(), plan)
    try:
        require(rep)
    except ValueError as exc:
        msg = str(exc)
        assert "does not lower cleanly" in msg
        assert "'ghost'" in msg                       # unknown node
        assert "not yet lowerable" in msg             # parallel_degree
        assert "not assigned by any stage" in msg     # dit/vae/action_head
    else:
        raise AssertionError("require() must fail closed on problems")
    good = lower_plan(_graph(), _plan())
    assert require(good) is good.placement


# -------------------------------------------------------------- end to end

def test_lowered_plan_runs_on_executor_with_matching_devices():
    g, plan = _graph(), _plan()
    placement = require(lower_plan(g, plan))
    ex = WalkExecutor(g, _impls(), placement=placement)
    out = ex.run(Walk([Seq("text_encoder"), Seq("dit"), Seq("vae"),
                       Seq("action_head")]), session="s1", ctx={})
    assert out["frames"] == 3 and out["action"] == [0.1]
    used = ex.devices_used()
    # every component ran exactly on the label its plan stage lowers to
    stage_device = {st.id: st.device for st in plan.stages}
    rep = lower_plan(g, plan)
    for cid, stage_id in rep.stage_of.items():
        assert used[cid] == {f"gpu{stage_device[stage_id]}"}
    # and the set of GPU indices touched matches the plan's own account
    touched = {int(d.removeprefix("gpu"))
               for devs in used.values() for d in devs}
    assert touched == plan.devices_used()


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
