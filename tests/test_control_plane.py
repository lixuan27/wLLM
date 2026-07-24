"""Control-plane unit tests: spec, registry, receipt, state, inspect, cli.

Every fail-closed rule gets a negative test — the promote gate, spec
validation, pass legality, fingerprint tamper detection, and rollback
semantics are the safety core of wLLM and must never regress silently.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.cli import main as cli_main
from wllm.control.inspect import inspect_project
from wllm.control.receipt import Receipt
from wllm.control.registry import (
    BackendCap, PassCap, default_registry, legal_passes, parse_capability,
    rank_backends, scan_log,
)
from wllm.control.spec import OptimizeSpec
from wllm.control.state import DeployManager, REFERENCE


# ------------------------------------------------------------------- spec

def test_spec_defaults_valid():
    assert OptimizeSpec().validate() == []


def test_spec_bad_values_fail_closed():
    s = OptimizeSpec.from_dict({
        "objective": {"primary": "vibes"},
        "quality": {"policy": "lossless-ish"},
        "budget": "infinite",
        "contract": {"required_modalities": ["smell"]},
    })
    errs = s.validate()
    assert len(errs) >= 4
    joined = " ".join(errs)
    for frag in ("vibes", "lossless-ish", "infinite", "smell"):
        assert frag in joined


def test_spec_bounded_requires_budget():
    s = OptimizeSpec.from_dict({"quality": {"policy": "bounded"}})
    assert any("budget" in e for e in s.validate())
    s2 = OptimizeSpec.from_dict({
        "quality": {"policy": "bounded",
                    "budget": {"metric": "lpips", "max": 0.05}}})
    assert s2.validate() == []
    # exact + budget is ambiguous intent -> rejected
    s3 = OptimizeSpec.from_dict({
        "quality": {"policy": "exact",
                    "budget": {"metric": "lpips", "max": 0.05}}})
    assert any("must not carry" in e for e in s3.validate())


def test_spec_unknown_field_rejected():
    try:
        OptimizeSpec.from_dict({"project": ".", "turbo": True})
    except ValueError as exc:
        assert "turbo" in str(exc)
    else:
        raise AssertionError("unknown fields must be rejected")


def test_spec_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        s = OptimizeSpec.from_dict({
            "project": "/p",
            "hardware": {"accelerator": "H200", "count": 4},
            "objective": {"primary": "p95_first_output",
                          "secondary": ["gpu_seconds"]},
        })
        p = s.save(Path(td) / "spec.yaml")
        s2 = OptimizeSpec.load(p)
        assert s2.to_dict() == s.to_dict()


# --------------------------------------------------------------- registry

def _cap() -> BackendCap:
    return parse_capability({
        "backend": "toy",
        "models": {"exact": ["org/model-a"], "compatible": ["org/*"]},
        "modalities": {"input": ["text"], "output": ["video", "audio"]},
        "passes": {
            "compile": {"quality": "exact"},
            "cfg_parallel": {"quality": "exact",
                             "requires": {"min_num_gpus": 2,
                                          "model_uses_cfg": True}},
            "fp8": {"quality": "bounded"},
            "offload": {"quality": "exact", "conflicts": ["compile"]},
        },
        "invariants": {"forbidden_log_patterns": ["falling back to eager"]},
    })


def test_registry_model_matching():
    cap = _cap()
    assert cap.supports_model("org/model-a") == "exact"
    assert cap.supports_model("org/other") == "compatible"
    assert cap.supports_model("else/x") is None


def test_registry_rank_and_modalities():
    caps = {"toy": _cap()}
    assert rank_backends(caps, "org/model-a")[0][1] == "exact"
    # requiring an unsupported output modality filters the backend out
    assert rank_backends(caps, "org/model-a", required_out=["action"]) == []


def test_pass_legality_fail_closed():
    cap = _cap()
    dec = {d.name: d for d in legal_passes(cap, {"num_gpus": 1}, "exact")}
    assert dec["compile"].kept
    # bounded pass rejected under exact policy
    assert not dec["fp8"].kept and "exact policy" in dec["fp8"].reason
    # unmet numeric requirement, with the reason spelled out
    assert not dec["cfg_parallel"].kept
    assert "num_gpus" in dec["cfg_parallel"].reason
    # unknown fact fails closed (context does not state model_uses_cfg)
    dec2 = {d.name: d for d in legal_passes(cap, {"num_gpus": 4}, "exact")}
    assert not dec2["cfg_parallel"].kept
    assert "does not state" in dec2["cfg_parallel"].reason
    # all requirements met -> kept
    dec3 = {d.name: d for d in legal_passes(
        cap, {"num_gpus": 4, "model_uses_cfg": True}, "exact")}
    assert dec3["cfg_parallel"].kept


def test_pass_conflicts_first_kept_wins():
    cap = _cap()
    dec = {d.name: d for d in legal_passes(cap, {"num_gpus": 1}, "exact")}
    assert dec["compile"].kept and not dec["offload"].kept
    assert "conflicts" in dec["offload"].reason


def test_forbidden_log_scan_invalidates():
    cap = _cap()
    ok = scan_log(cap, "all good\nspeed 2x\n")
    assert not ok.invalidated and ok.patterns_scanned == 1
    bad = scan_log(cap, "warn: Falling Back To EAGER mode\n")
    assert bad.invalidated and bad.hits


def test_default_registry_loads():
    caps = default_registry()
    assert {"wllm-serving", "wllm-native", "torch-local"} <= set(caps)
    # the reference backend accepts anything but offers no passes
    ref = caps["torch-local"]
    assert ref.supports_model("any/model") == "compatible"
    assert ref.passes == {}
    # bounded pass in serving registry is refused under exact policy
    serv = caps["wllm-serving"]
    dec = {d.name: d for d in legal_passes(
        serv, {"num_gpus": 2, "model_uses_cfg": True}, "exact")}
    assert dec["cfg_branch_parallel"].kept
    assert not dec["torch_compile_max_autotune"].kept


# ---------------------------------------------------------------- receipt

def _receipt(**over) -> Receipt:
    base = dict(
        plan_id="plan-x", backend="wllm-serving", backend_version="0.0.1a0",
        source_revision="abc123", model_revision="rev1", hardware="1xH200",
        driver="580.95", torch_version="2.6", precision="bf16",
        passes=["static_kv_cache"],
        perf={"p50_ms": 100.0, "p95_ms": 120.0},
        baseline_perf={"p50_ms": 275.0},
        quality={"verdict": "exact"},
        authenticity={"cache_hit_rate_positive": True},
        fallback_hits=[],
    )
    base.update(over)
    return Receipt(**base)


def test_receipt_promote_pass_and_speedup():
    r = _receipt()
    assert r.promote_problems("exact") == []
    assert abs(r.speedup() - 2.75) < 1e-9


def test_receipt_promote_blocks():
    checks = [
        (dict(perf={}), "perf.p50_ms"),
        (dict(authenticity={"kernel_active": False}), "authenticity"),
        (dict(authenticity={}), "no authenticity checks"),
        (dict(fallback_hits=["falling back to eager"]), "forbidden log"),
        (dict(quality={}), "quality verdict missing"),
        (dict(quality={"verdict": "bounded"}), "exact policy"),
        (dict(quality={"verdict": "failed"}), "failed"),
        (dict(backend=""), "backend missing"),
    ]
    for over, frag in checks:
        problems = _receipt(**over).promote_problems("exact")
        assert any(frag in p for p in problems), (over, problems)


def test_receipt_fingerprint_sensitivity_and_tamper():
    r = _receipt()
    fp = r.fingerprint()
    assert _receipt(hardware="2xH200").fingerprint() != fp
    assert _receipt(perf={"p50_ms": 1.0, "p95_ms": 2.0}).fingerprint() == fp
    with tempfile.TemporaryDirectory() as td:
        path = r.save(td)
        # round-trip intact
        assert Receipt.load(path).fingerprint() == fp
        # tampering with a fingerprinted field is detected
        doc = json.loads(path.read_text())
        doc["hardware"] = "8xB200"
        path.write_text(json.dumps(doc))
        try:
            Receipt.load(path)
        except ValueError as exc:
            assert "fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("tampered receipt must be refused")


# ------------------------------------------------------------------ state

def test_apply_rollback_chain():
    with tempfile.TemporaryDirectory() as td:
        mgr = DeployManager(td)
        assert mgr.state().active == REFERENCE
        mgr.apply(_receipt(plan_id="p1"))
        mgr.apply(_receipt(plan_id="p2"))
        st = mgr.state()
        assert st.active == "p2" and st.last_known_good == "p1"
        assert mgr.active_receipt().plan_id == "p2"
        st = mgr.rollback()          # p2 -> p1
        assert st.active == "p1"
        st = mgr.rollback()          # p1 -> reference
        assert st.active == REFERENCE
        st = mgr.rollback()          # reference is sticky, still succeeds
        assert st.active == REFERENCE
        assert any(ev["event"] == "rollback" for ev in st.history)


def test_apply_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        mgr = DeployManager(td)
        bad = _receipt(plan_id="px", fallback_hits=["fallback detected"])
        try:
            mgr.apply(bad)
        except PermissionError as exc:
            assert "refusing to apply" in str(exc)
        else:
            raise AssertionError("apply must fail closed")
        assert mgr.state().active == REFERENCE


# ---------------------------------------------------------------- inspect

def _fake_project(td: Path) -> Path:
    (td / "scripts").mkdir()
    (td / "inference.py").write_text("print('hi')\n")
    (td / "scripts" / "demo_run.sh").write_text("echo demo\n")
    (td / "requirements.txt").write_text("torch>=2.6\ndiffusers\n")
    sub = td / "model"
    sub.mkdir()
    (sub / "config.json").write_text(json.dumps({
        "architectures": ["DemoDiT"], "_name_or_path": "org/demo-model"}))
    (sub / "model_index.json").write_text("{broken json")
    return td


def test_inspect_collects_evidence_and_unknowns():
    with tempfile.TemporaryDirectory() as td:
        man = inspect_project(_fake_project(Path(td)), probe_gpu=False)
        assert any(e.value == "inference.py" for e in man.entrypoints)
        assert any(e.value == "requirements.txt" for e in man.dependency_files)
        fw = {e.value for e in man.frameworks}
        assert {"pytorch", "diffusers"} <= fw
        assert any(e.value == "DemoDiT" for e in man.architectures)
        assert any(e.value == "org/demo-model" for e in man.checkpoint_refs)
        assert any("unparseable" in u for u in man.unknowns)
        assert any("git revision" in u for u in man.unknowns)
        assert man.gpu_probe == "not-attempted"


def test_inspect_empty_project_reports_unknowns():
    with tempfile.TemporaryDirectory() as td:
        man = inspect_project(td, probe_gpu=False)
        assert any("no entrypoint" in u for u in man.unknowns)
        assert any("no model identity" in u for u in man.unknowns)


# -------------------------------------------------------------------- cli

def test_cli_end_to_end_smoke(capsys=None):
    with tempfile.TemporaryDirectory() as td:
        proj = _fake_project(Path(td))
        assert cli_main(["inspect", str(proj), "--no-gpu-probe"]) == 0
        assert (proj / ".wllm" / "manifests" /
                "project-manifest.json").exists()
        rc = cli_main(["plan", str(proj), "--model", "Wan-AI/Wan2.2-TI2V-5B",
                       "--num-gpus", "2",
                       "--context", '{"model_uses_cfg": true}'])
        assert rc == 0
        plan_files = list((proj / ".wllm" / "plans").glob("*.json"))
        assert plan_files
        doc = json.loads(plan_files[0].read_text())
        serving = next(b for b in doc["backends"]
                       if b["backend"] == "wllm-serving")
        assert "cfg_branch_parallel" in serving["passes"]
        # receipt -> verify -> apply -> report -> rollback
        rec = _receipt(plan_id="cli-plan")
        rpath = rec.save(proj / ".wllm" / "receipts")
        assert cli_main(["verify", "--receipt", str(rpath)]) == 0
        assert cli_main(["apply", str(proj), "--receipt", str(rpath)]) == 0
        assert cli_main(["report", str(proj)]) == 0
        assert cli_main(["rollback", str(proj)]) == 0
        bad = _receipt(plan_id="bad-plan", authenticity={"x": False})
        bpath = bad.save(proj / ".wllm" / "receipts")
        assert cli_main(["verify", "--receipt", str(bpath)]) == 1
        assert cli_main(["apply", str(proj), "--receipt", str(bpath)]) == 1


def test_cli_plan_unknown_model_diagnose_only():
    with tempfile.TemporaryDirectory() as td:
        # only the universal reference anchor matches -> diagnose-only (3)
        rc = cli_main(["plan", td, "--model", "nobody/unknown"])
        assert rc == 3
        plan_files = list((Path(td) / ".wllm" / "plans").glob("*.json"))
        assert plan_files, "diagnose-only still records the plan evidence"
        doc = json.loads(plan_files[0].read_text())
        assert [b["backend"] for b in doc["backends"]] == ["torch-local"]


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
