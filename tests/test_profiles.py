"""Profile pack: evidence-gated validation, matching, and expiry."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.profiles import ModelProfile, load_profiles, match, stale_report


def _base_dict(**over) -> dict:
    d = {
        "model_family": "demo",
        "detection_ids": ["org/demo-1b"],
        "components": ["dit"],
        "runtime_support": [
            {"runtime": "wllm-serving", "tier": "launchable",
             "evidence": [{"kind": "job_log", "ref": "job 1"}]}],
        "optimizations_lossless": [
            {"name": "compile", "quality_class": "exact",
             "evidence": [{"kind": "report", "ref": "docs/x.md"}]}],
        "optimizations_bounded": [],
        "incompatibilities": ["a + b"],
        "authenticity_signals": ["ran"],
        "validation": ["parity"],
        "binding": {"hardware": "1xTEST", "last_validated": "2026-07-25",
                    "evidence_level": "measured"},
    }
    d.update(over)
    return d


# ---------------------------------------------------------------- real pack

def test_real_pack_loads_and_matches():
    profiles = load_profiles()
    assert {"wan2.2-ti2v", "qwen3-vl", "openvla",
            "cosmos3-nano", "vjepa2-vitl", "qwen3-omni"} <= set(profiles)
    for prof in profiles.values():
        assert prof.validate() == []
        assert prof.binding.evidence_level == "measured"
    # exact id and glob variant both match
    assert match(profiles, "Wan-AI/Wan2.2-TI2V-5B").model_family \
        == "wan2.2-ti2v"
    assert match(profiles, "Wan-AI/Wan2.2-TI2V-5B-Diffusers").model_family \
        == "wan2.2-ti2v"
    assert match(profiles, "nobody/mystery") is None
    # measured claims all carry evidence pointers
    wan = profiles["wan2.2-ti2v"]
    for entry in wan.optimizations_lossless + wan.optimizations_bounded:
        assert entry.evidence, entry.name
    # the batched-CFG danger lives in the trace store, not as a
    # pseudo-incompatibility that no pass name could ever trigger
    assert wan.incompatibilities == []


def test_real_pack_freshness_and_expiry():
    profiles = load_profiles()
    assert stale_report(profiles, "2026-07-27") == []
    stale = stale_report(profiles, "2027-06-01")
    assert len(stale) == len(profiles)   # everything expires eventually


# ------------------------------------------------------------- fail-closed

def test_validation_rejects_bad_vocabularies():
    bad_tier = ModelProfile.from_dict(_base_dict(
        runtime_support=[{"runtime": "x", "tier": "vibes"}]))
    assert any("unknown tier" in e for e in bad_tier.validate())
    bad_quality = ModelProfile.from_dict(_base_dict(
        optimizations_lossless=[{"name": "q", "quality_class": "magic"}]))
    assert any("unknown quality_class" in e for e in bad_quality.validate())
    misfiled = ModelProfile.from_dict(_base_dict(
        optimizations_lossless=[{
            "name": "fp8", "quality_class": "bounded",
            "evidence": [{"kind": "report", "ref": "r"}]}]))
    assert any("must not sit in the lossless list" in e
               for e in misfiled.validate())


def test_measured_without_evidence_is_invalid():
    naked = ModelProfile.from_dict(_base_dict(
        optimizations_lossless=[{"name": "compile",
                                 "quality_class": "exact"}]))
    assert any("no evidence refs" in e for e in naked.validate())
    weak_runtime = ModelProfile.from_dict(_base_dict(
        runtime_support=[{"runtime": "x", "tier": "optimized"}]))
    assert any("no evidence refs" in e for e in weak_runtime.validate())
    # a merely "declared" profile may make the same claims without refs
    declared = ModelProfile.from_dict(_base_dict(
        optimizations_lossless=[{"name": "compile",
                                 "quality_class": "exact"}],
        binding={"last_validated": "2026-07-25",
                 "evidence_level": "declared"}))
    assert declared.validate() == []


def test_bad_dates_and_unknown_fields():
    bad_date = ModelProfile.from_dict(_base_dict(
        binding={"last_validated": "2026-13-99",
                 "evidence_level": "measured"}))
    assert any("not a real" in e for e in bad_date.validate())
    assert bad_date.is_stale("2026-07-25")     # unparseable == stale
    try:
        ModelProfile.from_dict(_base_dict(turbo=True))
    except ValueError as exc:
        assert "turbo" in str(exc)
    else:
        raise AssertionError("unknown profile fields must be rejected")


def test_store_rejects_broken_pack(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "ok.yaml").write_text(
            "model_family: ok\ndetection_ids: [a/b]\n"
            "binding: {last_validated: '2026-07-25'}\n")
        (p / "broken.yaml").write_text(
            "model_family: broken\ndetection_ids: []\n"
            "binding: {last_validated: 'yesterday'}\n")
        try:
            load_profiles(p)
        except ValueError as exc:
            msg = str(exc)
            assert "broken.yaml" in msg and "detection_ids is empty" in msg
        else:
            raise AssertionError("a broken pack must reject the whole load")
        (p / "broken.yaml").unlink()
        (p / "dup.yaml").write_text(
            "model_family: ok\ndetection_ids: [c/d]\n"
            "binding: {last_validated: '2026-07-25'}\n")
        try:
            load_profiles(p)
        except ValueError as exc:
            assert "duplicate model_family" in str(exc)
        else:
            raise AssertionError("duplicate families must be rejected")


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
