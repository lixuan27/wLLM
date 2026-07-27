"""Guardrail readiness and cost accounting: fail-closed by construction."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.guardrail import (CacheRequirement, GuardrailTiming,
                                    attribute, check_hf_cache,
                                    overlap_hypothesis, repo_dirname)


# ------------------------------------------------------------- helpers
def _make_repo(root: Path, repo_id: str, rev: str = "abc123",
               files=(), *, write_ref=True, extra_rev=None):
    repo = root / "hub" / repo_dirname(repo_id)
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    for rel in files:
        p = snap / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    if write_ref:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        # No trailing newline: this is what `hf download` writes, and
        # what the loader requires (it does not strip the ref).
        (repo / "refs" / "main").write_text(rev)
    if extra_rev:
        (repo / "snapshots" / extra_rev).mkdir(parents=True, exist_ok=True)
    return repo


# ------------------------------------------------------- cache layout
def test_repo_dirname_matches_hf_cache_layout():
    assert repo_dirname("nvidia/Cosmos-1.0-Guardrail") == \
        "models--nvidia--Cosmos-1.0-Guardrail"


def test_ready_when_every_required_entry_is_present(tmp_path):
    _make_repo(tmp_path, "org/guard", files=("blocklist/nltk_data/w.txt",
                                             "face/model.pth"))
    rep = check_hf_cache(tmp_path, [
        CacheRequirement("org/guard", ("blocklist/nltk_data",
                                       "face/model.pth"))])
    assert rep.ready
    assert rep.checked == 2
    assert rep.blockers == []
    assert "org/guard" in rep.resolved


def test_empty_requirements_cannot_certify_anything(tmp_path):
    (tmp_path / "hub").mkdir()
    rep = check_hf_cache(tmp_path, [])
    assert not rep.ready
    assert rep.checked == 0
    assert "empty expectation" in rep.blockers[0]


def test_missing_hub_tree_is_a_blocker_not_a_crash(tmp_path):
    rep = check_hf_cache(tmp_path / "nope", [CacheRequirement("org/g")])
    assert not rep.ready
    assert "cache tree missing" in rep.blockers[0]


def test_missing_repo_is_named_in_the_blocker(tmp_path):
    _make_repo(tmp_path, "org/present", files=("a.txt",))
    rep = check_hf_cache(tmp_path, [
        CacheRequirement("org/present", ("a.txt",)),
        CacheRequirement("org/absent", ("b.txt",), note="prompt guard")])
    assert not rep.ready
    assert any("org/absent (prompt guard)" in b for b in rep.blockers)
    # the repo that IS there must not be smeared by its neighbour
    assert not any("org/present" in b for b in rep.blockers)


def test_missing_entry_inside_a_present_repo_is_a_blocker(tmp_path):
    _make_repo(tmp_path, "org/guard", files=("blocklist/x",))
    rep = check_hf_cache(tmp_path, [
        CacheRequirement("org/guard", ("blocklist/x", "face/model.pth"))])
    assert not rep.ready
    assert any("missing snapshot entry face/model.pth" in b
               for b in rep.blockers)


def test_partial_download_blocks_even_when_entries_exist(tmp_path):
    repo = _make_repo(tmp_path, "org/guard", files=("a.txt",))
    (repo / "blobs" / "deadbeef.incomplete").write_text("half")
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert not rep.ready
    assert any("partially downloaded" in b for b in rep.blockers)


def test_refs_main_selects_the_revision_a_loader_would_use(tmp_path):
    repo = _make_repo(tmp_path, "org/guard", rev="good",
                      files=("a.txt",), extra_rev="stale")
    (repo / "snapshots" / "stale" / "a.txt").write_text("x")
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert rep.ready
    assert rep.resolved["org/guard"].endswith("/good")


def test_missing_ref_is_refused_even_with_one_revision(tmp_path):
    # The loader resolves revisions through refs/main only; with no
    # ref it has no commit hash and fails. A "helpful" lone-snapshot
    # fallback here would certify a chain that cannot construct.
    _make_repo(tmp_path, "org/guard", rev="only", files=("a.txt",),
               write_ref=False)
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert not rep.ready
    assert any("no refs/main" in b for b in rep.blockers)


def test_dangling_ref_is_refused_and_names_the_revision(tmp_path):
    repo = _make_repo(tmp_path, "org/guard", rev="real", files=("a.txt",))
    (repo / "refs" / "main").write_text("vanished")
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert not rep.ready
    assert any("'vanished'" in b for b in rep.blockers)


def test_trailing_newline_in_ref_is_caught_and_explained(tmp_path):
    # Regression: job 202227. hub reads refs/main verbatim (f.read(),
    # no strip), so a ref written by `echo` names snapshots/<sha>\n,
    # which cannot exist -> LocalEntryNotFoundError at construction.
    # A precheck that strips would green-light this and burn a 28s
    # pipeline load plus a 56s warmup before failing.
    repo = _make_repo(tmp_path, "org/guard", rev="deadbeef",
                      files=("a.txt",))
    (repo / "refs" / "main").write_text("deadbeef\n")
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert not rep.ready
    blob = " ".join(rep.blockers)
    assert "trailing whitespace" in blob
    assert "'deadbeef\\n'" in blob or "deadbeef\\n" in blob


def test_exact_ref_without_newline_resolves(tmp_path):
    repo = _make_repo(tmp_path, "org/guard", rev="deadbeef",
                      files=("a.txt",))
    (repo / "refs" / "main").write_text("deadbeef")
    rep = check_hf_cache(tmp_path, [CacheRequirement("org/guard",
                                                     ("a.txt",))])
    assert rep.ready
    assert rep.resolved["org/guard"].endswith("/deadbeef")


# --------------------------------------------------------- accounting
def test_overhead_split_adds_up():
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=1500.0,
                                  text_stage_ms=200.0,
                                  video_stage_ms=290.0))
    assert a["overhead_ms"] == pytest.approx(500.0)
    assert a["overhead_pct"] == pytest.approx(50.0)
    assert a["attributed_ms"] == pytest.approx(490.0)
    assert a["unattributed_ms"] == pytest.approx(10.0)
    assert a["attribution_coherent"]
    assert a["significant"]


def test_large_unexplained_remainder_makes_the_split_untrustworthy():
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=1500.0,
                                  text_stage_ms=10.0,
                                  video_stage_ms=10.0))
    assert not a["attribution_coherent"]
    assert not a["significant"]
    assert overlap_hypothesis(a) is None


def test_stages_exceeding_the_measured_overhead_are_also_incoherent():
    # Instrumented segments cannot cost more than the end-to-end delta;
    # if they appear to, the wrappers are double counting.
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=1100.0,
                                  text_stage_ms=200.0,
                                  video_stage_ms=200.0))
    assert not a["attribution_coherent"]
    assert overlap_hypothesis(a) is None


def test_negative_overhead_never_becomes_a_finding():
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=950.0,
                                  text_stage_ms=0.0, video_stage_ms=0.0))
    assert a["overhead_ms"] == pytest.approx(-50.0)
    assert not a["attribution_coherent"]
    assert not a["significant"]
    assert overlap_hypothesis(a) is None


def test_small_coherent_overhead_is_not_promoted_to_a_question():
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=1050.0,
                                  text_stage_ms=20.0,
                                  video_stage_ms=25.0))
    assert a["attribution_coherent"]
    assert not a["significant"]
    assert overlap_hypothesis(a) is None


def test_hypothesis_is_labelled_unmeasured_and_claims_nothing():
    a = attribute(GuardrailTiming(baseline_ms=1000.0, guarded_ms=1500.0,
                                  text_stage_ms=200.0,
                                  video_stage_ms=290.0))
    h = overlap_hypothesis(a)
    assert h is not None
    assert h.startswith("HYPOTHESIS (unmeasured)")
    assert "No overlap was implemented or measured here." in h
    assert "ceiling" in h
    # it must not read as a delivered speedup
    for banned in ("speedup", "we achieved", "faster by"):
        assert banned not in h.lower()


def test_zero_baseline_does_not_crash_the_accounting():
    a = attribute(GuardrailTiming(baseline_ms=0.0, guarded_ms=10.0))
    assert a["overhead_pct"] != a["overhead_pct"]  # nan
    assert not a["significant"]
