"""Catalog importer acceptance: every manifest parses; known entries
normalize correctly across the two observed schema generations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.backends.worldfoundry.importer import CatalogImporter

WF = ROOT / "worldfoundry-upstream"


def load_all():
    imp = CatalogImporter(WF)
    entries = imp.load()
    return imp, entries


def test_all_manifests_parse():
    imp, entries = load_all()
    assert not imp.errors, imp.errors[:5]
    assert len(entries) == 258, len(entries)


def test_categories_cover_expected():
    _, entries = load_all()
    cats = {e.category for e in entries}
    assert {"video", "world_models", "vla_va_wam",
            "three_d_four_d", "hosted_api"} <= cats


def test_openvla_normalized():
    _, entries = load_all()
    e = {x.id: x for x in entries}["openvla"]
    assert e.integration_status == "integrated"
    assert e.license == "mit"
    assert e.runner_target and "WorldFoundryPipelineRunner" in e.runner_target
    assert e.runnable_claim
    repo_ids = {c.repo_id for c in e.checkpoints}
    assert "openvla/openvla-7b" in repo_ids


def test_wan22_normalized():
    _, entries = load_all()
    e = {x.id: x for x in entries}["wan2.2"]
    repo_ids = {c.repo_id for c in e.checkpoints}
    assert "Wan-AI/Wan2.2-TI2V-5B" in repo_ids
    ti2v = next(c for c in e.checkpoints if c.repo_id == "Wan-AI/Wan2.2-TI2V-5B")
    assert ti2v.license == "apache-2.0" and ti2v.gated is False
    assert "wan2.2-ti2v-5b" in e.variant_ids
    assert e.availability is not None


def test_inventory_shape():
    imp, entries = load_all()
    inv = imp.inventory(entries)
    assert inv["total_manifests"] == 258
    assert sum(inv["by_category"].values()) == 258
    assert inv["commit_sha"] and inv["commit_sha"].startswith("bc062d7")
    assert len(inv["runnable_claims"]) >= 100, len(inv["runnable_claims"])
    # strict evidence bar: only checkpoints with an explicit gated=false count
    assert len(inv["open_weight_entries"]) >= 20, len(inv["open_weight_entries"])
    assert inv["entries"][0]["id"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:300]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
