"""Catalog importer acceptance.

Runs entirely against synthetic manifests replicating the two schema
generations observed in real substrate checkouts, so it needs no local
substrate. When ``WLLM_SUBSTRATE_ROOT`` points at a real checkout, an
extra integration test additionally requires every real manifest to
parse cleanly.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.backends.catalog.importer import CatalogImporter, find_catalog_dir

# Generation A: integration.status + runtime.runner_target + checkpoint.repos
GEN_A = """\
id: demo-vla
name: Demo VLA
provider: demo-lab
tasks: [manipulation]
license: mit
integration:
  status: integrated
runtime:
  runner_target: substrate.evaluation.models.runners.pipeline:PipelineRunner
  environment: _unified
checkpoint:
  repos:
    - id: demo/demo-vla-7b
      license: mit
      gated: false
"""

# Generation B: top-level availability + variants with checkpoint_refs
GEN_B = """\
id: demo-video
name: Demo Video
availability: released
variants:
  - id: demo-video-5b
    checkpoint_refs:
      - repo_id: DemoAI/Demo-Video-5B
        license: apache-2.0
        gated: false
  - id: demo-video-14b
    checkpoint_refs:
      - repo_id: DemoAI/Demo-Video-14B
        gated: true
"""

BROKEN = ":\n:this is not yaml: [\n"


def _make_tree(td: Path) -> Path:
    cat = td / "pkg" / "data" / "models" / "catalog"
    (cat / "vla_va_wam").mkdir(parents=True)
    (cat / "video").mkdir(parents=True)
    (cat / "vla_va_wam" / "demo-vla.yaml").write_text(GEN_A)
    (cat / "video" / "demo-video.yaml").write_text(GEN_B)
    return td


def load_all():
    with tempfile.TemporaryDirectory() as td:
        root = _make_tree(Path(td))
        imp = CatalogImporter(root)
        entries = imp.load()
        return imp, entries


def test_catalog_dir_discovery():
    with tempfile.TemporaryDirectory() as td:
        root = _make_tree(Path(td))
        found = find_catalog_dir(root)
        assert found.is_dir() and found.name == "catalog"
        # pointing directly at the package works too
        assert find_catalog_dir(root / "pkg") == found
        # a root with no catalog yields a deterministic non-existent path
        with tempfile.TemporaryDirectory() as empty:
            assert not find_catalog_dir(Path(empty)).is_dir()


def test_all_synthetic_manifests_parse():
    imp, entries = load_all()
    assert not imp.errors, imp.errors
    assert len(entries) == 2
    assert {e.category for e in entries} == {"vla_va_wam", "video"}


def test_generation_a_normalized():
    _, entries = load_all()
    e = {x.id: x for x in entries}["demo-vla"]
    assert e.integration_status == "integrated"
    assert e.license == "mit"
    assert e.runner_target and e.runner_target.endswith(":PipelineRunner")
    assert e.environment == "_unified"
    assert e.runnable_claim
    assert {c.repo_id for c in e.checkpoints} == {"demo/demo-vla-7b"}
    assert e.has_open_weights


def test_generation_b_normalized():
    _, entries = load_all()
    e = {x.id: x for x in entries}["demo-video"]
    assert e.availability == "released"
    assert not e.runnable_claim          # no runner target -> claim denied
    repo_ids = {c.repo_id for c in e.checkpoints}
    assert repo_ids == {"DemoAI/Demo-Video-5B", "DemoAI/Demo-Video-14B"}
    lic = {c.repo_id: c.license for c in e.checkpoints}
    assert lic["DemoAI/Demo-Video-5B"] == "apache-2.0"
    gated = {c.repo_id: c.gated for c in e.checkpoints}
    assert gated["DemoAI/Demo-Video-14B"] is True
    assert e.variant_ids == ["demo-video-5b", "demo-video-14b"]


def test_parse_errors_recorded_not_raised():
    with tempfile.TemporaryDirectory() as td:
        root = _make_tree(Path(td))
        cat = find_catalog_dir(root)
        (cat / "video" / "broken.yaml").write_text(BROKEN)
        imp = CatalogImporter(root)
        entries = imp.load()
        assert len(entries) == 2            # good manifests still load
        assert len(imp.errors) == 1
        assert "broken.yaml" in imp.errors[0][0]


def test_inventory_shape():
    imp, entries = load_all()
    inv = imp.inventory(entries)
    assert inv["total_manifests"] == 2
    assert sum(inv["by_category"].values()) == 2
    assert inv["runnable_claims"] == ["demo-vla"]
    assert "demo-video" in inv["open_weight_entries"]
    assert inv["gated_checkpoints"] == ["DemoAI/Demo-Video-14B"]
    assert len(inv["entries"]) == 2


def test_real_substrate_if_present():
    root = os.environ.get("WLLM_SUBSTRATE_ROOT", "")
    if not root or not Path(root).is_dir():
        return  # optional integration check
    imp = CatalogImporter(root)
    entries = imp.load()
    assert not imp.errors, imp.errors[:5]
    assert len(entries) >= 200, len(entries)


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
