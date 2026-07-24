"""Asset readiness: content-level validation, explicit blockers."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.backends.catalog.assets import (
    AssetSpec, ExpectedFile, check_asset, spec_for_hf_layout,
)


def _stage(td: Path, files: dict[str, bytes]) -> Path:
    for rel, body in files.items():
        p = td / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return td


def test_ready_when_all_files_pass():
    with tempfile.TemporaryDirectory() as td:
        root = _stage(Path(td), {
            "config.json": b'{"architectures": ["DemoDiT"]}',
            "model-00001.safetensors": b"x" * 4096,
        })
        spec = spec_for_hf_layout("demo", str(root),
                                  ["model-00001.safetensors"])
        rep = check_asset(spec)
        assert rep.ready and rep.blockers == [] and rep.checked == 2


def test_blockers_are_specific():
    with tempfile.TemporaryDirectory() as td:
        root = _stage(Path(td), {
            "config.json": b'{"architectures": ["DemoDiT"',   # corrupt
            "model-00001.safetensors": b"Entry not found",    # proxy stub
        })
        spec = spec_for_hf_layout("demo", str(root),
                                  ["model-00001.safetensors",
                                   "model-00002.safetensors"])
        rep = check_asset(spec)
        assert not rep.ready
        joined = " ".join(rep.blockers)
        assert "corrupt json: config.json" in joined
        assert "short file: model-00001.safetensors" in joined
        assert "proxy stub" in joined
        assert "missing file: model-00002.safetensors" in joined


def test_missing_root_and_empty_expectation_fail_closed():
    rep = check_asset(AssetSpec("ghost", "/nonexistent/path",
                                [ExpectedFile("a")]))
    assert not rep.ready and any("root missing" in b for b in rep.blockers)
    with tempfile.TemporaryDirectory() as td:
        rep2 = check_asset(AssetSpec("empty", td, []))
        assert not rep2.ready
        assert any("no expected files" in b for b in rep2.blockers)


def test_report_serializes():
    with tempfile.TemporaryDirectory() as td:
        rep = check_asset(AssetSpec("x", td, [ExpectedFile("missing.bin")]))
        d = rep.to_dict()
        assert d["name"] == "x" and d["ready"] is False and d["blockers"]


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
