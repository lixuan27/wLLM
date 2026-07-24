"""Checkpoint asset readiness: content-level checks, explicit blockers.

Mirrors and proxies ship broken files, so presence on disk proves
nothing. What this module can and cannot certify, honestly:

* catches: missing files, proxy stub bodies (~15 B "Entry not found"),
  files below a declared minimum size, files deviating from a declared
  **exact** size (`expected_bytes`, when the caller knows it from an
  upstream index), and JSON that fails to parse at the syntax level.
* does NOT catch: mid-file binary corruption at unchanged length,
  tensor-level damage inside weight shards (no checksum or shard-header
  validation yet), or JSON whose corruption still parses. Truncation is
  detected only when `expected_bytes` is declared.

Anything failed lands in ``blockers`` with the exact file and reason;
nothing is guessed, nothing silently passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Bodies smaller than this are suspicious for weight shards ("Entry not
# found" proxy stubs are ~15 bytes).
STUB_THRESHOLD_BYTES = 1024


@dataclass
class ExpectedFile:
    relpath: str
    min_bytes: int = 1
    expected_bytes: int | None = None   # exact size from an upstream index
    must_parse_json: bool = False


@dataclass
class AssetSpec:
    name: str
    root: str
    expected: list[ExpectedFile] = field(default_factory=list)


@dataclass
class ReadinessReport:
    name: str
    ready: bool
    checked: int
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "ready": self.ready,
                "checked": self.checked, "blockers": self.blockers}


def check_asset(spec: AssetSpec) -> ReadinessReport:
    root = Path(spec.root)
    blockers: list[str] = []
    if not spec.expected:
        blockers.append("asset spec lists no expected files; an empty "
                        "expectation cannot certify readiness")
    if not root.is_dir():
        blockers.append(f"asset root missing: {root}")
        return ReadinessReport(spec.name, False, 0, blockers)
    for exp in spec.expected:
        path = root / exp.relpath
        if not path.is_file():
            blockers.append(f"missing file: {exp.relpath}")
            continue
        size = path.stat().st_size
        if size < exp.min_bytes:
            blockers.append(
                f"short file: {exp.relpath} is {size} B "
                f"(< {exp.min_bytes} B expected; possible proxy stub)")
            continue
        if exp.expected_bytes is not None and size != exp.expected_bytes:
            blockers.append(
                f"size mismatch: {exp.relpath} is {size} B, upstream "
                f"index declares {exp.expected_bytes} B (truncated or "
                f"partially downloaded)")
            continue
        if exp.must_parse_json:
            try:
                json.loads(path.read_text(errors="replace"))
            except json.JSONDecodeError as exc:
                blockers.append(
                    f"corrupt json: {exp.relpath} ({exc.msg} at "
                    f"line {exc.lineno})")
    return ReadinessReport(spec.name, ready=not blockers,
                           checked=len(spec.expected), blockers=blockers)


def spec_for_hf_layout(name: str, root: str, weight_shards: list[str],
                       ) -> AssetSpec:
    """Conventional transformer-checkpoint expectation set."""
    expected = [
        ExpectedFile("config.json", min_bytes=2, must_parse_json=True),
    ]
    for shard in weight_shards:
        expected.append(ExpectedFile(shard, min_bytes=STUB_THRESHOLD_BYTES))
    return AssetSpec(name=name, root=root, expected=expected)
