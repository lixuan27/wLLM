"""Catalog importer for the WorldFoundry model substrate.

Reads catalog manifests (heterogeneous YAML generations) into one
normalized CatalogEntry shape and emits a machine-readable inventory.

Normalization policy: the YAML files are the source of truth; generated
doc indexes lag them.  Manifests span multiple schema generations, so
every field is probed along several paths and absence is recorded, never
guessed.  A manifest "claiming" integrated status is imported as a claim
(`runnable_claim`); actual runnability (support tier T1+) is only granted
by wLLM's own launch evidence later.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class CheckpointRef:
    repo_id: str
    revision: str | None = None
    license: str | None = None
    gated: bool | None = None
    private: bool | None = None
    tasks: list[str] = field(default_factory=list)


@dataclass
class CatalogEntry:
    id: str
    category: str
    path: str
    name: str = ""
    provider: str = ""
    aliases: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    availability: str | None = None
    integration_status: str | None = None
    backend_stage: str | None = None
    runner_target: str | None = None
    pipeline_target: str | None = None
    pipeline_binding: str | None = None
    runtime_profile: str | None = None
    environment: str | None = None
    checkpoints: list[CheckpointRef] = field(default_factory=list)
    variant_ids: list[str] = field(default_factory=list)
    license: str | None = None
    schema_hints: list[str] = field(default_factory=list)  # which paths matched

    @property
    def runnable_claim(self) -> bool:
        return (self.integration_status in ("integrated", "verified")
                and bool(self.runner_target or self.pipeline_binding))

    @property
    def has_open_weights(self) -> bool:
        return any(c.gated is False for c in self.checkpoints)


def _get(d: dict, *paths: str):
    """Return (value, matched_path) for the first dotted path present."""
    for p in paths:
        cur = d
        ok = True
        for key in p.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur, p
    return None, None


def _checkpoints(doc: dict) -> list[CheckpointRef]:
    refs: dict[str, CheckpointRef] = {}

    def add(repo_id, revision=None, lic=None, gated=None, private=None,
            tasks=None):
        if not repo_id or not isinstance(repo_id, str):
            return
        ref = refs.setdefault(repo_id, CheckpointRef(repo_id=repo_id))
        ref.revision = ref.revision or revision
        ref.license = ref.license or lic
        ref.gated = ref.gated if ref.gated is not None else gated
        ref.private = ref.private if ref.private is not None else private
        for t in tasks or []:
            if t not in ref.tasks:
                ref.tasks.append(t)

    ckpt = doc.get("checkpoint")
    if isinstance(ckpt, dict):
        add(ckpt.get("repo_id"), ckpt.get("revision") or ckpt.get("sha"),
            ckpt.get("license"), ckpt.get("gated"), ckpt.get("private"))
        for repo in ckpt.get("repos") or []:
            if isinstance(repo, dict):
                add(repo.get("id") or repo.get("repo_id"),
                    repo.get("sha") or repo.get("revision"),
                    repo.get("license"), repo.get("gated"),
                    repo.get("private"), repo.get("tasks"))
    for repo in (doc.get("checkpoints") or []):
        if isinstance(repo, dict):
            add(repo.get("id") or repo.get("repo_id"),
                repo.get("revision") or repo.get("sha"),
                repo.get("license"), repo.get("gated"), repo.get("private"),
                repo.get("tasks"))
    hf = (doc.get("official_sources") or {}).get("huggingface")
    for repo in hf or []:
        if isinstance(repo, dict):
            add(repo.get("repo_id"), repo.get("revision"),
                repo.get("license"))
    for var in doc.get("variants") or []:
        if isinstance(var, dict):
            for ref in var.get("checkpoint_refs") or []:
                if isinstance(ref, dict):
                    add(ref.get("repo_id"), ref.get("revision"),
                        ref.get("license"), ref.get("gated"),
                        ref.get("private"))
    return list(refs.values())


def parse_manifest(path: Path, category: str) -> CatalogEntry:
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: manifest root is not a mapping")
    hints: list[str] = []

    def probe(*paths):
        val, matched = _get(doc, *paths)
        if matched:
            hints.append(matched)
        return val

    variants = doc.get("variants") or []
    entry = CatalogEntry(
        id=str(doc.get("id") or path.stem),
        category=category,
        path=str(path),
        name=str(doc.get("name") or ""),
        provider=str(doc.get("provider") or ""),
        aliases=[str(a) for a in doc.get("aliases") or []],
        tasks=[str(t) for t in doc.get("tasks") or []],
        availability=probe("availability", "source_status.status"),
        integration_status=probe("integration.status", "integration_status",
                                 "status.integration"),
        backend_stage=probe("backend_stage", "runtime.backend_stage",
                            "integration.backend_stage"),
        runner_target=probe("runner_target", "runtime.runner_target"),
        pipeline_target=probe("pipeline_target", "runtime.pipeline_target"),
        pipeline_binding=probe("pipeline_binding"),
        runtime_profile=probe("runtime_profile", "runtime.profile"),
        environment=probe("environment", "runtime.environment",
                          "runtime.environment_name"),
        checkpoints=_checkpoints(doc),
        variant_ids=[str(v.get("id")) for v in variants
                     if isinstance(v, dict) and v.get("id")],
        license=probe("license"),
        schema_hints=hints,
    )
    return entry


class CatalogImporter:
    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root)
        self.catalog_dir = self.root / "worldfoundry/data/models/catalog"
        self.errors: list[tuple[str, str]] = []

    def commit_sha(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    def load(self) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        self.errors = []
        for yml in sorted(self.catalog_dir.glob("*/*.yaml")):
            category = yml.parent.name
            try:
                entries.append(parse_manifest(yml, category))
            except Exception as exc:  # noqa: BLE001
                self.errors.append((str(yml), repr(exc)))
        return entries

    def inventory(self, entries: list[CatalogEntry]) -> dict:
        def hist(values):
            out: dict[str, int] = {}
            for v in values:
                key = str(v) if v is not None else "(absent)"
                out[key] = out.get(key, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: -kv[1]))

        return {
            "commit_sha": self.commit_sha(),
            "total_manifests": len(entries),
            "parse_errors": self.errors,
            "by_category": hist(e.category for e in entries),
            "by_integration_status": hist(e.integration_status for e in entries),
            "by_availability": hist(e.availability for e in entries),
            "runnable_claims": sorted(e.id for e in entries if e.runnable_claim),
            "open_weight_entries": sorted(
                e.id for e in entries if e.has_open_weights),
            "gated_checkpoints": sorted({
                c.repo_id for e in entries for c in e.checkpoints if c.gated}),
            "license_histogram": hist(
                c.license for e in entries for c in e.checkpoints),
            "entries": [asdict(e) | {"runnable_claim": e.runnable_claim}
                        for e in entries],
        }

    def save_inventory(self, out_path: str | Path) -> dict:
        entries = self.load()
        inv = self.inventory(entries)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(inv, indent=1, ensure_ascii=False))
        return inv
