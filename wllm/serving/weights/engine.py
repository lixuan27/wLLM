"""Download-and-convert engine for ``checkpoints/``.

Each app declares the components it needs in
``wllm/apps/<app>/weights.py``; this engine fetches whatever is not
already present. Components shared between apps point at the same target
directory, so they download once and are reused (a symlinked directory
counts as present). All sources are official repos; the only processing
is for models whose official release is a training-state pickle, which
are converted to a plain safetensors + config.json on download (the
FastVideo-style checkpoint-conversion approach).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from typing import Iterable, List

from wllm.serving.weights.components import Component

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINTS_DIR = os.path.join(_REPO_ROOT, "checkpoints")


def app_components(app: str) -> List[Component]:
    mod = importlib.import_module(f"wllm.apps.{app}.weights")
    return list(mod.COMPONENTS)


def merge_components(apps: Iterable[str]) -> List[Component]:
    """Deduplicate by target across apps; identical targets must carry
    identical descriptors (the cross-app reuse contract)."""
    by_target: dict[str, Component] = {}
    for app in apps:
        for comp in app_components(app):
            prev = by_target.get(comp.target)
            if prev is None:
                by_target[comp.target] = comp
            elif prev != comp:
                raise ValueError(
                    f"conflicting descriptors for checkpoints/{comp.target}: "
                    f"{prev} vs {comp}"
                )
    return list(by_target.values())


def is_present(comp: Component) -> bool:
    path = os.path.join(CHECKPOINTS_DIR, comp.target)
    return os.path.isdir(path) and bool(os.listdir(path))


def _clean_staging_metadata(path: str) -> None:
    cache = os.path.join(path, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)


def _run_convert(comp: Component, staging: str, target_path: str) -> None:
    kind, source_file, cast, cfg_items = comp.convert
    if kind != "generator_pt":
        raise ValueError(f"unknown conversion kind {kind!r}")
    from wllm.serving.weights.convert import generator_pt_to_safetensors

    config = {k: json.loads(v) for k, v in cfg_items}
    generator_pt_to_safetensors(
        os.path.join(staging, source_file), target_path, cast or None, config
    )


def ensure(comp: Component, quiet: bool = False) -> bool:
    """Make checkpoints/<target> exist. Returns True if work was done."""
    target_path = os.path.join(CHECKPOINTS_DIR, comp.target)
    if is_present(comp):
        if not quiet:
            print(f"[weights] present: checkpoints/{comp.target}")
        return False

    from huggingface_hub import snapshot_download

    staging = os.path.join(
        CHECKPOINTS_DIR, ".staging", comp.target.replace("/", "_")
    )
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    print(f"[weights] downloading {comp.repo} ({', '.join(comp.patterns)}) "
          f"-> checkpoints/{comp.target}", flush=True)
    snapshot_download(comp.repo, allow_patterns=list(comp.patterns), local_dir=staging)
    _clean_staging_metadata(staging)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if comp.convert is not None:
        print(f"[weights] converting to checkpoints/{comp.target} "
              "(one-time; the downloaded pickle is deleted afterwards)", flush=True)
        _run_convert(comp, staging, target_path)
    else:
        source = staging if comp.rename_from is None else os.path.join(staging, comp.rename_from)
        _clean_staging_metadata(source)
        shutil.move(source, target_path)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[weights] done: checkpoints/{comp.target}", flush=True)
    return True


def ensure_all(apps: List[str], dry_run: bool = False) -> None:
    comps = merge_components(apps)
    missing = [c for c in comps if not is_present(c)]
    for c in comps:
        status = "present" if is_present(c) else "missing"
        print(f"  checkpoints/{c.target:<28} {status:<8} {c.note}")
    if not missing:
        print("[weights] everything present; nothing to download")
        return
    if dry_run:
        return
    for c in comps:
        ensure(c)
