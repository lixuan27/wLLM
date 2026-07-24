"""Opaque (L0) launch adapter for catalog-imported models.

Builds an OpaqueSpec for a normalized CatalogEntry so any runnable entry
can be driven through the substrate's own CLI in its own environment —
wLLM optimizes placement/replicas around it without touching model code.

Command shape (from the substrate's documented inference path):

    conda run -n <env> python -m worldfoundry.studio.workspace_job infer \
        --model-id <id> --prompt "..." --output-dir <dir> --device cuda

Environment resolution honors the per-model env kind recorded in the
manifest; `unified` entries use the substrate's unified env name, which
the caller supplies (it is installation-specific).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...backends.subprocess_cli.opaque import ArtifactSpec, OpaqueRunner, OpaqueSpec
from .importer import CatalogEntry


@dataclass
class SubstrateInstall:
    """Where the model substrate lives on this machine."""
    repo_root: str
    unified_env: str = "worldfoundry-unified-cu128"
    conda_exe: str = "conda"
    ckpt_dir: str | None = None      # WORLDFOUNDRY_CKPT_DIR, if used


def resolve_env_name(entry: CatalogEntry, install: SubstrateInstall) -> str:
    env = (entry.environment or "").strip()
    if not env or env in ("_unified", "unified"):
        return install.unified_env
    return env


def build_infer_spec(
    entry: CatalogEntry,
    install: SubstrateInstall,
    *,
    output_dir: str,
    prompt: str = "",
    input_path: str | None = None,
    extra_args: list[str] | None = None,
    gpu_indices: list[int] | None = None,
    timeout_s: float = 1800.0,
) -> OpaqueSpec:
    """One generation request as a self-contained subprocess launch."""
    env_name = resolve_env_name(entry, install)
    argv = [
        install.conda_exe, "run", "--no-capture-output", "-n", env_name,
        "python", "-m", "worldfoundry.studio.workspace_job", "infer",
        "--model-id", entry.id,
        "--output-dir", output_dir,
        "--device", "cuda",
    ]
    if prompt:
        argv += ["--prompt", prompt]
    if input_path:
        argv += ["--input-path", input_path]
    argv += list(extra_args or [])

    env = {"PYTHONUNBUFFERED": "1"}
    if install.ckpt_dir:
        env["WORLDFOUNDRY_CKPT_DIR"] = install.ckpt_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return OpaqueSpec(
        id=f"wf:{entry.id}",
        argv=argv,
        env=env,
        cwd=install.repo_root,
        timeout_s=timeout_s,
        gpu_indices=gpu_indices or [],
        artifacts=[ArtifactSpec(kind="output_dir", path_template=output_dir)],
    )


def launch(entry: CatalogEntry, install: SubstrateInstall, **kw):
    """Convenience: build the spec and run it, returning the OpaqueResult."""
    spec = build_infer_spec(entry, install, **kw)
    return OpaqueRunner(spec).run()
