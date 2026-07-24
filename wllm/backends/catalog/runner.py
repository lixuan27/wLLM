"""Opaque (L0) launch adapter for catalog-imported models.

Builds an OpaqueSpec for a normalized CatalogEntry so any runnable entry
can be driven through the substrate's own CLI in its own environment —
wLLM optimizes placement/replicas around it without touching model code.

Command shape (from the substrate's documented inference path):

    conda run -n <env> python -m <job_module> infer \
        --model-id <id> --prompt "..." --output-dir <dir> --device cuda

The substrate's job module and checkpoint-dir variable are installation
details, never hard-coded here: set them on :class:`SubstrateInstall`
directly or via ``WLLM_SUBSTRATE_JOB_MODULE`` / ``WLLM_SUBSTRATE_CKPT_ENV``
/ ``WLLM_SUBSTRATE_UNIFIED_ENV`` (see ``docs/ENGINES.md``).

Environment resolution honors the per-model env kind recorded in the
manifest; `unified` entries use the substrate's unified env name, which
is installation-specific.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ...backends.subprocess_cli.opaque import ArtifactSpec, OpaqueRunner, OpaqueSpec
from .importer import CatalogEntry


def _env_default(var: str, fallback: str = "") -> str:
    return os.environ.get(var, fallback).strip()


@dataclass
class SubstrateInstall:
    """Where the model substrate lives on this machine."""
    repo_root: str
    unified_env: str = field(
        default_factory=lambda: _env_default("WLLM_SUBSTRATE_UNIFIED_ENV",
                                             "substrate-unified"))
    conda_exe: str = "conda"
    ckpt_dir: str | None = None
    # Dotted module exposing the substrate's `infer` job CLI.
    job_module: str = field(
        default_factory=lambda: _env_default("WLLM_SUBSTRATE_JOB_MODULE"))
    # Name of the env var the substrate reads for its checkpoint dir.
    ckpt_env_var: str = field(
        default_factory=lambda: _env_default("WLLM_SUBSTRATE_CKPT_ENV"))


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
    if not install.job_module:
        raise RuntimeError(
            "SubstrateInstall.job_module is unset. Set it (or "
            "WLLM_SUBSTRATE_JOB_MODULE) to the substrate's job-CLI module "
            "path; see docs/ENGINES.md.")
    env_name = resolve_env_name(entry, install)
    argv = [
        install.conda_exe, "run", "--no-capture-output", "-n", env_name,
        "python", "-m", install.job_module, "infer",
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
    if install.ckpt_dir and install.ckpt_env_var:
        env[install.ckpt_env_var] = install.ckpt_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return OpaqueSpec(
        id=f"catalog:{entry.id}",
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
