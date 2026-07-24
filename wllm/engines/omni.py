"""Late binding to the external omni-modal serving engine.

Several apps delegate stage execution (async stage engines, talker
modules, paged-attention kernels, codec predictors) to an external
omni-modal serving engine. The engine is not vendored and not imported
by name anywhere in this tree: set ``WLLM_OMNI_ENGINE`` to the
installed engine's package name before launching a stage that needs it
(see ``docs/ENGINES.md``).

Stage-config YAMLs shipped with the apps reference engine-internal
classes through the ``__WLLM_OMNI_ENGINE__`` placeholder; call
:func:`render_stage_config` right before handing a config path to the
engine so the placeholder is substituted with the bound package name.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from types import ModuleType

ENV_VAR = "WLLM_OMNI_ENGINE"
PLACEHOLDER = "__WLLM_OMNI_ENGINE__"


class OmniEngineNotBound(ImportError):
    """Raised when an app stage needs the omni engine but none is bound."""


def package_name() -> str:
    """Name of the bound engine package, from ``WLLM_OMNI_ENGINE``."""
    name = os.environ.get(ENV_VAR, "").strip()
    if not name:
        raise OmniEngineNotBound(
            f"This stage needs the external omni serving engine. Set "
            f"{ENV_VAR} to the installed engine's package name "
            f"(see docs/ENGINES.md)."
        )
    return name


def package() -> ModuleType:
    """Import and return the bound engine package."""
    return importlib.import_module(package_name())


def submodule(dotted: str) -> ModuleType:
    """Import ``<engine>.<dotted>`` and return the module."""
    return importlib.import_module(f"{package_name()}.{dotted}")


def attr(dotted: str, name: str):
    """Return ``name`` from ``<engine>.<dotted>``."""
    return getattr(submodule(dotted), name)


def async_engine_cls():
    """The engine's async multi-stage entrypoint class."""
    return getattr(package(), "AsyncOmni")


def render_stage_config(path: str | os.PathLike, run_dir: str | None = None) -> str:
    """Substitute the engine-package placeholder in a stage-config YAML.

    Returns the original path unchanged when it contains no placeholder;
    otherwise writes the rendered file next to ``run_dir`` (or a temp
    dir) and returns the rendered path.
    """
    src = Path(path)
    text = src.read_text()
    if PLACEHOLDER not in text:
        return str(src)
    rendered = text.replace(PLACEHOLDER, package_name())
    out_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="wllm_stagecfg_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / src.name
    out.write_text(rendered)
    return str(out)
