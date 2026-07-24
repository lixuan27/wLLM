"""Catch the two install-path regressions reported on Modal / L40S.

Both failure modes used to silently affect every Linux user who didn't
follow the exact sequence in INSTALL.md / README.md. They are now
prevented at build / import time, so this file is the regression net:

1. ``test_kernels_so_lives_in_wllm_native`` —
   CMake must drop ``wllm_native_kernels*.so`` into ``wllm_native/`` at
   build time (via ``LIBRARY_OUTPUT_DIRECTORY``), so a plain
   ``cmake --build`` is self-sufficient. If this regresses, the user
   has to remember a manual ``cp`` or ``ninja install`` step that the
   docs don't make obvious.

2. ``test_import_without_flash_attn`` —
   ``import wllm_native`` and the default RTX Pi0 / Pi0.5 backend
   instantiation must succeed even when the upstream ``flash-attn``
   pip wheel is missing (Modal has no prebuilt wheel for
   torch 2.5.1 / CUDA 12.5 / Py3.12 → 30–60 min sdist compile per
   cold image). The vendored ``wllm_native_fa2.so`` is enough for the
   default path; ``flash-attn`` is only needed for legacy bisection
   sites.

Run:
    PYTHONPATH=. python -m pytest tests/test_install_smoke.py -v
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


def test_kernels_so_lives_in_wllm_native():
    """The compiled .so must land inside the ``wllm_native/`` package
    at build time. ``cmake --build`` alone should be sufficient — no
    follow-up ``cp`` / ``ninja install`` step.
    """
    import wllm_native
    pkg_dir = pathlib.Path(wllm_native.__file__).parent
    matches = list(pkg_dir.glob("wllm_native_kernels*.so")) \
        + list(pkg_dir.glob("wllm_native_kernels*.pyd"))
    assert matches, (
        f"wllm_native_kernels*.so not found in {pkg_dir}. "
        "Did `cmake --build` leave it under build/ instead? "
        "CMakeLists.txt must set LIBRARY_OUTPUT_DIRECTORY on the "
        "wllm_native_kernels target."
    )


def test_import_without_flash_attn(monkeypatch):
    """``import wllm_native`` must not require the upstream flash-attn
    pip package. The default RTX path uses ``wllm_native_fa2.so``; the
    upstream wheel is only needed for legacy bisection paths.
    """
    # Block flash_attn from importing — sys.modules[name] = None makes
    # `import flash_attn` raise ImportError without touching the real
    # site-packages copy (if any).
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    # Purge any cached wllm_native submodules so the import re-runs end
    # to end. Only top-level + hardware/rtx submodules need clearing —
    # those are where the offending import used to live.
    to_purge = [k for k in sys.modules if k.startswith("wllm_native")]
    for k in to_purge:
        sys.modules.pop(k, None)

    # Top-level import must succeed.
    wllm_native = importlib.import_module("wllm_native")
    assert wllm_native.__version__

    # The RTX attention backend module must also import without
    # touching upstream flash_attn. Backend instantiation is GPU-bound
    # and is exercised separately in tests/test_pi05_*.py.
    importlib.import_module("wllm_native.hardware.rtx.attn_backend")
    importlib.import_module("wllm_native.hardware.rtx.attn_backend_groot")
    importlib.import_module("wllm_native.hardware.rtx.attn_backend_groot_n17")


def test_legacy_path_raises_clear_error_without_flash_attn(monkeypatch):
    """When the user opts into the legacy upstream path
    (``FVK_RTX_FA2=0``) but ``flash-attn`` is absent, the proxy must
    surface a clear, actionable ImportError pointing at the prebuilt
    wheel page — not a generic ``ModuleNotFoundError`` from the deep
    backend init.
    """
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    to_purge = [k for k in sys.modules if k.startswith("wllm_native")]
    for k in to_purge:
        sys.modules.pop(k, None)

    from wllm.native.hardware.rtx.attn_backend import _make_flash_attn_proxy

    with pytest.raises(ImportError) as excinfo:
        _make_flash_attn_proxy(need_legacy=True)
    msg = str(excinfo.value)
    assert "FVK_RTX_FA2" in msg
    assert "flash-attention/releases" in msg
