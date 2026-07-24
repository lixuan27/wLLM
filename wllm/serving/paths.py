import os


def repo_root() -> str:
    """Absolute path to the repository root (the directory containing ``wllm``).

    This file lives at ``wllm/serving/paths.py``, so the root is three
    levels up (serving -> wllm -> root).
    """
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def app_dir(app: str) -> str:
    """Absolute path to ``wllm/apps/<app>`` (apps live beside, not inside,
    the serving subsystem since the unification)."""
    return os.path.join(repo_root(), "wllm", "apps", app)
