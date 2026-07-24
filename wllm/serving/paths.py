import os


def repo_root() -> str:
    """Absolute path to the repository root (the directory containing ``wllm``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_dir(app: str) -> str:
    """Absolute path to ``wllm/apps/<app>``."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps", app)
