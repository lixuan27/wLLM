"""Project inspector: evidence-listing discovery, absence recorded.

`wllm inspect .` walks a project root and produces a machine-readable
manifest of what was *found* (entrypoints, dependency files, model
configs, checkpoint references, GPU visibility), each with the file that
evidences it. Nothing is guessed: what could not be detected lands in
`unknowns` so downstream planning can ask instead of assume.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENTRYPOINT_GLOBS = (
    "inference*.py", "infer*.py", "demo*.py", "serve*.py", "server*.py",
    "generate*.py", "app.py", "main.py", "run*.py",
    "scripts/inference*.py", "scripts/infer*.py", "scripts/demo*.py",
    "scripts/*.sh",
)
DEPENDENCY_FILES = (
    "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "environment.yml", "Dockerfile", "setup.py",
)
MODEL_CONFIG_NAMES = ("model_index.json", "config.json")
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".wllm",
              "checkpoints", "logs"}


@dataclass
class Evidence:
    value: str
    source: str


@dataclass
class ProjectManifest:
    root: str
    entrypoints: list[Evidence] = field(default_factory=list)
    dependency_files: list[Evidence] = field(default_factory=list)
    model_configs: list[Evidence] = field(default_factory=list)
    architectures: list[Evidence] = field(default_factory=list)
    checkpoint_refs: list[Evidence] = field(default_factory=list)
    frameworks: list[Evidence] = field(default_factory=list)
    git_revision: str | None = None
    gpus: list[str] = field(default_factory=list)
    gpu_probe: str = "not-attempted"
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1))
        return p


_FRAMEWORK_MARKERS = {
    "torch": "pytorch", "diffusers": "diffusers",
    "transformers": "transformers", "jax": "jax", "flax": "jax",
    "onnxruntime": "onnx", "tensorrt": "tensorrt",
}


def _iter_files(root: Path, max_depth: int = 4):
    def walk(d: Path, depth: int):
        if depth > max_depth:
            return
        try:
            children = sorted(d.iterdir())
        except OSError:
            return
        for c in children:
            if c.name in _SKIP_DIRS or c.name.startswith("."):
                continue
            if c.is_dir():
                yield from walk(c, depth + 1)
            else:
                yield c
    yield from walk(root, 0)


def _probe_gpus(timeout_s: float = 5.0) -> tuple[list[str], str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out"
    if out.returncode != 0:
        return [], f"nvidia-smi exit {out.returncode}"
    gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return gpus, "ok"


def inspect_project(root: str | Path, probe_gpu: bool = True
                    ) -> ProjectManifest:
    rootp = Path(root).resolve()
    man = ProjectManifest(root=str(rootp))

    for pattern in ENTRYPOINT_GLOBS:
        for hit in sorted(rootp.glob(pattern)):
            rel = str(hit.relative_to(rootp))
            man.entrypoints.append(Evidence(value=rel, source=rel))

    for name in DEPENDENCY_FILES:
        f = rootp / name
        if f.is_file():
            man.dependency_files.append(Evidence(value=name, source=name))
            text = f.read_text(errors="replace")
            for marker, fw in _FRAMEWORK_MARKERS.items():
                if re.search(rf"\b{marker}\b", text):
                    if all(e.value != fw for e in man.frameworks):
                        man.frameworks.append(Evidence(value=fw, source=name))

    for f in _iter_files(rootp):
        if f.name in MODEL_CONFIG_NAMES:
            rel = str(f.relative_to(rootp))
            man.model_configs.append(Evidence(value=rel, source=rel))
            try:
                doc = json.loads(f.read_text(errors="replace"))
            except (json.JSONDecodeError, OSError):
                man.unknowns.append(f"unparseable model config: {rel}")
                continue
            for arch in (doc.get("architectures") or []):
                man.architectures.append(Evidence(value=str(arch), source=rel))
            for key in ("_name_or_path", "model_id", "pretrained_model_name_or_path"):
                val = doc.get(key)
                if isinstance(val, str) and _HF_ID.match(val):
                    man.checkpoint_refs.append(Evidence(value=val, source=rel))

    git = rootp / ".git"
    if git.exists():
        try:
            out = subprocess.run(["git", "-C", str(rootp), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            man.git_revision = out.stdout.strip() or None
        except Exception:  # noqa: BLE001
            man.git_revision = None
    if man.git_revision is None:
        man.unknowns.append("git revision unavailable")

    if probe_gpu:
        man.gpus, man.gpu_probe = _probe_gpus()
    if not man.entrypoints:
        man.unknowns.append("no entrypoint detected; user must supply a "
                            "runnable command")
    if not man.checkpoint_refs and not man.architectures:
        man.unknowns.append("no model identity detected from configs")
    return man
