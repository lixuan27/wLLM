"""L0 opaque integration: wrap any existing CLI / server / runner process.

The lowest-friction entry: the model's own launch command is the unit of
execution.  wLLM can then optimize process/GPU placement, replicas, and
multi-model pipelines without seeing inside.  Process hygiene is built in:
every launch is wrapped in try/finally kill of the whole process group,
and a post-run liveness check guarantees no orphans survive the call.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtifactSpec:
    kind: str                 # e.g. "generated_video", "action_trace"
    path_template: str        # may contain {output_dir}, {sample_id}


@dataclass
class OpaqueSpec:
    """Declarative description of one opaque runnable."""

    id: str
    argv: list[str]           # tokens may contain {placeholders}
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_s: float = 3600.0
    gpu_indices: list[int] = field(default_factory=list)  # -> CUDA_VISIBLE_DEVICES
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    ready_line: str | None = None   # for server-style processes (future)


@dataclass
class OpaqueResult:
    spec_id: str
    status: str               # ok | error | timeout
    returncode: int | None
    wall_seconds: float
    artifacts: dict[str, str] = field(default_factory=dict)  # kind -> path
    missing_artifacts: list[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _fill(template: str, params: dict[str, str]) -> str:
    out = template
    for key, val in params.items():
        out = out.replace("{" + key + "}", str(val))
    return out


class OpaqueRunner:
    def __init__(self, spec: OpaqueSpec):
        self.spec = spec

    def run(self, params: dict[str, str] | None = None,
            tail_bytes: int = 4096) -> OpaqueResult:
        params = dict(params or {})
        argv = [_fill(tok, params) for tok in self.spec.argv]
        env = os.environ.copy()
        env.update({k: _fill(v, params) for k, v in self.spec.env.items()})
        if self.spec.gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in self.spec.gpu_indices)

        start = time.monotonic()
        proc: subprocess.Popen | None = None
        status, rc, err = "ok", None, ""
        out_b = err_b = b""
        try:
            proc = subprocess.Popen(
                argv, env=env, cwd=self.spec.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True)   # own pgid -> killable as a tree
            try:
                out_b, err_b = proc.communicate(timeout=self.spec.timeout_s)
                rc = proc.returncode
                if rc != 0:
                    status, err = "error", f"exit code {rc}"
            except subprocess.TimeoutExpired:
                status, err = "timeout", f"exceeded {self.spec.timeout_s}s"
        except OSError as exc:
            status, err = "error", repr(exc)
        finally:
            if proc is not None:
                self._kill_tree(proc)
        wall = time.monotonic() - start

        artifacts: dict[str, str] = {}
        missing: list[str] = []
        for art in self.spec.artifacts:
            path = _fill(art.path_template, params)
            if Path(path).exists():
                artifacts[art.kind] = path
            else:
                missing.append(art.kind)
        if status == "ok" and missing:
            status = "error"
            err = f"missing artifacts: {missing}"

        return OpaqueResult(
            spec_id=self.spec.id, status=status, returncode=rc,
            wall_seconds=wall, artifacts=artifacts,
            missing_artifacts=missing,
            stdout_tail=out_b[-tail_bytes:].decode(errors="replace"),
            stderr_tail=err_b[-tail_bytes:].decode(errors="replace"),
            error=err)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Terminate the whole process group; verify it is gone."""
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        # reap and confirm
        assert proc.poll() is not None, "opaque subprocess still alive"
