"""L0 opaque runner + L1 Application API + profiler tests."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wllm.api import Application, infer_modality
from wllm.backends.subprocess_cli.opaque import (
    ArtifactSpec, OpaqueRunner, OpaqueSpec,
)
from wllm.graph.streams import Modality
from wllm.profiling.report import profile_thunk


def test_opaque_runner_ok_and_artifacts():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/out.txt"
        spec = OpaqueSpec(
            id="echo", argv=["/bin/sh", "-c", "echo hello > {outfile}"],
            timeout_s=20,
            artifacts=[ArtifactSpec(kind="text", path_template="{outfile}")])
        res = OpaqueRunner(spec).run({"outfile": out})
        assert res.ok, (res.status, res.error, res.stderr_tail)
        assert res.artifacts["text"] == out
        assert res.wall_seconds < 10


def test_opaque_runner_missing_artifact_and_bad_exit():
    spec = OpaqueSpec(id="fail", argv=["/bin/sh", "-c", "exit 3"],
                      timeout_s=20)
    res = OpaqueRunner(spec).run()
    assert res.status == "error" and res.returncode == 3

    with tempfile.TemporaryDirectory() as td:
        spec2 = OpaqueSpec(
            id="noart", argv=["/bin/true"], timeout_s=20,
            artifacts=[ArtifactSpec(kind="video",
                                    path_template=f"{td}/missing.mp4")])
        res2 = OpaqueRunner(spec2).run()
        assert res2.status == "error"
        assert res2.missing_artifacts == ["video"]


def test_opaque_runner_timeout_kills_tree():
    spec = OpaqueSpec(id="sleepy", argv=["/bin/sh", "-c", "sleep 60"],
                      timeout_s=1.0)
    res = OpaqueRunner(spec).run()
    assert res.status == "timeout"
    # no orphan: the shell's pgid is gone (poll asserted inside _kill_tree)


def test_modality_inference():
    assert infer_modality("a prompt") == Modality.TOKEN
    assert infer_modality("clip.mp4") == Modality.FRAME
    assert infer_modality("sound.wav") == Modality.AUDIO
    assert infer_modality([0.1, 0.2]) == Modality.ACTION
    assert infer_modality(3) == Modality.CONTROL


def test_application_from_callable_and_baseline():
    calls = []

    def run(prompt: str, video: str = "in.mp4"):
        calls.append(prompt)
        return {"result": f"{prompt}:{video}"}

    app = Application.from_callable(
        run, example_inputs={"prompt": "hi", "video": "in.mp4"})
    assert app.program.validate() == []

    report = app.baseline(repeats=3, warmup=1)
    assert report.repeats == 3 and len(report.wall_ms) == 3
    assert report.median_ms >= 0.0
    assert len(calls) == 4  # 1 warmup + 3 measured

    out = app.reference_run(prompt="bye")
    assert out == {"result": "bye:in.mp4"}


def test_application_optimize_l1_is_honest():
    def run(prompt: str):
        return prompt

    app = Application.from_callable(run, example_inputs={"prompt": "x"})
    plans = app.optimize(num_gpus=4)
    ids = {p.id for p in plans.plans}
    # single opaque node: placement-only surface, no fabricated parallelism
    assert "baseline_1gpu" in ids
    assert not any(i.startswith("disagg") for i in ids)
    assert plans.best() is not None
    assert "kept" in plans.report() or "keep" in plans.report()


def test_profiler_report_save():
    with tempfile.TemporaryDirectory() as td:
        rep = profile_thunk("noop", lambda: None, repeats=4, warmup=0)
        path = rep.save(td, tag="t")
        assert path.exists() and path.stat().st_size > 50
        assert rep.p95_ms >= rep.median_ms >= 0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:200]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
