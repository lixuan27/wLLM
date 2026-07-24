"""In-tree omni engine: config parsing, schedulers, batching parity, engine."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.omni import AsyncOmni, ModelNotSupported, SamplingParams
from wllm.omni.core.sched.base import ScheduledRequest
from wllm.omni.core.sched.omni_ar_scheduler import OmniARScheduler
from wllm.omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from wllm.omni.stage_config import load_stage_config, resolve_scheduler
from wllm.omni.stages import EchoStage, create_stage

STAGE_YAML = """\
async_chunk: false
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime: {devices: "0"}
    engine_args:
      model_stage_backend: echo
      max_num_seqs: 8
      scheduler_cls: __WLLM_OMNI_ENGINE__.core.sched.omni_ar_scheduler.OmniARScheduler
      engine_output_type: latent
    final_output: true
    final_output_type: text
"""

TWO_STAGE_YAML = """\
stage_args:
  - stage_id: 0
    stage_type: codec
    engine_args:
      model_stage_backend: echo
      scheduler_cls: __WLLM_OMNI_ENGINE__.core.sched.omni_generation_scheduler.OmniGenerationScheduler
  - stage_id: 1
    stage_type: llm
    engine_args:
      model_stage_backend: echo
      scheduler_cls: __WLLM_OMNI_ENGINE__.core.sched.omni_ar_scheduler.OmniARScheduler
    final_output: true
"""


def _write(tmp, text, name="cfg.yaml"):
    p = Path(tmp) / name
    p.write_text(text)
    return str(p)


# ------------------------------------------------------------ stage config

def test_stage_config_parses_and_placeholder_resolves():
    with tempfile.TemporaryDirectory() as td:
        cfg = load_stage_config(_write(td, STAGE_YAML))
        assert not cfg.async_chunk and len(cfg.stages) == 1
        st = cfg.final_stage()
        assert st.stage_type == "llm" and st.devices == "0"
        cls = resolve_scheduler(st.scheduler_path)
        assert cls is OmniARScheduler


def test_stage_config_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        for bad, frag in [
            ("stage_args: []", "empty"),
            ("stage_args:\n  - {stage_type: warp}", "unknown stage_type"),
            ("stage_args:\n  - {stage_type: llm}", "final_output"),
            ("stage_args:\n"
             "  - {stage_type: llm, final_output: true}\n"
             "  - {stage_type: llm, final_output: true}", "final_output"),
        ]:
            try:
                load_stage_config(_write(td, bad))
            except ValueError as exc:
                assert frag.split()[0] in str(exc).lower() or frag in str(exc)
            else:
                raise AssertionError(f"config must be rejected: {bad!r}")
    try:
        resolve_scheduler("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty scheduler path must be rejected")


# -------------------------------------------------------------- schedulers

def test_ar_scheduler_continuous_batching():
    sched = OmniARScheduler(max_num_seqs=4)
    stage = EchoStage()
    a = ScheduledRequest("a", stage.tokenize("one"),
                         SamplingParams(max_tokens=3, seed=1))
    b = ScheduledRequest("b", stage.tokenize("two two"),
                         SamplingParams(max_tokens=5, seed=2))
    sched.add(a)
    sched.step(stage.decode_batch)          # only a running
    sched.add(b)                            # joins mid-flight
    while sched.has_work:
        sched.step(stage.decode_batch)
    assert a.finish_reason == "length" and len(a.output_token_ids) == 3
    assert b.finish_reason == "length" and len(b.output_token_ids) == 5
    assert sched.stats.max_step_batch == 2   # continuous batching happened
    assert sched.stats.completed == 2


def test_ar_scheduler_parity_under_batching():
    """A request's tokens are identical alone vs batched with others."""
    stage = EchoStage()
    def run(requests):
        sched = OmniARScheduler(max_num_seqs=8)
        for r in requests:
            sched.add(r)
        while sched.has_work:
            sched.step(stage.decode_batch)
    solo = ScheduledRequest("solo", stage.tokenize("hello world"),
                            SamplingParams(max_tokens=4, seed=7))
    run([solo])
    together = ScheduledRequest("together", stage.tokenize("hello world"),
                                SamplingParams(max_tokens=4, seed=7))
    noise = [ScheduledRequest(f"n{i}", stage.tokenize(f"noise {i}"),
                              SamplingParams(max_tokens=6, seed=i))
             for i in range(3)]
    run([together] + noise)
    assert together.output_token_ids == solo.output_token_ids


def test_generation_scheduler_whole_request():
    sched = OmniGenerationScheduler(max_num_seqs=8)
    stage = EchoStage()
    reqs = [ScheduledRequest(f"g{i}", stage.tokenize(f"p {i}"))
            for i in range(3)]
    for r in reqs:
        sched.add(r)
    finished = sched.step(stage.generate_batch)
    assert len(finished) == 3 and sched.stats.max_step_batch == 3
    for r in reqs:
        assert r.finished and "checksum" in r.context["result"]


def test_scheduler_crash_fails_batch_and_stays_usable():
    sched = OmniARScheduler(max_num_seqs=4)
    a = ScheduledRequest("a", [1])
    b = ScheduledRequest("b", [2])
    sched.add(a)
    sched.add(b)

    def boom(batch):
        raise RuntimeError("model died")

    try:
        sched.step(boom)
    except RuntimeError:
        pass
    else:
        raise AssertionError("step must re-raise the model crash")
    # the poisoned batch is failed AND retired — no resident re-crasher
    assert not sched.running
    assert a.finished and a.finish_reason == "error"
    assert "model died" in a.context["error"]
    # the scheduler still serves new requests afterwards
    stage = EchoStage()
    c = ScheduledRequest("c", stage.tokenize("fresh"),
                         SamplingParams(max_tokens=2))
    sched.add(c)
    while sched.has_work:
        sched.step(stage.decode_batch)
    assert c.finish_reason == "length"


def test_scheduler_abort_removes_everywhere():
    sched = OmniARScheduler(max_num_seqs=1)
    a = ScheduledRequest("a", [1], SamplingParams(max_tokens=1))
    b = ScheduledRequest("b", [2], SamplingParams(max_tokens=1))
    sched.add(a)
    sched.add(b)
    stage = EchoStage()
    sched.step(stage.decode_batch)          # a admitted (and finishes)
    assert sched.abort("b") is True         # still waiting -> removed
    assert sched.abort("ghost") is False
    assert not sched.has_work


def test_scheduler_guards():
    sched = OmniARScheduler(max_num_seqs=2)
    sched.add(ScheduledRequest("x", [1]))
    try:
        sched.add(ScheduledRequest("x", [2]))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate ids must be rejected")
    try:
        sched.step(lambda batch: [])       # wrong length
    except RuntimeError as exc:
        assert "refusing to guess" in str(exc)
    else:
        raise AssertionError("token-count mismatch must fail closed")


# ------------------------------------------------------------------ stages

def test_stage_resolution_fail_closed():
    assert isinstance(create_stage("echo"), EchoStage)
    os.environ.pop("WLLM_OMNI_ALLOW_STUB", None)
    try:
        create_stage("org/very-real-model", {"model_arch": "MysteryArch"})
    except ModelNotSupported as exc:
        assert "Refusing to fall back silently" in str(exc)
    else:
        raise AssertionError("unknown model must fail closed")
    os.environ["WLLM_OMNI_ALLOW_STUB"] = "1"
    try:
        assert isinstance(create_stage("org/very-real-model"), EchoStage)
    finally:
        os.environ.pop("WLLM_OMNI_ALLOW_STUB", None)


# ------------------------------------------------------------------ engine

def test_engine_generate_streams_and_finishes():
    with tempfile.TemporaryDirectory() as td:
        engine = AsyncOmni(model="echo",
                           stage_configs_path=_write(td, STAGE_YAML))

        async def drive():
            outs = []
            async for out in engine.generate(
                    prompt={"prompt": "hello omni"}, request_id="r1",
                    sampling_params_list=[SamplingParams(max_tokens=4, seed=3)],
                    output_modalities=["text"]):
                outs.append(out)
            return outs

        outs = asyncio.run(drive())
        assert outs and outs[-1].finished
        comp = outs[-1].request_output.outputs[0]
        assert len(comp.token_ids) == 4 and comp.finish_reason == "length"
        assert comp.text and comp.multimodal_output["0"]  # latent tables
        assert len(outs) >= 4                              # streamed per step
        stats = engine.stats()
        assert stats["stages"][0]["scheduler"] == "OmniARScheduler"
        assert stats["stages"][0]["steps"] >= 4


def test_engine_concurrent_generates_batch_together():
    with tempfile.TemporaryDirectory() as td:
        engine = AsyncOmni(model="echo",
                           stage_configs_path=_write(td, STAGE_YAML))

        async def one(i):
            final = None
            async for out in engine.generate(
                    prompt=f"req {i}", request_id=f"c{i}",
                    sampling_params_list=[SamplingParams(max_tokens=6, seed=i)]):
                final = out
            return final

        async def drive():
            return await asyncio.gather(*(one(i) for i in range(3)))

        finals = asyncio.run(drive())
        assert all(f.finished for f in finals)
        # authenticity: continuous batching provably engaged
        assert engine.stats()["stages"][0]["max_step_batch"] >= 2


def test_engine_two_stage_chaining():
    with tempfile.TemporaryDirectory() as td:
        engine = AsyncOmni(model="echo",
                           stage_configs_path=_write(td, TWO_STAGE_YAML))

        async def drive():
            final = None
            async for out in engine.generate(
                    prompt="chain me", request_id="chain",
                    sampling_params_list=[SamplingParams(max_tokens=2)]):
                final = out
            return final

        final = asyncio.run(drive())
        assert final.finished
        stats = {s["stage_id"]: s for s in engine.stats()["stages"]}
        assert stats[0]["scheduler"] == "OmniGenerationScheduler"
        assert stats[0]["completed"] == 1 and stats[1]["completed"] == 1


def test_engine_unknown_model_fails_at_construction():
    try:
        AsyncOmni(model="org/unregistered-model")
    except ModelNotSupported:
        pass
    else:
        raise AssertionError("engine must fail closed on unknown models")


def test_engine_rejects_malformed_stage_configs():
    cases = {
        "missing scheduler": """\
stage_args:
  - {stage_type: llm, final_output: true,
     engine_args: {model_stage_backend: echo}}
""",
        "duplicate stage_id": """\
stage_args:
  - {stage_id: 0, stage_type: llm,
     engine_args: {model_stage_backend: echo,
                   scheduler_cls: wllm.omni.core.sched.omni_ar_scheduler.OmniARScheduler}}
  - {stage_id: 0, stage_type: llm, final_output: true,
     engine_args: {model_stage_backend: echo,
                   scheduler_cls: wllm.omni.core.sched.omni_ar_scheduler.OmniARScheduler}}
""",
        "final not last": """\
stage_args:
  - {stage_id: 0, stage_type: llm, final_output: true,
     engine_args: {model_stage_backend: echo,
                   scheduler_cls: wllm.omni.core.sched.omni_ar_scheduler.OmniARScheduler}}
  - {stage_id: 1, stage_type: codec,
     engine_args: {model_stage_backend: echo,
                   scheduler_cls: wllm.omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler}}
""",
    }
    with tempfile.TemporaryDirectory() as td:
        for name, text in cases.items():
            try:
                AsyncOmni(model="echo",
                          stage_configs_path=_write(td, text, f"{name}.yaml"))
            except ValueError:
                pass
            else:
                raise AssertionError(f"config must be rejected: {name}")


def test_engine_generation_payload_lands_on_contract_keys():
    cfg = """\
stage_args:
  - stage_id: 0
    stage_type: codec
    engine_args:
      model_stage_backend: echo
      scheduler_cls: wllm.omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
    final_output: true
    final_output_type: audio
"""
    with tempfile.TemporaryDirectory() as td:
        engine = AsyncOmni(model="echo",
                           stage_configs_path=_write(td, cfg))

        async def drive():
            final = None
            async for out in engine.generate(prompt="vocode this",
                                             request_id="gen1"):
                final = out
            return final

        final = asyncio.run(drive())
        mm = final.request_output.outputs[0].multimodal_output
        # dict payloads surface on their own keys, never buried in "result"
        assert "payload_ids" in mm and "checksum" in mm
        assert "result" not in mm


def test_engine_crash_infection_is_raised_not_faked():
    from wllm.omni.stages import register_stage

    class CrashStage(EchoStage):
        def decode_batch(self, batch):
            raise RuntimeError("kernel exploded")

    register_stage("crash", CrashStage)
    engine = AsyncOmni(model="crash")

    async def drive():
        async for _ in engine.generate(prompt="boom", request_id="r1"):
            pass

    try:
        asyncio.run(drive())
    except RuntimeError as exc:
        assert "kernel exploded" in str(exc) or "failed in-stage" in str(exc)
    else:
        raise AssertionError("crash must surface, never a fake final output")
    # nothing poisoned stays resident
    assert not engine._final.scheduler.has_work


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
