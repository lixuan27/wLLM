"""wGraph v0 unit tests: a toy streaming-video program plus failure cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wllm.graph import (
    Backpressure,
    Modality,
    Node,
    NodeOp,
    Program,
    QualityContract,
    QualityMode,
    Region,
    RegionKind,
    StateKind,
    StateScope,
    StateSpec,
    StreamSpec,
)


def build_toy_program() -> Program:
    """text encoder -> chunked DiT rollout -> streaming codec decode."""
    dit_region = Region(
        id="rollout",
        kind=RegionKind.CHUNK_ROLLOUT,
        attrs={"chunk_size": 4},
        children=[
            Region(
                id="denoise",
                kind=RegionKind.DIFFUSION,
                attrs={"num_steps": 4},
                nodes=[
                    Node(id="dit", op=NodeOp.TRANSFORMER,
                         reads=["prompt_ctx"], writes=["kv_main"]),
                ],
            ),
        ],
        nodes=[Node(id="vae", op=NodeOp.CODEC, reads=["vae_cache"],
                    writes=["vae_cache"])],
    )
    root = Region(
        id="app",
        kind=RegionKind.SEQUENTIAL,
        children=[dit_region],
        nodes=[Node(id="text_enc", op=NodeOp.ENCODER, writes=["prompt_ctx"])],
    )
    states = {
        "prompt_ctx": StateSpec(id="prompt_ctx", kind=StateKind.IMMUTABLE_SESSION,
                                scope=StateScope.SESSION, ordered=False,
                                recomputable=True),
        "kv_main": StateSpec(id="kv_main", kind=StateKind.KV, ordered=True,
                             verified=True, evidence="tests/evidence/kv_probe.json"),
        "vae_cache": StateSpec(id="vae_cache", kind=StateKind.RECURRENT,
                               ordered=True),
    }
    streams = {
        "latents": StreamSpec(id="latents", modality=Modality.LATENT,
                              producer="denoise", consumer="vae",
                              chunk_size=1, bounded_queue=2,
                              backpressure=Backpressure.BLOCK),
        "frames": StreamSpec(id="frames", modality=Modality.FRAME,
                             producer="vae", consumer="app",
                             rate_hz=16.0, bounded_queue=2, deadline_ms=62.5,
                             backpressure=Backpressure.DROP_OLDEST),
    }
    return Program(name="toy_video", root=root, states=states, streams=streams,
                   quality=QualityContract())


def test_toy_program_validates():
    prog = build_toy_program()
    assert prog.validate() == []


def test_summary_and_warnings():
    prog = build_toy_program()
    text = prog.summary()
    assert "rollout" in text and "chunk_rollout" in text
    warns = prog.warnings()
    # two unverified states -> two warnings
    assert sum("unverified" in w for w in warns) == 2


def test_duplicate_node_id_rejected():
    prog = build_toy_program()
    prog.root.nodes.append(Node(id="dit", op=NodeOp.CUSTOM))
    assert any("duplicate node id 'dit'" in e for e in prog.validate())


def test_undeclared_state_rejected():
    prog = build_toy_program()
    prog.root.nodes[0].reads.append("ghost_state")
    assert any("undeclared state 'ghost_state'" in e for e in prog.validate())


def test_feedback_region_requires_staleness_and_deadline():
    region = Region(id="fb", kind=RegionKind.FEEDBACK)
    errs = region.validate()
    assert any("max_staleness_ms" in e for e in errs)
    assert any("deadline_ms" in e for e in errs)


def test_ordered_state_single_writer():
    prog = build_toy_program()
    prog.root.nodes.append(
        Node(id="rogue", op=NodeOp.CUSTOM, writes=["kv_main"]))
    assert any("multiple writers" in e for e in prog.validate())


def test_verified_state_requires_evidence():
    spec = StateSpec(id="s", kind=StateKind.KV, verified=True)
    assert any("evidence" in e for e in spec.validate())


def test_multi_agent_state_requires_partition_key():
    spec = StateSpec(id="m", kind=StateKind.MULTI_AGENT)
    assert any("partition_key" in e for e in spec.validate())


def test_stream_bounds_enforced():
    bad = StreamSpec(id="x", modality=Modality.FRAME, producer="a",
                     consumer="b", bounded_queue=0)
    assert any("bounded_queue" in e for e in bad.validate())
    loop = StreamSpec(id="y", modality=Modality.FRAME, producer="a",
                      consumer="a")
    assert any("self-loop" in e for e in loop.validate())


def test_quality_contract_modes():
    exact_with_budget = QualityContract(mode=QualityMode.EXACT,
                                        budgets={"vbench_drop_max": 0.1})
    assert exact_with_budget.validate()
    bounded_empty = QualityContract(mode=QualityMode.BOUNDED_DEGRADATION)
    assert bounded_empty.validate()
    good = QualityContract(mode=QualityMode.BOUNDED_DEGRADATION,
                           budgets={"vbench_drop_max": 0.005})
    assert good.validate() == []
    assert good.allows_approximate()


def test_stream_endpoint_must_exist():
    prog = build_toy_program()
    prog.streams["frames"].consumer = "nonexistent"
    assert any("consumer 'nonexistent' not found" in e for e in prog.validate())


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)
