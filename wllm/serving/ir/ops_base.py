"""Convenience helpers for defining IR operators.

The primary way to define an operator is to subclass ``IROperator`` and
override ``execute`` (and optionally ``should_run``). Subclassing is the
right pattern when an operator is parameterized (e.g., one class
``DiTDenoiseStep(step_idx=k)`` instantiated for every denoising step).

For one-off leaf operators with no per-instance parameterization, the
``@ir_operator`` decorator wraps a function into an ``IROperator`` subclass
instance with the metadata baked in:

    @ir_operator(
        name="text_encoder_prefill",
        op_type=OpType.EXPOSED,
        inputs=[TensorPort("prompt")],
        outputs=[TensorPort("embeds")],
        state_writes=["prompt_embeds"],
    )
    def text_encoder_prefill(inputs, ctx, state):
        embeds = ctx.text_encoder.encode(inputs["prompt"])
        state.set("prompt_embeds", embeds)
        return {"embeds": embeds}

    graph.add_operator(text_encoder_prefill)

The decorated object is an ``IROperator`` instance (not a function), so it
plugs straight into ``graph.add_operator``. The original function is
available as ``op.wrapped_fn`` for testing or reuse.
"""

from __future__ import annotations

from typing import Any, Callable

from wllm.serving.ir.graph import (
    IROperator,
    OpType,
    StateStore,
    StreamMode,
    TensorPort,
)

ExecuteFn = Callable[[dict[str, Any], Any, StateStore], dict[str, Any]]
ShouldRunFn = Callable[[Any], bool]


def ir_operator(
    *,
    name: str,
    op_type: OpType,
    inputs: list[TensorPort] | None = None,
    outputs: list[TensorPort] | None = None,
    state_reads: list[str] | None = None,
    state_writes: list[str] | None = None,
    stream_mode: StreamMode = StreamMode.BATCH,
    sub_graph: str | None = None,
    config: dict[str, Any] | None = None,
    should_run: ShouldRunFn | None = None,
) -> Callable[[ExecuteFn], IROperator]:
    """Turn a function into an ``IROperator`` instance with baked-in metadata.

    The wrapped function must have the signature
    ``(inputs: dict, context: Any, state: StateStore) -> dict`` and return a
    dict with one entry per declared output port.

    If ``should_run`` is provided, it is invoked as ``should_run(context)``
    at the start of every execution; when it returns False, the operator is
    skipped and its outputs are not produced.

    Returns the decorator that produces the operator instance — use
    ``graph.add_operator(the_decorated_object)`` to register it.
    """

    def decorator(fn: ExecuteFn) -> IROperator:
        class_body: dict[str, Any] = {
            "__init__": _make_init(
                name=name,
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
                state_reads=state_reads,
                state_writes=state_writes,
                stream_mode=stream_mode,
                sub_graph=sub_graph,
                config=config,
                fn=fn,
            ),
            "execute": _make_execute(fn),
        }
        if should_run is not None:
            class_body["should_run"] = _make_should_run(should_run)

        cls = type(f"_Op_{name}", (IROperator,), class_body)
        return cls()

    return decorator


def _make_init(
    *,
    name: str,
    op_type: OpType,
    inputs: list[TensorPort] | None,
    outputs: list[TensorPort] | None,
    state_reads: list[str] | None,
    state_writes: list[str] | None,
    stream_mode: StreamMode,
    sub_graph: str | None,
    config: dict[str, Any] | None,
    fn: ExecuteFn,
) -> Callable[[IROperator], None]:
    def __init__(self: IROperator) -> None:
        IROperator.__init__(
            self,
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            state_reads=state_reads,
            state_writes=state_writes,
            stream_mode=stream_mode,
            sub_graph=sub_graph,
            config=config,
        )
        self.wrapped_fn = fn

    return __init__


def _make_execute(fn: ExecuteFn) -> ExecuteFn:
    def execute(
        self: IROperator,
        inputs: dict[str, Any],
        context: Any,
        state: StateStore,
    ) -> dict[str, Any]:
        del self
        return fn(inputs, context, state)

    return execute


def _make_should_run(user_fn: ShouldRunFn) -> Callable[[IROperator, Any], bool]:
    def should_run(self: IROperator, context: Any) -> bool:
        del self
        return user_fn(context)

    return should_run
