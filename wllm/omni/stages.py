"""Model stages: pluggable execution behind the schedulers.

A stage turns scheduled requests into tokens (AR stages) or payloads
(generation stages). Real model runners register themselves under a
backend name or model architecture; resolution **fails closed** — an
unregistered model raises ``ModelNotSupported`` instead of silently
running something slower or emptier than the user asked for.

The deterministic :class:`EchoStage` exists for tests, dry runs, and
scheduler/protocol verification. It must be selected explicitly
(``model="echo"``, ``engine_args.model_stage_backend: echo``, or
``WLLM_OMNI_ALLOW_STUB=1``); it is never a fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable, Protocol

from .core.sched.base import ScheduledRequest
from .sampling import read_param


class ModelNotSupported(RuntimeError):
    pass


class ModelStage(Protocol):
    def tokenize(self, prompt: str) -> list[int]: ...
    def decode_batch(self, batch: list[ScheduledRequest]) -> list[int]: ...
    def generate_batch(self, batch: list[ScheduledRequest]) -> list: ...
    def detokenize(self, token_ids: list[int]) -> str: ...


_REGISTRY: dict[str, Callable[..., ModelStage]] = {}


def register_stage(name: str, factory: Callable[..., ModelStage]) -> None:
    _REGISTRY[name] = factory


def registered_stages() -> list[str]:
    return sorted(_REGISTRY)


class EchoStage:
    """Deterministic stub: same prompt + seed => same tokens, always.

    Token ids derive from a stable hash of (prompt tokens, seed, step),
    so scheduler interleaving and batching provably cannot change any
    request's output — which is exactly the parity law the real engine
    must keep, made testable without weights.
    """

    STOP_AFTER = 1 << 30      # never emits a stop token by itself
    name = "echo"

    def __init__(self, model: str = "echo", **_: object):
        self.model = model
        self.decode_calls = 0

    @staticmethod
    def _stable(parts: tuple) -> int:
        blob = "|".join(str(p) for p in parts).encode()
        return int.from_bytes(hashlib.sha256(blob).digest()[:4], "big")

    def tokenize(self, prompt: str) -> list[int]:
        return [self._stable(("tok", w)) % 50000
                for w in (prompt or "").split()] or [0]

    def decode_batch(self, batch: list[ScheduledRequest]) -> list[int]:
        self.decode_calls += 1
        out = []
        for req in batch:
            seed = read_param(req.params, "seed", 0) or 0
            step = len(req.output_token_ids)
            out.append(self._stable(
                ("echo", tuple(req.prompt_token_ids), seed, step)) % 50000)
        return out

    def generate_batch(self, batch: list[ScheduledRequest]) -> list:
        results = []
        for req in batch:
            results.append({
                "payload_ids": list(req.prompt_token_ids),
                "checksum": self._stable(("gen", tuple(req.prompt_token_ids))),
            })
        return results

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"<{t}>" for t in token_ids)

    def latent_tables(self, req: ScheduledRequest) -> dict:
        """Deterministic per-token tables for latent-output dry runs.

        Real model stages return their actual layer tables here; the
        engine refuses latent output from stages lacking this hook.
        """
        return {"0": [[float(t)] for t in req.output_token_ids],
                "24": [[float(t) + 0.5] for t in req.output_token_ids]}


register_stage("echo", EchoStage)


def create_stage(model: str, engine_args: dict | None = None,
                 trust_remote_code: bool = False) -> ModelStage:
    engine_args = engine_args or {}
    backend = str(engine_args.get("model_stage_backend") or "")
    if backend:
        if backend not in _REGISTRY:
            raise ModelNotSupported(
                f"model_stage_backend {backend!r} is not registered; "
                f"registered: {registered_stages()}")
        return _REGISTRY[backend](model=model, **engine_args)
    arch = str(engine_args.get("model_arch") or "")
    if arch in _REGISTRY:
        return _REGISTRY[arch](model=model, **engine_args)
    if model in _REGISTRY:
        return _REGISTRY[model](model=model, **engine_args)
    if os.environ.get("WLLM_OMNI_ALLOW_STUB", "") == "1":
        # process-global escape hatch for dry runs: make it loud so a
        # leaked env var can never silently turn real models into stubs
        logging.getLogger(__name__).error(
            "WLLM_OMNI_ALLOW_STUB=1: serving DETERMINISTIC STUB tokens "
            "for model %r — dry-run output, not real inference", model)
        return EchoStage(model=model)
    raise ModelNotSupported(
        f"no registered stage backend for model {model!r} "
        f"(arch {arch or 'unknown'!r}); registered: {registered_stages()}. "
        f"Refusing to fall back silently — register a runner or request "
        f"the echo stage explicitly for a dry run.")
