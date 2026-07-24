"""AsyncOmni: the in-tree async multi-stage omni engine.

Contract-compatible with what the apps call today:

    engine = AsyncOmni(model=..., stage_configs_path=..., trust_remote_code=True)
    async for out in engine.generate(prompt=..., request_id=...,
                                     sampling_params_list=[sp],
                                     output_modalities=["text"]):
        ...

Execution model: stages run in stage-id order per request; non-final
stages run to completion and deposit their payload into the request
context; the final AR stage streams one yield per decode step. All
requests in flight share each stage's scheduler, so concurrent
``generate`` calls are continuously batched — ``engine.stats()``
exposes ``max_step_batch`` as the authenticity signal that batching
really happened.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .core.sched.base import ScheduledRequest
from .core.sched.omni_ar_scheduler import OmniARScheduler
from .core.sched.omni_generation_scheduler import OmniGenerationScheduler
from .outputs import CompletionOutput, OmniOutput, RequestOutput
from .stage_config import StageConfig, StageSpec, load_stage_config, resolve_scheduler
from .stages import ModelNotSupported, create_stage

__all__ = ["AsyncOmni", "ModelNotSupported"]

_DEFAULT_STAGE = StageSpec(
    stage_id=0, stage_type="llm",
    engine_args={"scheduler_cls":
                 "wllm.omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                 "max_num_seqs": 64},
    final_output=True, final_output_type="text")


@dataclass
class _Stage:
    spec: StageSpec
    scheduler: object
    model: object

    @property
    def is_ar(self) -> bool:
        return isinstance(self.scheduler, OmniARScheduler)


class AsyncOmni:
    def __init__(self, model: str, stage_configs_path: str | None = None,
                 trust_remote_code: bool = False, **extra_engine_kwargs):
        self.model = model
        self.extra_engine_kwargs = extra_engine_kwargs
        if stage_configs_path:
            cfg = load_stage_config(stage_configs_path)
        else:
            cfg = StageConfig(stages=[_DEFAULT_STAGE])
        self._stages: list[_Stage] = []
        seen_ids: set[int] = set()
        for spec in sorted(cfg.stages, key=lambda s: s.stage_id):
            if spec.stage_id in seen_ids:
                raise ValueError(f"duplicate stage_id {spec.stage_id}; "
                                 f"stage results would silently collide")
            seen_ids.add(spec.stage_id)
            # no default substitution: an empty scheduler path must fail
            # in resolve_scheduler, never be silently replaced
            sched_cls = resolve_scheduler(spec.scheduler_path)
            if not issubclass(sched_cls,
                              (OmniARScheduler, OmniGenerationScheduler)):
                raise TypeError(
                    f"stage {spec.stage_id}: scheduler {sched_cls.__name__} "
                    f"is not a supported stage scheduler")
            scheduler = sched_cls(
                max_num_seqs=int(spec.engine_args.get("max_num_seqs", 64)))
            stage_model = create_stage(model, spec.engine_args,
                                       trust_remote_code)
            if (spec.engine_args.get("engine_output_type") == "latent"
                    and not hasattr(stage_model, "latent_tables")):
                raise ModelNotSupported(
                    f"stage {spec.stage_id} declares engine_output_type="
                    f"latent but its model stage provides no latent_tables(); "
                    f"the engine will not fabricate latent data")
            self._stages.append(_Stage(spec, scheduler, stage_model))
        self._final = next(s for s in self._stages if s.spec.final_output)
        if self._stages[-1] is not self._final:
            raise ValueError(
                "the final_output stage must have the highest stage_id; "
                "stages after it would silently never run")
        self._pump_lock = asyncio.Lock()

    # ------------------------------------------------------------- generate
    async def generate(self, prompt, request_id: str,
                       sampling_params_list=None, output_modalities=None):
        prompt_text, final_token_ids = self._normalize_prompt(prompt)
        context: dict = {"output_modalities": list(output_modalities or [])}

        for index, stage in enumerate(self._stages):
            if stage is self._final:
                break
            params = self._stage_params(sampling_params_list, index, stage)
            ids = (stage.model.tokenize(prompt_text) if prompt_text
                   else list(final_token_ids))
            req = ScheduledRequest(
                request_id=f"{request_id}/s{stage.spec.stage_id}",
                prompt_token_ids=ids, params=params, context=dict(context))
            stage.scheduler.add(req)
            try:
                while not req.finished:
                    self._pump_stage(stage)
                    await asyncio.sleep(0)
            finally:
                stage.scheduler.abort(req.request_id)
            self._raise_if_failed(req)
            if not stage.is_ar and "result" not in req.context:
                raise RuntimeError(
                    f"stage {stage.spec.stage_id} finished without a "
                    f"result payload; refusing to pass None downstream")
            context[f"stage{stage.spec.stage_id}_result"] = \
                req.context.get("result")

        params = self._stage_params(sampling_params_list,
                                    len(self._stages) - 1, self._final)
        req = ScheduledRequest(request_id=request_id,
                               prompt_token_ids=list(final_token_ids),
                               params=params, context=context)
        self._final.scheduler.add(req)
        emitted = 0
        try:
            while not req.finished:
                async with self._pump_lock:
                    if not req.finished:
                        self._pump_stage(self._final)
                if len(req.output_token_ids) > emitted and not req.finished:
                    emitted = len(req.output_token_ids)
                    yield self._output(req, prompt_text, final=False)
                await asyncio.sleep(0)
        finally:
            # a cancelled generator must never leave its request resident
            self._final.scheduler.abort(req.request_id)
        self._raise_if_failed(req)
        # exactly one final output, even when another request's pump
        # finished this one between our checks
        yield self._output(req, prompt_text, final=True)

    @staticmethod
    def _stage_params(sampling_params_list, index: int, stage: "_Stage"):
        """Per-stage params: caller list by stage order, else the stage's
        declared defaults (a dict works — params access is duck-typed)."""
        lst = sampling_params_list or []
        if index < len(lst) and lst[index] is not None:
            return lst[index]
        if len(lst) == 1 and lst[0] is not None:
            return lst[0]
        return stage.spec.default_sampling_params or None

    @staticmethod
    def _raise_if_failed(req: ScheduledRequest) -> None:
        if req.finish_reason == "error":
            raise RuntimeError(
                f"request {req.request_id!r} failed in-stage: "
                f"{req.context.get('error', 'unknown error')}")

    # ------------------------------------------------------------ internals
    def _pump_stage(self, stage: _Stage) -> None:
        if not stage.scheduler.has_work:
            return
        if stage.is_ar:
            stage.scheduler.step(stage.model.decode_batch)
        else:
            stage.scheduler.step(stage.model.generate_batch)

    def _normalize_prompt(self, prompt) -> tuple[str, list[int]]:
        if isinstance(prompt, dict):
            if "prompt_token_ids" in prompt:
                ids = [int(t) for t in prompt["prompt_token_ids"]]
                return "", ids
            text = str(prompt.get("prompt", ""))
        else:
            text = str(prompt or "")
        return text, self._final.model.tokenize(text)

    def _output(self, req: ScheduledRequest, prompt_text: str,
                final: bool) -> OmniOutput:
        text = ""
        if req.output_token_ids and hasattr(self._final.model, "detokenize"):
            text = self._final.model.detokenize(req.output_token_ids)
        multimodal: dict = {}
        if self._final.spec.engine_args.get("engine_output_type") == "latent":
            # tables come from the model stage (checked at construction);
            # the engine never fabricates latent data
            multimodal.update(self._final.model.latent_tables(req))
        if "result" in req.context:
            result = req.context["result"]
            if isinstance(result, dict):
                # generation payloads land on their own keys (audio, sr, ...)
                # so consumers read them from the contract locations
                multimodal.update(result)
            else:
                multimodal["result"] = result
        completion = CompletionOutput(
            text=text, token_ids=list(req.output_token_ids),
            finish_reason=req.finish_reason if final else None,
            multimodal_output=multimodal)
        request_output = RequestOutput(
            request_id=req.request_id,
            prompt_token_ids=list(req.prompt_token_ids),
            outputs=[completion], finished=final)
        return OmniOutput(request_id=req.request_id,
                          request_output=request_output,
                          stage_id=self._final.spec.stage_id)

    # ------------------------------------------------------------- evidence
    def stats(self) -> dict:
        return {
            "model": self.model,
            "stages": [{
                "stage_id": s.spec.stage_id,
                "stage_type": s.spec.stage_type,
                "scheduler": type(s.scheduler).__name__,
                **s.scheduler.stats.as_dict(),
            } for s in self._stages],
        }
