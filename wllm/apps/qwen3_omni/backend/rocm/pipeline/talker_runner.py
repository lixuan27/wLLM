# VENDORED (unmodified base) from wllm/apps/qwen3_omni/reference/runner.py
# Origin is the read-only reference backend. Kept as the exact-numerics
# base for IR validation and streaming variants. Variants that need to
# change talker behavior (TP, per-step scheduling) copy + modify THIS
# file within their own subpackage.
"""Streaming Qwen3-Omni Talker runner.

Drives the in-process engine Talker adapter one codec frame at a
time so the worker can interleave thinker token arrivals with talker
codec generation while keeping scheduling outside the engine.

Per-step protocol mirrors the Qwen3-Omni Talker generation loop:

  1. Sample the FIRST codec layer from ``codec_head`` logits at the
     last position. Apply temperature, repetition penalty, top-k/top-p,
     and suppress the special-codec band the model must not emit.
  2. Feed [past_hidden_last, embed(first_layer)] into ``code_predictor.generate``
     to autoregressively produce the OTHER 15 RVQ layers.
  3. Sum: ``embed(first_layer) + sum(15 mid_residual_hiddens) + embed(last_residual)``
     -> ``codec_input_embed`` for the next forward.
  4. Add the next ``trailing_text_hidden`` slot (one projected thinker
     decode token at a time, or ``tts_pad_embed`` once the thinker queue
     is exhausted).
  5. Forward through the talker -> new logits/hidden -> repeat.

Streaming hook: the trailing queue can be extended at any time by
``append_thinker_decode_token``, which projects the new thinker decode
embedding into talker space and pushes it onto the queue. The runner
stalls (returns ``None`` from ``step``) only when the queue is empty
*and* the thinker session is still in progress.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch
from transformers import AutoConfig

from wllm.serving.logger import init_logger
from wllm.serving.models.qwen3_omni_talker import (
    build_assistant_parts,
    build_user_part,
    get_tts_special_embeds,
    load_vllm_talker_model,
    split_thinker_segments,
)

logger = init_logger(__name__)


# Qwen3-Omni chat-template token ids (mirrors the worker / upstream).
_IM_START_TOKEN_ID = 151644
_SYSTEM_TOKEN_ID = 8948
_USER_TOKEN_ID = 872
_ASSISTANT_TOKEN_ID = 77091

_DEFAULT_AUDIO_TOKEN_ID = 151646
_DEFAULT_IMAGE_TOKEN_ID = 151655
_DEFAULT_VIDEO_TOKEN_ID = 151656


def _resolve_relative_cuda_index(physical_gpu_index: int) -> int:
    """Translate a physical GPU index into the parent process's relative
    ``cuda:N`` index. The yaml uses absolute physical ids (matching the
    AsyncOmni convention); the runner lives in the parent process whose
    CUDA driver was already initialized with the parent's full visibility
    list, so we have to translate. Falls back to passthrough when
    ``CUDA_VISIBLE_DEVICES`` is unset.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return int(physical_gpu_index)
    parts = [p.strip() for p in visible.split(",") if p.strip()]
    try:
        return parts.index(str(int(physical_gpu_index)))
    except ValueError:
        return int(physical_gpu_index)


class Qwen3OmniTalkerRunner:
    """Streaming inference driver for the Qwen3-Omni Talker.

    Lock model: every public method that touches model state holds
    ``self._lock`` so the worker can safely call
    ``append_thinker_decode_token`` from one task while ``step`` runs
    from another. Append acquires the lock briefly to push to the
    queue; step holds it through its forward pass + sampling.
    """

    def __init__(
        self,
        model_path: str,
        *,
        gpu_index: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        # Sampling params for the first codec layer (matches upstream
        # ``talker_*`` defaults).
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        seed: int = 42,
        max_tokens: int = 4096,
        max_seq_len: Optional[int] = None,
        # MTP code-predictor sampling (upstream defaults: top_k=50, top_p=0.8).
        mtp_top_k: int = 50,
        mtp_top_p: float = 0.8,
        debug: bool = False,
    ):
        relative_idx = _resolve_relative_cuda_index(int(gpu_index))
        self.device = torch.device(f"cuda:{relative_idx}")
        torch.cuda.set_device(self.device)
        self.dtype = dtype
        logger.info(
            "Talker runner using physical GPU %s -> relative cuda:%d",
            gpu_index,
            relative_idx,
        )

        full_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        cfg = full_config.talker_config
        resolved_max_seq_len = self._resolve_max_seq_len(
            cfg, max_tokens=int(max_tokens), requested=max_seq_len,
        )

        logger.info(
            "Building Talker from engine internals "
            "(single-request manual scheduler)",
        )
        self.talker, self.special = load_vllm_talker_model(
            model_path,
            full_config=full_config,
            device=str(self.device),
            dtype=dtype,
            max_seq_len=resolved_max_seq_len,
        )
        self.talker.eval()
        self.talker.requires_grad_(False)
        self.codec_eos_token_id = self.special.codec_eos_token_id
        self.vocab_size = int(cfg.text_config.vocab_size)
        self.num_code_groups = int(cfg.num_code_groups)
        # Suppress the special-codec band, EXCEPT codec_eos (matches
        # ``talker_supppressed_tokens`` filter in upstream generation).
        self.suppressed_token_ids = [
            i
            for i in range(self.vocab_size - 1024, self.vocab_size)
            if i != self.codec_eos_token_id
        ]

        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self.mtp_top_k = int(mtp_top_k)
        self.mtp_top_p = float(mtp_top_p)
        self.talker.set_code_predictor_sampling_params(
            top_k=self.mtp_top_k,
            top_p=self.mtp_top_p,
        )
        self.debug = bool(debug)
        self._sampling_generator = torch.Generator(device=self.device)
        self._sampling_generator.manual_seed(self.seed)
        # vLLM seeds request sampling separately from model-side random ops.
        # The Talker layer-0 token uses the request seed above; CodePredictor's
        # internal torch.multinomial path uses the worker/model CUDA RNG, whose
        # vLLM ModelConfig default is 0.
        self._codepred_seed = 0
        logger.info(
            "Talker decode limits: max_tokens=%d, max_seq_len=%d",
            self.max_tokens,
            self.talker.max_seq_len,
        )

        torch.cuda.empty_cache()
        logger.info("Talker uses the engine's compile and CUDA graph paths")

        # Eager warmup: trigger torch.compile cold-start NOW, during the
        # backend init, so the first user prompt isn't paying ~30-45 s
        # of compile latency. Mirrors what vLLM does at engine startup
        # (``_dummy_run`` over each compile/cudagraph capture size).
        # Both the talker forward (prefill + decode shapes) and the
        # code predictor's per-step forward are warmed up.
        self._warmup_compile()

        # Per-session state (reset by ``start_session``).
        self._lock = threading.Lock()
        self._reset_session_state()

    @staticmethod
    def _resolve_max_seq_len(
        talker_config,
        *,
        max_tokens: int,
        requested: Optional[int],
    ) -> int:
        """Resolve total Talker context length.

        The engine treats ``max_tokens`` as generated codec frames, while
        the model's KV cache also has to hold prompt/prefill positions.
        Keep a prompt headroom margin by default so a 4096-token Talker
        generation does not run into a 4096-position cache.
        """
        text_cfg = talker_config.text_config
        model_limit = (
            getattr(text_cfg, "max_position_embeddings", None)
            or getattr(talker_config, "max_position_embeddings", None)
            or 65536
        )
        model_limit = int(model_limit)
        if requested is not None and int(requested) > 0:
            target = int(requested)
        else:
            target = int(max_tokens) + 4096

        if model_limit > 0 and target > model_limit:
            logger.warning(
                "Requested Talker max_seq_len=%d exceeds model limit %d; "
                "capping to model limit",
                target,
                model_limit,
            )
            target = model_limit

        if target <= int(max_tokens):
            logger.warning(
                "Talker max_seq_len=%d leaves no prefill headroom for "
                "max_tokens=%d; long generations may stop early",
                target,
                max_tokens,
            )
        return max(1, target)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _warmup_compile(self) -> None:
        """Run dummy forward passes through the talker + predictor so
        torch.compile pays its cold-start cost before any real prompt
        arrives.

        Mirrors the warmup loop in
        ``vllm/v1/worker/gpu_worker.py::_warmup_kernels`` (``Compile and
        warming up model for size N`` → ``model_runner._dummy_run``).
        Uses zeroed inputs at representative shapes:

          * Talker prefill at T = 64 (typical thinker-output length for
            short prompts; ``dynamic=True`` should generalize to other
            T from this single trace).
          * Talker decode at T = 1, with a non-empty KV cache so the
            attention path that actually runs at runtime is the one
            that gets compiled.
          * Predictor at the full top-k AR loop (covers all internal
            sequence lengths 2..G).

        Logs the compile time for observability. Skipped silently if
        anything fails — the model still works in eager mode.
        """
        import time as _time
        H = self.talker.text_cfg.hidden_size
        # Warmup at multiple prefill lengths so torch.compile's
        # dynamic-shape graph stops re-specializing on the first real
        # prompt. We mark the seq-len dim dynamic explicitly via
        # ``torch._dynamo.mark_dynamic`` and prime several specific
        # values: a small one (8) to force dynamic recompilation, then
        # a sweep that covers the typical real prefill length
        # (~20-60 tokens for short prompts, larger for longer ones).
        prefill_Ts = [8, 16, 24, 26, 32, 48, 64, 96, 128, 192]
        warmup_t0 = _time.time()
        try:
            with torch.inference_mode():
                # Make sure the KV cache buffers are allocated before
                # we kick off compilation (so the graph captures the
                # final, persistent buffer addresses).
                self.talker.ensure_kv_cache(
                    device=self.device, dtype=self.dtype,
                )

                # 1) One talker prefill at the typical length, just to
                # exercise the path so the first real prompt doesn't
                # hit any lazy module init. Eager so no compile cost.
                t0 = _time.time()
                T_pre = 32
                dummy_embeds = torch.zeros(
                    1, T_pre, H, dtype=self.dtype, device=self.device,
                )
                dummy_pos = torch.arange(
                    T_pre, dtype=torch.long, device=self.device,
                ).unsqueeze(0)
                self.talker.forward_prefill(dummy_embeds, dummy_pos)
                torch.cuda.synchronize(self.device)
                logger.info(
                    "Warmup: talker prefill T=%d took %.2fs (eager)",
                    T_pre, _time.time() - t0,
                )

                # 2) Talker decode via vLLM paged attention.
                t0 = _time.time()
                last_T = prefill_Ts[-1]
                for step in range(8):
                    dummy_embeds = torch.zeros(
                        1, 1, H, dtype=self.dtype, device=self.device,
                    )
                    dummy_pos = torch.tensor(
                        [[last_T + step]],
                        dtype=torch.long, device=self.device,
                    )
                    cache_pos = torch.tensor(
                        last_T + step, dtype=torch.long, device=self.device,
                    )
                    self.talker.forward_decode(
                        dummy_embeds, dummy_pos, cache_pos,
                    )
                torch.cuda.synchronize(self.device)
                logger.info(
                    "Warmup: 8 talker decode steps took %.2fs",
                    _time.time() - t0,
                )

                # 3) engine CodePredictor (its AR loop covers every
                #    intra-loop seq_len in one call).
                t0 = _time.time()
                dummy_code = torch.zeros(
                    1, 1, dtype=torch.long, device=self.device,
                )
                dummy_layer0 = torch.zeros(
                    1, 1, H, dtype=self.dtype, device=self.device,
                )
                dummy_last = torch.zeros(
                    1, 1, H, dtype=self.dtype, device=self.device,
                )
                self.talker.talker_mtp_forward(
                    dummy_code,
                    dummy_layer0,
                    last_talker_hidden=dummy_last,
                    text_step=torch.zeros_like(dummy_layer0),
                )
                self.talker.talker_mtp_forward(
                    dummy_code,
                    dummy_layer0,
                    last_talker_hidden=dummy_last,
                    text_step=torch.zeros_like(dummy_layer0),
                )
                torch.cuda.synchronize(self.device)
                logger.info(
                    "Warmup: 2 graph-wrapped MTP calls took %.2fs",
                    _time.time() - t0,
                )

            logger.info(
                "Warmup complete in %.2fs total",
                _time.time() - warmup_t0,
            )
        except Exception:
            logger.exception("Warmup failed")

    def _reset_session_state(self) -> None:
        self._last_hidden = None  # tuple[torch.Tensor, ...] from last forward
        self._last_logits = None  # [1, T, vocab]
        self._sampled_token_history: list[int] = []
        self._sampling_generator.manual_seed(self.seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(self._codepred_seed)
        # Trailing queue is kept as a list of [hidden] CPU tensors so we
        # can append projected thinker decode tokens at any time without
        # realloc. Does NOT include tts_eos -- that is appended by
        # ``mark_thinker_finished`` and consumed once the thinker queue
        # runs out.
        self._trailing_decode_embeds: list[torch.Tensor] = []
        self._tts_eos_embed: Optional[torch.Tensor] = None  # [1, talker_hidden] CPU
        self._tts_pad_embed: Optional[torch.Tensor] = None  # [1, talker_hidden] CPU
        self._generation_step = 0
        self._cache_len: int = 0
        self._thinker_session_finished = False
        self._codec_eos_seen = False

    def start_session(
        self,
        *,
        thinker_prompt_token_ids: list[int],
        thinker_output_token_ids: list[int],
        thinker_prefill_embed: torch.Tensor,
        thinker_prefill_hidden: torch.Tensor,
        tts_bos_embed_thinker: torch.Tensor,
        tts_eos_embed_thinker: torch.Tensor,
        tts_pad_embed_thinker: torch.Tensor,
        speaker: str = "chelsie",
    ) -> None:
        """Build the talker prefill and run it.

        Requires the thinker prompt tokens AND at least 1 generated
        thinker token (the assistant prefill consumes the first 4
        chat-template/generated positions). The remaining generated
        tokens (token 4 onwards) become the initial trailing queue.
        """
        with self._lock:
            self._reset_session_state()

            speaker_id = self._resolve_speaker_id(speaker)

            full_token_ids = torch.tensor(
                list(thinker_prompt_token_ids) + list(thinker_output_token_ids),
                dtype=torch.long,
            )
            prompt_ids_t = torch.tensor(
                list(thinker_prompt_token_ids), dtype=torch.long,
            )

            need_rows = full_token_ids.shape[0]
            if thinker_prefill_embed.shape[0] < need_rows:
                raise ValueError(
                    f"thinker_prefill_embed has only {thinker_prefill_embed.shape[0]}"
                    f" rows but need {need_rows} (P={prompt_ids_t.shape[0]} +"
                    f" N={len(thinker_output_token_ids)})"
                )

            full_embed = thinker_prefill_embed[:need_rows].to(
                device=self.device, dtype=self.dtype
            )
            full_hidden = thinker_prefill_hidden[:need_rows].to(
                device=self.device, dtype=self.dtype
            )

            tts_bos, tts_eos, tts_pad = get_tts_special_embeds(
                talker=self.talker,
                tts_bos_thinker=tts_bos_embed_thinker.to(
                    device=self.device, dtype=self.dtype
                ),
                tts_eos_thinker=tts_eos_embed_thinker.to(
                    device=self.device, dtype=self.dtype
                ),
                tts_pad_thinker=tts_pad_embed_thinker.to(
                    device=self.device, dtype=self.dtype
                ),
                target_dtype=self.dtype,
            )
            self._tts_eos_embed = tts_eos.detach().to("cpu")
            self._tts_pad_embed = tts_pad.detach().to("cpu")

            user_segments, (assist_start, assist_end) = split_thinker_segments(
                thinker_input_token_ids=prompt_ids_t,
                thinker_output_token_ids=torch.tensor(
                    thinker_output_token_ids, dtype=torch.long
                ),
                full_thinker_embed=full_embed,
                full_thinker_hidden=full_hidden,
                multimodal_token_ids=(
                    _DEFAULT_AUDIO_TOKEN_ID,
                    _DEFAULT_IMAGE_TOKEN_ID,
                    _DEFAULT_VIDEO_TOKEN_ID,
                ),
                im_start_token_id=_IM_START_TOKEN_ID,
                user_token_id=_USER_TOKEN_ID,
                assistant_token_id=_ASSISTANT_TOKEN_ID,
                system_token_id=_SYSTEM_TOKEN_ID,
            )

            mm_mask = torch.zeros(
                full_token_ids.shape[0], dtype=torch.bool, device=self.device,
            )
            for tok_id in (
                _DEFAULT_AUDIO_TOKEN_ID,
                _DEFAULT_IMAGE_TOKEN_ID,
                _DEFAULT_VIDEO_TOKEN_ID,
            ):
                mm_mask |= full_token_ids.to(self.device) == tok_id

            user_parts = []
            user_ids = []
            for s, e in user_segments:
                user_parts.append(
                    build_user_part(
                        talker=self.talker,
                        thinker_embed_segment=full_embed[s:e],
                        thinker_hidden_segment=full_hidden[s:e],
                        multimodal_mask_segment=mm_mask[s:e],
                        target_dtype=self.dtype,
                    )
                )
                user_ids.append(full_token_ids[s:e].to(self.device))

            assistant_thinker_embed = full_embed[assist_start:assist_end]
            assist_part, trailing_decode = build_assistant_parts(
                talker=self.talker,
                assistant_thinker_embed=assistant_thinker_embed,
                speaker_id=speaker_id,
                tts_pad_embed=tts_pad,
                tts_bos_embed=tts_bos,
                tts_eos_embed=tts_eos,
                special=self.special,
                target_dtype=self.dtype,
            )
            for row in trailing_decode:
                self._trailing_decode_embeds.append(row.detach().to("cpu"))

            prefill_embed = torch.cat(user_parts + [assist_part], dim=0).unsqueeze(0)
            assist_ids = torch.full(
                (1, assist_part.shape[0]),
                fill_value=self.special.tts_pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            user_ids_t = (
                torch.cat(user_ids, dim=0).unsqueeze(0)
                if user_ids
                else torch.empty((1, 0), dtype=torch.long, device=self.device)
            )
            prefill_input_ids = torch.cat([user_ids_t, assist_ids], dim=1)

            T_prefill = int(prefill_embed.shape[1])
            if T_prefill >= self.talker.max_seq_len:
                raise ValueError(
                    "Talker prefill length "
                    f"{T_prefill} exceeds max_seq_len={self.talker.max_seq_len}; "
                    "increase sampling.talker_max_seq_len"
                )
            if T_prefill + self.max_tokens > self.talker.max_seq_len:
                logger.warning(
                    "Talker max_seq_len=%d allows only %d generated frames "
                    "after prefill T=%d, less than max_tokens=%d",
                    self.talker.max_seq_len,
                    self.talker.max_seq_len - T_prefill,
                    T_prefill,
                    self.max_tokens,
                )
            prefill_pos_ids = torch.arange(
                0, T_prefill, dtype=torch.long, device=self.device,
            ).unsqueeze(0)

            if self.debug:
                logger.debug(
                    "Talker prefill: T=%d, user_segs=%d, "
                    "trailing_decode=%d, speaker_id=%d",
                    T_prefill,
                    len(user_parts),
                    len(self._trailing_decode_embeds),
                    speaker_id,
                )
                logger.debug(
                    "Prefill embed stats: dtype=%s, mean=%.4f, std=%.4f, "
                    "abs_max=%.4f",
                    prefill_embed.dtype,
                    prefill_embed.float().mean().item(),
                    prefill_embed.float().std().item(),
                    prefill_embed.float().abs().max().item(),
                )

            with torch.inference_mode():
                logits, hidden = self.talker.forward_prefill(
                    prefill_embed, prefill_pos_ids,
                )
            # The vLLM paged KV cache is mutated in place by forward_prefill;
            # _cache_len tracks the next write position for decode.
            self._last_hidden = (hidden,)
            self._last_logits = logits  # [1, T_prefill, vocab]
            self._generation_step = 0
            self._sampled_token_history = []
            self._cache_len = T_prefill

            if self.debug:
                last_logit = logits[:, -1, :].float()
                top5 = torch.topk(last_logit, k=5, dim=-1)
                logger.debug(
                    "Prefill done: last_hidden_shape=%s, logits_shape=%s, "
                    "top5_logits=%s, top5_ids=%s, codec_eos_id=%d, "
                    "codec_eos_logit=%.4f",
                    tuple(hidden.shape),
                    tuple(logits.shape),
                    top5.values[0].tolist(),
                    top5.indices[0].tolist(),
                    self.codec_eos_token_id,
                    last_logit[0, self.codec_eos_token_id].item(),
                )

    def _resolve_speaker_id(self, speaker: str) -> int:
        spk_lower = speaker.lower()
        if spk_lower in self.special.speaker_ids:
            return int(self.special.speaker_ids[spk_lower])
        fallback = {"chelsie": 8000, "ethan": 8001}
        if spk_lower in fallback:
            logger.warning(
                "Speaker %r not in talker_config.speaker_id; using fallback id %d",
                speaker,
                fallback[spk_lower],
            )
            return fallback[spk_lower]
        logger.warning(
            "Unknown speaker %r; defaulting to id 8000 (chelsie)", speaker,
        )
        return 8000

    def append_thinker_decode_token(self, thinker_decode_embed: torch.Tensor) -> None:
        """Push one (or more) thinker decode embeds onto the trailing queue.

        Each row is projected through ``talker.text_projection`` and added
        to the trailing list. We do the GPU projection WITHOUT holding
        ``self._lock`` (which the per-step forward holds for ~1s); a
        list ``append`` is GIL-atomic so we don't need the lock for the
        list mutation either. Safe to call from any thread.
        """
        if thinker_decode_embed.ndim == 1:
            thinker_decode_embed = thinker_decode_embed.unsqueeze(0)
        with torch.inference_mode():
            projected = self.talker.text_projection(
                thinker_decode_embed.to(device=self.device, dtype=self.dtype)
            ).detach().to("cpu")
        for row in projected:
            self._trailing_decode_embeds.append(row)
        if self.debug:
            logger.debug(
                "append_thinker_decode_token: added %d rows; trailing_total=%d",
                int(projected.shape[0]),
                len(self._trailing_decode_embeds),
            )

    def mark_thinker_finished(self) -> None:
        # No need to hold the heavy step-lock; this is just a flag flip.
        self._thinker_session_finished = True
        if self.debug:
            logger.debug(
                "mark_thinker_finished: trailing_total=%d",
                len(self._trailing_decode_embeds),
            )

    # ------------------------------------------------------------------
    # Per-step driving
    # ------------------------------------------------------------------

    def step(self) -> Optional[torch.Tensor]:
        """Generate one codec frame, or return None if we should stall."""
        with self._lock:
            if self._codec_eos_seen:
                return None
            # Prefill done = the talker model's KV cache is populated +
            # we have last_logits to sample the first codec layer from.
            if self._last_logits is None:
                raise RuntimeError("step() called before start_session()")

            if self._generation_step >= self.max_tokens:
                logger.warning(
                    "Talker reached max_tokens=%d without codec_eos; stopping",
                    self.max_tokens,
                )
                self._codec_eos_seen = True
                return None
            if self._cache_len >= self.talker.max_seq_len:
                logger.warning(
                    "Talker KV cache exhausted at cache_len=%d "
                    "(max_seq_len=%d, generation_step=%d); stopping. "
                    "Increase sampling.talker_max_seq_len to allow longer output.",
                    self._cache_len,
                    self.talker.max_seq_len,
                    self._generation_step,
                )
                self._codec_eos_seen = True
                return None

            n_decode = len(self._trailing_decode_embeds)
            gen_step = self._generation_step
            if gen_step < n_decode:
                cond = self._trailing_decode_embeds[gen_step]
            elif self._thinker_session_finished:
                if gen_step == n_decode:
                    cond = self._tts_eos_embed
                else:
                    cond = self._tts_pad_embed
            else:
                return None

            return self._step_unlocked(cond)

    def _profile_log(
        self, predictor_ms: float, talker_ms: float, total_ms: float,
    ) -> None:
        """Log a rolling per-step timing breakdown every 10 steps so we can
        see where the wall time is going (predictor MTP loop vs the 30B
        talker forward) without spamming the log on every frame."""
        if not self.debug:
            return
        if not getattr(self, "_profile_buf", None):
            self._profile_buf = []
        self._profile_buf.append((predictor_ms, talker_ms, total_ms))
        if len(self._profile_buf) >= 10:
            ps = [p for p, _, _ in self._profile_buf]
            ts = [t for _, t, _ in self._profile_buf]
            tots = [x for _, _, x in self._profile_buf]
            logger.debug(
                "Talker step timing avg over last 10 steps: "
                "predictor=%.1fms, talker_fwd=%.1fms, total=%.1fms",
                sum(ps) / len(ps), sum(ts) / len(ts), sum(tots) / len(tots),
            )
            self._profile_buf = []

    def _step_unlocked(self, cond_cpu: torch.Tensor) -> Optional[torch.Tensor]:
        # Lightweight per-step timer that does NOT cuda-sync mid-step
        # (each sync drains the launch queue and serialises with no
        # benefit). Just one wall-clock read at start and end -- the GPU
        # work between them dispatches asynchronously, so the printed
        # "predictor" / "talker_fwd" sub-times are best-effort
        # approximations rather than precise breakdowns.
        import time as _time
        _t0 = _time.perf_counter()
        # 1) Sample the FIRST codec layer from the last position's logits.
        last_pos_logits = self._last_logits[:, -1, :]  # [1, vocab]
        first_layer_token_id = self._sample_first_layer(
            last_pos_logits,
            generator=self._sampling_generator,
        )
        first_layer_token = torch.tensor(
            [[first_layer_token_id]], dtype=torch.long, device=self.device,
        )

        is_eos = first_layer_token_id == self.codec_eos_token_id
        if is_eos:
            self._sampled_token_history.append(first_layer_token_id)
            self._codec_eos_seen = True
            if self.debug:
                top5 = torch.topk(last_pos_logits.float(), k=5, dim=-1)
                logger.warning(
                    "Talker sampled codec_eos at step %d; stopping before "
                    "MTP/code2wav. trailing_decode_embeds=%d, thinker_done=%s, "
                    "history_len=%d, current_top5_ids=%s, "
                    "current_codec_eos_logit=%.4f",
                    self._generation_step + 1,
                    len(self._trailing_decode_embeds),
                    self._thinker_session_finished,
                    len(self._sampled_token_history),
                    top5.indices[0].tolist(),
                    last_pos_logits[0, self.codec_eos_token_id].float().item(),
                )
            return None

        # 2) Embed the sampled first-layer token (talker's main codec
        #    embedding — distinct from the code_predictor's
        #    ``codec_embedding`` ModuleList which embeds residual codes).
        layer0_embed = self.talker.get_input_embeddings()(first_layer_token)

        # 3) MTP residual codes via the engine's CodePredictor. Returns
        #    all RVQ codes plus the summed next-talker-input embedding.
        last_layer_hidden = self._last_hidden[-1][:, -1:].to(
            device=layer0_embed.device, dtype=layer0_embed.dtype,
        )
        cond = cond_cpu.to(device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        _t_pred_start = _time.perf_counter()
        next_input_embed, all_token_ids = self.talker.talker_mtp_forward(
            first_layer_token,
            layer0_embed,
            last_talker_hidden=last_layer_hidden,
            text_step=cond,
        )
        all_token_ids = all_token_ids.to(torch.long).clone()
        residual_token_ids = all_token_ids[:, 1:]
        predictor_ms = (_time.perf_counter() - _t_pred_start) * 1000.0

        # 4) Forward the fast talker (1 token decode against the
        #    pre-allocated KV cache; CUDA-graphed via reduce-overhead).
        decode_pos_ids = torch.tensor(
            [[self._cache_len]], dtype=torch.long, device=self.device,
        )
        cache_pos_t = torch.tensor(
            self._cache_len, dtype=torch.long, device=self.device,
        )
        _t_tlk_start = _time.perf_counter()
        with torch.inference_mode():
            logits, hidden = self.talker.forward_decode(
                next_input_embed, decode_pos_ids, cache_pos_t,
            )
        talker_ms = (_time.perf_counter() - _t_tlk_start) * 1000.0
        self._last_hidden = (hidden,)
        self._last_logits = logits

        # 7) Bookkeeping.
        self._sampled_token_history.append(first_layer_token_id)
        self._generation_step += 1
        self._cache_len += 1
        if self.debug and (
            self._generation_step <= 5
            or self._generation_step % 10 == 0
            ):
            top5 = torch.topk(logits[:, -1, :].float(), k=5, dim=-1)
            n_decode_avail = len(self._trailing_decode_embeds)
            logger.debug(
                "Step %d: sampled_layer0=%d, trailing_avail=%d, "
                "thinker_done=%d, next_top5_ids=%s, "
                "next_top5_logits=%s, next_codec_eos_logit=%.4f",
                self._generation_step,
                first_layer_token_id,
                n_decode_avail,
                int(self._thinker_session_finished),
                top5.indices[0].tolist(),
                [round(v, 2) for v in top5.values[0].tolist()],
                logits[0, -1, self.codec_eos_token_id].item(),
            )
        # Keep the codec frame on the GPU to avoid a per-step
        # ``.cpu()`` sync (each forced GPU drain costs ~5-10 ms on top
        # of the actual compute). The c2w dispatcher is fine with GPU
        # tensors -- ``torch.stack`` of GPU frames stays on GPU, and
        # the ``.tolist()`` inside ``_build_code2wav_prompt`` only
        # syncs once per ``codec_chunk_frames`` frames instead of per
        # frame.
        codec_frame = torch.cat(
            [first_layer_token, residual_token_ids], dim=-1,
        ).reshape(-1).to(torch.long).clone()

        total_ms = (_time.perf_counter() - _t0) * 1000.0
        self._profile_log(predictor_ms, talker_ms, total_ms)

        return codec_frame

    def _sample_first_layer(
        self,
        logits: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Sample with suppression / repetition penalty / temperature / top_k / top_p."""
        logits = self._apply_first_layer_constraints(logits)
        if self.temperature == 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        if self.temperature > 0 and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            topk_vals, _ = torch.topk(logits, k=self.top_k, dim=-1)
            kth = topk_vals[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if 0.0 < self.top_p < 1.0:
            # Match vLLM V1 apply_top_k_top_p_pytorch: sort ascending,
            # drop the low-probability tail whose cumulative mass is
            # <= 1 - top_p, then scatter back to vocab order.
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=False, dim=-1,
            )
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs <= (1.0 - self.top_p)
            sorted_mask[..., -1] = False
            sorted_logits = sorted_logits.masked_fill(
                sorted_mask, float("-inf"),
            )
            logits = logits.scatter(
                dim=-1, index=sorted_indices, src=sorted_logits,
            )

        probs = torch.softmax(logits, dim=-1)
        # vLLM's V1 sampler uses exponential-race sampling instead of
        # torch.multinomial to avoid a CPU/GPU sync. For batch size 1 with
        # a request generator, random_sample() does not touch the default
        # CUDA RNG; preserve that so the following CodePredictor multinomial
        # sees the same worker RNG state as the engine.
        q = torch.empty_like(probs)
        if generator is not None:
            q[0].exponential_(generator=generator)
        else:
            q.exponential_()
        sampled = probs.div_(q).argmax(dim=-1)
        return int(sampled.item())

    def _apply_first_layer_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply vLLM-style non-sampling first-layer logit transforms."""
        logits = logits.clone().to(torch.float32)
        if self.suppressed_token_ids:
            logits[..., self.suppressed_token_ids] = float("-inf")
        if self.repetition_penalty != 1.0 and self._sampled_token_history:
            # Vectorized repetition penalty: avoid the per-unique-token
            # Python loop. Build the unique-token tensor once and apply
            # the penalty in a single ``torch.where``.
            hist_unique = torch.tensor(
                list(set(self._sampled_token_history)),
                dtype=torch.long, device=logits.device,
            )
            cur = logits[..., hist_unique]
            penalised = torch.where(
                cur < 0,
                cur * self.repetition_penalty,
                cur / self.repetition_penalty,
            )
            logits.index_copy_(-1, hist_unique, penalised)
        return logits

    # ------------------------------------------------------------------
    # Helpers / state
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        with self._lock:
            return self._codec_eos_seen

    def codec_frames_emitted(self) -> int:
        with self._lock:
            return self._generation_step

    def shutdown(self) -> None:
        with self._lock:
            self._reset_session_state()
            self.talker = None
            torch.cuda.empty_cache()
