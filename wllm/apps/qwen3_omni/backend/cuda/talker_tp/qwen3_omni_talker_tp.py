"""[VENDORED] from wllm/models/qwen3_omni_talker.py (read-only shared
runtime). Divergence: the detached Talker is made tensor-parallel-aware --
_ensure_vllm_runtime / _build_vllm_config / OmniTalker.__init__ /
load_vllm_talker_model take (tp_size, rank, world_size, init_method) so the
vLLM Talker module shards across `tp_size` GPUs when driven by `world_size`
SPMD ranks sharing a tcp:// rendezvous. Used only by the talker_tp2 variant
(wllm/apps/qwen3_omni/backend/talker_tp/). Unchanged code paths preserve the
reference numerics at tp_size=1; the TP path is validated by teacher-forced
logit parity.

Original module docstring follows.

Qwen3-Omni Talker adapter + prompt-embedding helpers.

The production path is ``OmniTalker``: a detached single-request
wrapper around the engine's Talker module. wLLM still owns the
Thinker/Talker/Vocoder scheduling and streaming decisions, while the
Talker forward pass, paged attention, fused MoE, weight loading, and
CodePredictor come from the omni engine.

The helper functions below build the Talker prefill ``inputs_embeds`` and
trailing decode queue from Thinker outputs. They mirror upstream
Qwen3-Omni segment construction.
"""

from __future__ import annotations

import logging
import math
import os
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.distributed import init_distributed_environment
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager
from wllm.engines import omni as omni_engine

EngineQwen3OmniMoeTalker = omni_engine.attr(
    "model_executor.models.qwen3_omni.qwen3_omni_moe_talker",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
)


logger = logging.getLogger(__name__)


def _free_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return f"tcp://127.0.0.1:{port}"


def _hf_snapshot_cache_dir() -> str | None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None
    hub_dir = os.path.join(hf_home, "hub")
    return hub_dir if os.path.isdir(hub_dir) else hf_home


def _ensure_vllm_runtime(
    vllm_config: VllmConfig,
    *,
    device: torch.device,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int = 1,
    init_method: str | None = None,
) -> None:
    """Initialize the vLLM runtime state (TP-aware) for detached modules.

    For tp_size>1 this must be called from all `world_size` ranks with the
    same `init_method` (a shared tcp:// rendezvous), distinct `rank`, and a
    `device` per rank. The vLLM Talker layers read vLLM's parallel_state
    (get_tensor_model_parallel_world_size etc.), so the talker is sharded
    automatically once this group is up."""
    init_workspace_manager(device)
    with set_current_vllm_config(vllm_config):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            local_rank = device.index if device.index is not None else 0
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=init_method or _free_tcp_init_method(),
                local_rank=local_rank,
                backend="nccl",
            )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )


class OmniTalker(nn.Module):
    """Detached engine Talker with manual single-request scheduling.

    This keeps wLLM's custom Thinker/Talker/Vocoder scheduler while
    reusing the engine's actual Talker module, paged-attention kernels, fused
    MoE kernels, weight loader, and CodePredictor implementation.
    """

    uses_omni_engine = True

    def __init__(
        self,
        *,
        model_path: str,
        full_config,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
        block_size: int | None = None,
        tp_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        init_method: str | None = None,
        enforce_eager: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_seq_len = int(max_seq_len)
        self.tp_size = int(tp_size)
        self.enforce_eager = bool(enforce_eager)

        talker_config = full_config.talker_config
        self.config = talker_config
        self.talker_config = talker_config
        self.text_cfg = talker_config.text_config
        self.hidden_size = self.text_cfg.hidden_size

        self.vllm_config = self._build_vllm_config(
            model_path=model_path,
            talker_config=talker_config,
            dtype=dtype,
            max_seq_len=self.max_seq_len,
            block_size=block_size,
            tp_size=self.tp_size,
            enforce_eager=self.enforce_eager,
        )
        self.block_size = int(self.vllm_config.cache_config.block_size)
        _ensure_vllm_runtime(self.vllm_config, device=device, tp_size=self.tp_size,
                             rank=rank, world_size=world_size, init_method=init_method)

        logger.info(
            "Building engine Talker module on %s (max_seq_len=%d)",
            device,
            self.max_seq_len,
        )
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with set_current_vllm_config(self.vllm_config):
                self.model = EngineQwen3OmniMoeTalker(
                    vllm_config=self.vllm_config,
                    prefix="",
                ).to(device=device, dtype=dtype)
        finally:
            torch.set_default_dtype(old_default_dtype)

        self._attention_layers = {
            name: layer
            for name, layer in self.vllm_config.compilation_config.static_forward_context.items()
            if isinstance(layer, Attention)
        }
        if not self._attention_layers:
            raise RuntimeError("engine Talker did not register attention layers")

        self._num_blocks = math.ceil(self.max_seq_len / self.block_size)
        self._metadata_builder = None
        self._talker_decode_graph = None
        self._decode_graph_warmup_done = False
        self._decode_batch_desc = BatchDescriptor(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
        )
        self._decode_inputs_embeds = torch.empty(
            (1, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._decode_positions = torch.empty(
            (1,),
            dtype=torch.long,
            device=self.device,
        )
        self._decode_query_start_loc = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=self.device,
        )
        self._decode_query_start_loc_cpu = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cpu",
        )
        self._decode_seq_lens = torch.empty(
            (1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._decode_seq_lens_cpu = torch.empty(
            (1,),
            dtype=torch.int32,
            device="cpu",
        )
        self._decode_block_table = torch.arange(
            self._num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        self._decode_is_prefilling = torch.tensor(
            [False],
            dtype=torch.bool,
            device=self.device,
        )
        self._talker_mtp_graph = None
        self._mtp_graph_warmup_done = False
        self._mtp_batch_desc = BatchDescriptor(
            num_tokens=1,
            num_reqs=1,
            uniform=True,
        )
        self._mtp_input_ids = torch.empty(
            (1, 1),
            dtype=torch.long,
            device=self.device,
        )
        self._mtp_inputs_embeds = torch.empty(
            (1, 1, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._mtp_last_talker_hidden = torch.empty(
            (1, 1, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._mtp_text_step = torch.empty(
            (1, 1, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_current_vllm_config(self.vllm_config):
            self._allocate_kv_cache()

    @staticmethod
    def _build_vllm_config(
        *,
        model_path: str,
        talker_config,
        dtype: torch.dtype,
        max_seq_len: int,
        block_size: int | None,
        tp_size: int = 1,
        enforce_eager: bool = False,
    ) -> VllmConfig:
        model_config = ModelConfig(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=max_seq_len,
            skip_tokenizer_init=True,
            enforce_eager=enforce_eager,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=CacheConfig(
                block_size=block_size,
                cache_dtype="auto",
                gpu_memory_utilization=0.9,
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=int(tp_size),
                pipeline_parallel_size=1,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=max_seq_len,
                is_encoder_decoder=False,
                max_num_batched_tokens=max_seq_len,
                max_num_seqs=1,
            ),
            device_config=DeviceConfig(device="cuda"),
            load_config=LoadConfig(
                load_format="auto",
                download_dir=_hf_snapshot_cache_dir(),
            ),
        )
        return vllm_config.with_hf_config(
            talker_config,
            architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
        )

    def _allocate_kv_cache(self) -> None:
        for layer in self._attention_layers.values():
            spec = layer.get_kv_cache_spec(self.vllm_config)
            shape = layer.attn_backend.get_kv_cache_shape(
                self._num_blocks,
                self.block_size,
                spec.num_kv_heads,
                spec.head_size,
                self.vllm_config.cache_config.cache_dtype,
            )
            try:
                stride_order = layer.attn_backend.get_kv_cache_stride_order()
                assert len(stride_order) == len(shape)
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(len(shape)))
            physical_shape = tuple(shape[i] for i in stride_order)
            inv_order = [stride_order.index(i) for i in range(len(stride_order))]
            layer.kv_cache = torch.zeros(
                physical_shape,
                dtype=spec.dtype,
                device=self.device,
            ).permute(*inv_order)

        first_layer = next(iter(self._attention_layers.values()))
        spec = first_layer.get_kv_cache_spec(self.vllm_config)
        self._metadata_builder = first_layer.attn_backend.get_builder_cls()(
            spec,
            list(self._attention_layers.keys()),
            self.vllm_config,
            self.device,
        )

    def _build_attention_context(
        self,
        *,
        query_positions: torch.Tensor,
        seq_len: int,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        query_positions = query_positions.to(
            device=self.device,
            dtype=torch.long,
        ).flatten()
        q_len = int(query_positions.numel())
        if q_len <= 0:
            raise ValueError("Talker forward requires at least one query token")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Talker seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        if q_len == 1:
            query_start_loc = self._decode_query_start_loc
            query_start_loc_cpu = self._decode_query_start_loc_cpu
            seq_lens = self._decode_seq_lens
            seq_lens_cpu = self._decode_seq_lens_cpu
            seq_lens.fill_(seq_len)
            seq_lens_cpu.fill_(seq_len)
            block_table = self._decode_block_table
            is_prefilling = self._decode_is_prefilling
        else:
            query_start_loc = torch.tensor(
                [0, q_len],
                dtype=torch.int32,
                device=self.device,
            )
            query_start_loc_cpu = query_start_loc.cpu()
            seq_lens = torch.tensor(
                [seq_len],
                dtype=torch.int32,
                device=self.device,
            )
            seq_lens_cpu = seq_lens.cpu()
            block_table = torch.arange(
                self._num_blocks,
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(0)
            is_prefilling = torch.tensor(
                [True],
                dtype=torch.bool,
                device=self.device,
            )
        common = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=1,
            num_actual_tokens=q_len,
            max_query_len=q_len,
            max_seq_len=seq_len,
            block_table_tensor=block_table,
            slot_mapping=query_positions,
            causal=True,
            is_prefilling=is_prefilling,
            _seq_lens_cpu=seq_lens_cpu,
        )
        metadata = self._metadata_builder.build(0, common)
        metadata_by_layer = {
            name: metadata for name in self._attention_layers
        }
        slot_mapping_by_layer = {
            name: query_positions for name in self._attention_layers
        }
        return metadata_by_layer, slot_mapping_by_layer

    def load_weights_from_checkpoint(self) -> None:
        """Load Talker weights through vLLM's native loader/mappers."""
        with torch.cuda.device(self.device):
            loader = DefaultModelLoader(self.vllm_config.load_config)
            source = DefaultModelLoader.Source(
                model_or_path=self.model_path,
                revision=self.vllm_config.model_config.revision,
            )
            weights = (
                (name, tensor)
                for name, tensor in loader._get_weights_iterator(source)
                if name.startswith("talker.")
            )
            loaded = AutoWeightsLoader(self.model).load_weights(
                weights,
                mapper=self.model.hf_to_vllm_mapper,
            )
            logger.info(
                "engine Talker loaded %d parameter tensors",
                len(loaded),
            )
            process_weights_after_loading(
                self.model,
                self.vllm_config.model_config,
                self.device,
            )
            self.model.eval()
            self.model.requires_grad_(False)
            if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self._talker_decode_graph = CUDAGraphWrapper(
                    self._decode_runnable,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    cudagraph_options=CUDAGraphOptions(weak_ref_output=False),
                )
                self._talker_mtp_graph = CUDAGraphWrapper(
                    self._talker_mtp_runnable,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    cudagraph_options=CUDAGraphOptions(weak_ref_output=False),
                )

    @property
    def text_projection(self) -> nn.Module:
        return self.model.text_projection

    @property
    def hidden_projection(self) -> nn.Module:
        return self.model.hidden_projection

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.model.codec_embedding

    def set_code_predictor_sampling_params(self, top_k: int, top_p: float) -> None:
        self.model.code_predictor.set_sampling_params(top_k=top_k, top_p=top_p)

    def ensure_kv_cache(self, *, device: torch.device, dtype: torch.dtype) -> None:
        # The paged KV cache is allocated at construction and retained across
        # sessions; every prefill overwrites slots [0:T].
        return None

    @torch.inference_mode()
    def forward_prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.device):
            inputs_2d = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            positions_1d = position_ids.reshape(-1).to(device=self.device)
            seq_len = int(inputs_2d.shape[0])
            metadata, slots = self._build_attention_context(
                query_positions=positions_1d,
                seq_len=seq_len,
            )
            with set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=seq_len,
                slot_mapping=slots,
            ):
                hidden = self.model(
                    None,
                    positions_1d,
                    inputs_2d,
                )
                logits = self.model.compute_logits(hidden)
        return logits.unsqueeze(0), hidden.unsqueeze(0)

    @torch.inference_mode()
    def forward_decode(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.device):
            self._decode_inputs_embeds.copy_(
                inputs_embeds.reshape(1, self.hidden_size).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            self._decode_positions.copy_(
                position_ids.reshape(1).to(device=self.device, dtype=torch.long)
            )
            seq_len = int(cache_pos.item()) + 1
            metadata, slots = self._build_attention_context(
                query_positions=self._decode_positions,
                seq_len=seq_len,
            )
            runner = self._talker_decode_graph or self._decode_runnable
            use_decode_graph = (
                self._talker_decode_graph is not None
                and self._decode_graph_warmup_done
            )
            runtime_mode = (
                CUDAGraphMode.FULL
                if use_decode_graph
                else CUDAGraphMode.NONE
            )
            with set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=1,
                slot_mapping=slots,
                cudagraph_runtime_mode=runtime_mode,
                batch_descriptor=self._decode_batch_desc,
            ):
                logits, hidden = runner(
                    self._decode_inputs_embeds,
                    self._decode_positions,
                )
            if self._talker_decode_graph is not None and not use_decode_graph:
                self._decode_graph_warmup_done = True
        return logits.unsqueeze(0), hidden.unsqueeze(0)

    @torch.inference_mode()
    def code_predictor_forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        last_talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.device):
            return self.model.code_predictor_forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                last_talker_hidden=last_talker_hidden,
            )

    def _decode_runnable(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model(
            None,
            positions,
            inputs_embeds,
        )
        logits = self.model.compute_logits(hidden)
        return logits, hidden

    def _talker_mtp_runnable(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code_predictor_codes, summed_embeddings = self.model.code_predictor_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            last_talker_hidden=last_talker_hidden,
        )
        next_inputs_embeds = summed_embeddings.reshape(-1, 1, self.hidden_size)
        next_inputs_embeds = next_inputs_embeds + text_step
        return next_inputs_embeds, code_predictor_codes.squeeze(-1)

    @torch.inference_mode()
    def talker_mtp_forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the engine's graph-wrapped Talker MTP helper for one decode step."""
        with torch.cuda.device(self.device):
            self._mtp_input_ids.copy_(
                input_ids.reshape(1, 1).to(device=self.device, dtype=torch.long)
            )
            self._mtp_inputs_embeds.copy_(
                inputs_embeds.reshape(1, 1, self.hidden_size).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            self._mtp_last_talker_hidden.copy_(
                last_talker_hidden.reshape(1, 1, self.hidden_size).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            self._mtp_text_step.copy_(
                text_step.reshape(1, 1, self.hidden_size).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            runner = self._talker_mtp_graph or self._talker_mtp_runnable
            use_mtp_graph = (
                self._talker_mtp_graph is not None
                and self._mtp_graph_warmup_done
            )
            runtime_mode = (
                CUDAGraphMode.FULL
                if use_mtp_graph
                else CUDAGraphMode.NONE
            )
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=1,
                cudagraph_runtime_mode=runtime_mode,
                batch_descriptor=self._mtp_batch_desc,
            ):
                next_inputs_embeds, code_predictor_codes = runner(
                    self._mtp_input_ids,
                    self._mtp_inputs_embeds,
                    self._mtp_last_talker_hidden,
                    self._mtp_text_step,
                )
            if self._talker_mtp_graph is not None and not use_mtp_graph:
                self._mtp_graph_warmup_done = True
        return next_inputs_embeds, code_predictor_codes



@dataclass
class TalkerSpecialTokenIds:
    """Token IDs embedded into the assistant codec prefill bundle."""

    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_eos_token_id: int
    tts_pad_token_id: int
    speaker_ids: dict  # speaker_name -> speaker_id

    @classmethod
    def from_model_config(cls, full_config) -> "TalkerSpecialTokenIds":
        tc = full_config.talker_config
        return cls(
            codec_nothink_id=int(tc.codec_nothink_id),
            codec_think_bos_id=int(tc.codec_think_bos_id),
            codec_think_eos_id=int(tc.codec_think_eos_id),
            codec_pad_id=int(tc.codec_pad_id),
            codec_bos_id=int(tc.codec_bos_id),
            codec_eos_token_id=int(tc.codec_eos_token_id),
            tts_pad_token_id=int(full_config.tts_pad_token_id),
            speaker_ids=dict(getattr(tc, "speaker_id", {}) or {}),
        )


def load_vllm_talker_model(
    model_path: str,
    *,
    full_config=None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8192,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int = 1,
    init_method: str | None = None,
    enforce_eager: bool = False,
) -> Tuple[OmniTalker, TalkerSpecialTokenIds]:
    """Load the Talker using the engine's native module and kernels.

    For tp_size>1 call from each of `world_size` ranks with the shared
    `init_method`, distinct `rank`, and a per-rank `device`.
    enforce_eager=True disables torch.compile + CUDA graphs (needed when
    co-hosting with a TP AsyncOmni engine whose compile-cache device ids
    collide)."""
    if full_config is None:
        full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    special = TalkerSpecialTokenIds.from_model_config(full_config)
    torch_device = torch.device(device)
    with torch.cuda.device(torch_device):
        talker = OmniTalker(
            model_path=model_path,
            full_config=full_config,
            device=torch_device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            tp_size=tp_size,
            rank=rank,
            world_size=world_size,
            init_method=init_method,
            enforce_eager=enforce_eager,
        )
        talker.load_weights_from_checkpoint()
    logger.info("engine Talker ready on %s (dtype=%s, tp=%d rank=%d)",
                device, dtype, tp_size, rank)
    return talker, special


def build_user_part(
    *,
    talker: OmniTalker,
    thinker_embed_segment: torch.Tensor,    # [seg_len, thinker_hidden]
    thinker_hidden_segment: torch.Tensor,   # [seg_len, thinker_hidden]
    multimodal_mask_segment: torch.Tensor,  # [seg_len], bool
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Project a USER segment from thinker space to talker space.

    Mirrors ``_get_talker_user_parts``: for multimodal token positions,
    use ``hidden_projection`` on the final-layer hidden states; for text
    token positions, use ``text_projection`` on the layer-0 embeddings.
    """
    seg_len = thinker_embed_segment.shape[0]
    user_part = torch.empty(
        (seg_len, talker.config.text_config.hidden_size),
        device=thinker_embed_segment.device,
        dtype=target_dtype,
    )

    if multimodal_mask_segment.any():
        mm_hidden = thinker_hidden_segment[multimodal_mask_segment]
        projected_mm = talker.hidden_projection(mm_hidden.to(target_dtype))
        user_part[multimodal_mask_segment] = projected_mm

    if (~multimodal_mask_segment).any():
        text_embed = thinker_embed_segment[~multimodal_mask_segment]
        projected_text = talker.text_projection(text_embed.to(target_dtype))
        user_part[~multimodal_mask_segment] = projected_text

    return user_part


def build_assistant_parts(
    *,
    talker: OmniTalker,
    assistant_thinker_embed: torch.Tensor,  # [seg_len, thinker_hidden]
    speaker_id: int,
    tts_pad_embed: torch.Tensor,            # [1, talker_hidden]
    tts_bos_embed: torch.Tensor,            # [1, talker_hidden]
    tts_eos_embed: torch.Tensor,            # [1, talker_hidden]
    special: TalkerSpecialTokenIds,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the assistant prefill embeddings + trailing queue.

    Mirrors upstream ``_get_talker_assistant_parts``. The
    talker prefill assistant span is exactly 9 tokens:
      [3 chat-template tokens] + [4 tts_pad] + [tts_bos] + [first thinker
      generated token].
    Together with the user span, these become the talker's prefill
    ``inputs_embeds``. The trailing queue holds the *rest* of the thinker
    generated tokens (projected) followed by tts_eos.

    Returns:
      (assistant_prefill_embed, trailing_decode_embeds):
        - ``assistant_prefill_embed``: [9, talker_hidden]
        - ``trailing_decode_embeds``: [n_trailing, talker_hidden] -- the
          PROJECTED thinker tokens *after* position 3 of the assistant
          span, in order. Does NOT include ``tts_eos`` -- the runner
          appends that itself once the thinker session is finished, so
          new thinker decode tokens can be appended *before* it during
          streaming.
    """
    if assistant_thinker_embed.shape[0] < 4:
        raise ValueError(
            "assistant span must contain at least 4 thinker tokens "
            "(3 chat-template + 1 first-generated); got "
            f"{assistant_thinker_embed.shape[0]}"
        )

    talker_hidden = talker.config.text_config.hidden_size
    device = assistant_thinker_embed.device

    assistant_hidden = talker.text_projection(
        assistant_thinker_embed.to(target_dtype)
    )  # [seg_len, talker_hidden]

    tts_pad_2d = tts_pad_embed.reshape(1, talker_hidden)
    tts_bos_2d = tts_bos_embed.reshape(1, talker_hidden)
    assistant_text_hidden = torch.cat(
        (
            assistant_hidden[:3],
            tts_pad_2d.expand(4, -1),
            tts_bos_2d,
            assistant_hidden[3:4],
        ),
        dim=0,
    )

    codec_special_tokens = torch.tensor(
        [
            special.codec_nothink_id,
            special.codec_think_bos_id,
            special.codec_think_eos_id,
            int(speaker_id),
            special.codec_pad_id,
            special.codec_bos_id,
        ],
        device=device,
        dtype=torch.long,
    )
    codec_special_embeds = talker.get_input_embeddings()(codec_special_tokens).to(
        target_dtype
    )  # [6, talker_hidden]
    assistant_codec_hidden = torch.cat(
        (
            torch.zeros(
                (3, talker_hidden), device=device, dtype=target_dtype,
            ),
            codec_special_embeds,
        ),
        dim=0,
    )  # [9, talker_hidden]

    assistant_prefill_embed = assistant_text_hidden + assistant_codec_hidden  # [9, hidden]

    trailing_decode_embeds = assistant_hidden[4:]

    return assistant_prefill_embed, trailing_decode_embeds


def get_tts_special_embeds(
    *,
    talker: OmniTalker,
    tts_bos_thinker: torch.Tensor,
    tts_eos_thinker: torch.Tensor,
    tts_pad_thinker: torch.Tensor,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the three thinker-side TTS-marker embeddings into talker space."""
    bos = talker.text_projection(tts_bos_thinker.to(target_dtype)).reshape(1, -1)
    eos = talker.text_projection(tts_eos_thinker.to(target_dtype)).reshape(1, -1)
    pad = talker.text_projection(tts_pad_thinker.to(target_dtype)).reshape(1, -1)
    return bos, eos, pad


def split_thinker_segments(
    thinker_input_token_ids: torch.Tensor,    # [P]
    thinker_output_token_ids: torch.Tensor,   # [N]
    full_thinker_embed: torch.Tensor,         # [P+N, thinker_hidden]
    full_thinker_hidden: torch.Tensor,        # [P+N, thinker_hidden]
    multimodal_token_ids: Tuple[int, ...],
    im_start_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    system_token_id: int,
) -> Tuple[list, Tuple[int, int]]:
    """Locate the ``<|im_start|>`` segment boundaries and isolate the
    final assistant span.

    Returns:
      (user_segments, (assistant_start, assistant_end)):
        - ``user_segments``: list of (start, end) tuples covering each
          user span. System spans and history-assistant spans are
          dropped.
        - ``(assistant_start, assistant_end)``: indices of the trailing
          assistant span (the one whose response we are voicing).
    """
    full_seq = torch.cat(
        [
            thinker_input_token_ids.flatten(),
            thinker_output_token_ids.flatten(),
        ],
        dim=0,
    )
    seq_len = full_seq.shape[0]
    inp_len = thinker_input_token_ids.shape[-1]

    im_start_idxs = (thinker_input_token_ids.flatten() == im_start_token_id).nonzero(
        as_tuple=False
    ).flatten().tolist() + [seq_len]

    user_segments = []
    assistant_span: Optional[Tuple[int, int]] = None

    for i in range(len(im_start_idxs) - 1):
        s = int(im_start_idxs[i])
        e = int(im_start_idxs[i + 1])
        if s + 1 >= inp_len:
            role = assistant_token_id
        else:
            role = int(thinker_input_token_ids.flatten()[s + 1].item())
        if role == system_token_id:
            continue
        if role == user_token_id:
            user_segments.append((s, e))
            continue
        if role == assistant_token_id:
            if i == len(im_start_idxs) - 2:
                assistant_span = (s, e)
            continue

    if assistant_span is None:
        raise ValueError(
            "could not find trailing assistant span in thinker prompt"
        )
    return user_segments, assistant_span
