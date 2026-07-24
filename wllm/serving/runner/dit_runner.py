from wllm.serving.models.dit.base import BaseDiT
from wllm.serving.layers.rope.rope_3d import Wan3DRotaryPosEmbed
from wllm.serving.layers.attention_backends.base_attention_backend import BaseAttentionBackend
from wllm.serving.models.loader.component_loader import DitLoader
from wllm.serving.rt_config import RTConfig
from wllm.serving.memory.base_cache import BaseKVCache
from wllm.serving.runner.forward_batch import ForwardBatchInfo
import torch
from typing import Tuple
from time import sleep
import gc
from tqdm import tqdm

class DiTRunner:

    def __init__(self, cfg: RTConfig, dtype: torch.dtype, device: str):

        self.cfg = cfg
        self.transformer: BaseDiT = None
        self.rotary_emb: Wan3DRotaryPosEmbed = None
        self.attention_backend: BaseAttentionBackend = None
        self.kv_memory: BaseKVCache = None
        self.cudagraphs = {}
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        self.dtype = dtype
        self.device = device
        self._build()

    def _build(self):

        self._load_model()
        self._load_rope()
        self._init_memory()
        self._init_attention_backend()
        self._compile()

    def _load_model(self):
        loader = DitLoader()
        self.transformer = loader.load(
            self.cfg.model_name,
            self.cfg.transformer_path,
            dtype=self.dtype,
            device=self.device,
            lora_path=self.cfg.lora_path,
            lora_rank=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
        )

    def _load_rope(self):

        self.rotary_emb = Wan3DRotaryPosEmbed(self.cfg).freq_cis.to(self.device)

    def _init_memory(self):

        if self.cfg.kv_memory == "preallocated_prope":
            from wllm.serving.memory.preallocated_cache import PreAllocatedPropeKVCache
            self.kv_memory = PreAllocatedPropeKVCache(self.cfg, self.device)
        elif self.cfg.kv_memory == "preallocated":
            from wllm.serving.memory.preallocated_cache import PreAllocatedKVCache
            self.kv_memory = PreAllocatedKVCache(self.cfg, self.device)
        else:
            raise

    def _init_attention_backend(self):

        backend = self.cfg.attention_backend
        if backend == "auto":
            from wllm.serving.platforms import current_platform
            if current_platform.is_rocm():
                backend = "aiter"
            else:
                # prefer the fastest wheel if present, else the portable
                # SDPA path (torch's fused kernels; runs on any arch)
                try:
                    from flash_attn.cute.interface import flash_attn_func  # noqa: F401
                    backend = "fa4"
                except Exception:  # noqa: BLE001
                    backend = "sdpa"

        if backend == "sdpa":

            from wllm.serving.layers.attention_backends.sdpa_backend import SDPABackend
            self.attention_backend = SDPABackend()

        elif backend == "fa4":

            from wllm.serving.layers.attention_backends.flashattention_4_backend import FA4Backend
            self.attention_backend = FA4Backend()

        elif backend == "flashinfer":

            from wllm.serving.layers.attention_backends.flashinfer_backend import FlashInferBackend
            self.attention_backend = FlashInferBackend()

        elif backend == "aiter":

            from wllm.serving.layers.attention_backends.aiter_flash_attention_backend import AiterFlashAttentionBackend
            self.attention_backend = AiterFlashAttentionBackend()

        else:
            raise ValueError(f"unsupported attention_backend {backend!r}")

    def _compile(self):

        self.transformer.optimize()
        if self.cfg.use_cuda_graph:
            self._capture_cuda_graphs()


    def _capture_cuda_graphs(self):

        gc.collect()
        for input_arg in tqdm(self.cfg.capture_shapes,desc="capture_cuda_graphs"):
            if input_arg not in self.cudagraphs:
                self.cudagraphs[input_arg] = self._capture_one_cuda_graph(input_arg)

    def _capture_one_cuda_graph(self, input_arg: Tuple[int, int, bool]):

        rope_start, rope_end, is_cache = input_arg
        num_latents = (rope_end - rope_start) // self.cfg.kv_spatial
        static_latents = torch.full(size=(1, self.cfg.dit_config.out_channels, num_latents, self.cfg.latent_height, self.cfg.latent_width),
                                fill_value=1.0,
                                dtype=self.dtype, device=self.device)
        static_timestep = torch.full(size=(num_latents,),
                                fill_value=1.0,
                                dtype=torch.float32, device=self.device)
        static_viewmats = torch.full(size=(1, num_latents, 4, 4),
                                fill_value=1.0,
                                dtype=torch.float32, device=self.device)
        static_Ks = torch.full(size=(1, num_latents, 3, 3),
                                fill_value=1.0,
                                dtype=torch.float32, device=self.device)
        static_action = torch.full(size=(1, num_latents),
                                fill_value=1.0,
                                dtype=torch.float32, device=self.device)

        i2v_channels = self.cfg.dit_config.in_channels - self.cfg.dit_config.out_channels
        static_i2v = None
        if i2v_channels > 0:
            static_i2v = torch.zeros(
                1, i2v_channels, num_latents,
                self.cfg.latent_height, self.cfg.latent_width,
                dtype=self.dtype, device=self.device,
            )

        static_forward_batch_info = ForwardBatchInfo.init_new_dit(
            cache_start=rope_start,
            cache_end=rope_end,
            rope_start=rope_start,
            rope_end=rope_end,
            model_runner=self,
            is_cache=is_cache,
            Ks=static_Ks,
            viewmats=static_viewmats,
            action=static_action,
            i2v_condition=static_i2v,
        )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_output = self._run(
                    static_latents, static_timestep, static_forward_batch_info
                )
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)
        sleep(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.mempool):
            static_output = self._run(
                    static_latents, static_timestep,
                    static_forward_batch_info
                )

        def func(latents: torch.Tensor,
            timestep: torch.Tensor,
            viewmats: torch.Tensor,
            Ks: torch.Tensor,
            action: torch.Tensor,
            i2v_condition: torch.Tensor = None):
            static_latents.copy_(latents)
            static_timestep.copy_(timestep)
            static_forward_batch_info.viewmats.copy_(viewmats)
            static_forward_batch_info.Ks.copy_(Ks)
            static_forward_batch_info.action.copy_(action)
            if i2v_condition is not None and static_forward_batch_info.i2v_condition is not None:
                static_forward_batch_info.i2v_condition.copy_(i2v_condition)
            graph.replay()
            if static_output is not None:
                return static_output.clone()

        return func

    def _run(self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        forward_batch_info: ForwardBatchInfo):

        if forward_batch_info.prefill_cond:
            rotary_emb = None
        else:
            cond_prefix = int(getattr(forward_batch_info, "cond_prefix_tokens", 0) or 0)
            rope_start = int(forward_batch_info.rope_start)
            rope_end = int(forward_batch_info.rope_end)

            rope_start = max(0, rope_start - cond_prefix)
            rope_end = max(rope_start, rope_end - cond_prefix)
            rope_span = rope_end - rope_start

            if rope_span == 0:
                rotary_emb = self.rotary_emb[:0].to(torch.float32)
            else:
                assert rope_span <= self.rotary_emb.shape[0]
                start_mod = rope_start % self.rotary_emb.shape[0]
                end_mod = start_mod + rope_span
                if end_mod <= self.rotary_emb.shape[0]:
                    rotary_emb = self.rotary_emb[start_mod:end_mod].to(torch.float32)
                else:
                    rotary_emb = torch.cat([self.rotary_emb[start_mod:], self.rotary_emb[:end_mod - self.rotary_emb.shape[0]]], dim=0).to(torch.float32)

        return  self.transformer(
            hidden_states=latents.to(self.dtype),
            timestep=timestep.to(torch.float32),
            rotary_emb=rotary_emb,
            forward_batch_info=forward_batch_info
        )

    def run(self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        is_cache: bool,
        cache_start: int,
        cache_end: int,
        rope_start: int,
        rope_end: int,
        viewmats: torch.Tensor = None,
        Ks: torch.Tensor = None,
        action: torch.Tensor = None,
        i2v_condition: torch.Tensor = None,
        audio_input: torch.Tensor = None,
        cond_latents: torch.Tensor = None,
        prefill_cond: bool = False,
        cond_prefix_tokens: int = 0,
        ref_latents: torch.Tensor = None,
        motion_latents: torch.Tensor = None,
        motion_frames_raw: int = 0,
        motion_frames_latent: int = 0,
    ):

        forward_batch_info = ForwardBatchInfo.init_new_dit(
            cache_start=cache_start,
            cache_end=cache_end,
            rope_start=rope_start,
            rope_end=rope_end,
            is_cache=is_cache,
            viewmats=viewmats,
            Ks=Ks,
            action=action,
            i2v_condition=i2v_condition,
            audio_input=audio_input,
            cond_latents=cond_latents,
            prefill_cond=prefill_cond,
            cond_prefix_tokens=cond_prefix_tokens,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            motion_frames_raw=motion_frames_raw,
            motion_frames_latent=motion_frames_latent,
            model_runner=self
        )

        input_arg = (cache_start, cache_end, is_cache)
        # TODO: reenable cudagraphs
        if (
            input_arg in self.cudagraphs
            and audio_input is None
            and cond_latents is None
            and not prefill_cond
            and cond_prefix_tokens == 0
            and ref_latents is None
            and motion_latents is None
        ):
            return self.cudagraphs[input_arg](latents, timestep, viewmats, Ks, action, i2v_condition)
        else:
            return self._run(latents, timestep, forward_batch_info)


    def encode(self, prompt_embed: torch.Tensor):

        forward_batch_info = ForwardBatchInfo.init_new_dit(
            cache_start=0,
            cache_end=0,
            rope_start=0,
            rope_end=0,
            is_cache=False,
            viewmats=None,
            Ks=None,
            action=None,
            model_runner=self
        )

        self.transformer.fill_encoder_kv(prompt_embed, forward_batch_info)

    def clear(self):

        for key in self.cudagraphs.keys():
            self.cudagraphs[key] = None
        self.cudagraphs.clear()
