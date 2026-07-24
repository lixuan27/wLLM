from wllm.serving.runner.dit_runner import DiTRunner
from wllm.serving.models.loader.component_loader import DitLoader
from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.forward_batch import ForwardBatchInfo
import torch
import gc
from tqdm import tqdm


class DualDiTRunner(DiTRunner):
    def __init__(self, cfg: RTConfig, dtype: torch.dtype, device: str):
        self.transformer_low = None
        self.cudagraphs_low = {}
        self.boundary = cfg.boundary_ratio * 1000.0 if cfg.boundary_ratio else 947.0
        self._encoder_kv_high = None  # list of (k, v) tuples per layer
        self._encoder_kv_low = None
        self._active_encoder = None
        super().__init__(cfg, dtype, device)

    def _install_encoder_kv(self, model_name: str):
        if model_name == self._active_encoder:
            return
        source = self._encoder_kv_high if model_name == "high" else self._encoder_kv_low
        if source is None:
            return
        mem = self.kv_memory
        for i, (k, v) in enumerate(source):
            mem.encoder_k_cache[i].copy_(k)
            mem.encoder_v_cache[i].copy_(v)
        self._active_encoder = model_name

    def _load_model(self):
        loader_high = DitLoader()
        self.transformer = loader_high.load(
            self.cfg.model_name,
            self.cfg.transformer_path_high or self.cfg.transformer_path,
            dtype=self.dtype,
            device=self.device,
        )
        loader_low = DitLoader()
        self.transformer_low = loader_low.load(
            self.cfg.model_name,
            self.cfg.transformer_path_low or self.cfg.transformer_path,
            dtype=self.dtype,
            device=self.device,
        )

    def _compile(self):
        self.transformer.optimize()
        self.transformer_low.optimize()
        if self.cfg.use_cuda_graph:
            self._capture_cuda_graphs()

    def _capture_cuda_graphs(self):
        gc.collect()

        # High-noise model (self.transformer already points to it)
        for input_arg in tqdm(self.cfg.capture_shapes, desc="capture_cuda_graphs (high)"):
            if input_arg not in self.cudagraphs:
                self.cudagraphs[input_arg] = self._capture_one_cuda_graph(input_arg)

        # Low-noise model: temporarily swap self.transformer
        saved = self.transformer
        self.transformer = self.transformer_low
        for input_arg in tqdm(self.cfg.capture_shapes, desc="capture_cuda_graphs (low)"):
            if input_arg not in self.cudagraphs_low:
                self.cudagraphs_low[input_arg] = self._capture_one_cuda_graph(input_arg)
        self.transformer = saved

    def _select(self, timestep_value: float):
        if timestep_value >= self.boundary:
            return self.transformer, self.cudagraphs, "high"
        return self.transformer_low, self.cudagraphs_low, "low"

    def encode(self, prompt_embed: torch.Tensor):
        mem = self.kv_memory
        num_layers = self.cfg.dit_config.num_layers

        # Encode with high-noise model
        super().encode(prompt_embed)
        self._encoder_kv_high = [
            (mem.encoder_k_cache[i].clone(), mem.encoder_v_cache[i].clone())
            for i in range(num_layers)
        ]

        # Encode with low-noise model
        saved = self.transformer
        self.transformer = self.transformer_low
        super().encode(prompt_embed)
        self.transformer = saved
        self._encoder_kv_low = [
            (mem.encoder_k_cache[i].clone(), mem.encoder_v_cache[i].clone())
            for i in range(num_layers)
        ]

        self._active_encoder = None
        self._install_encoder_kv("high")

    def run(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        is_cache: bool,
        cache_start: int,
        cache_end: int,
        rope_start: int,
        rope_end: int,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        action: torch.Tensor,
        i2v_condition: torch.Tensor = None,
        timestep_value: float = 1000.0,
    ):
        transformer, cudagraphs, model_name = self._select(timestep_value)

        # Install the correct encoder KV for this model
        self._install_encoder_kv(model_name)

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
            model_runner=self,
        )

        input_arg = (cache_start, cache_end, is_cache)
        if input_arg in cudagraphs:
            return cudagraphs[input_arg](latents, timestep, viewmats, Ks, action, i2v_condition)
        else:
            # Temporarily swap so _run uses the correct model
            saved = self.transformer
            self.transformer = transformer
            result = self._run(latents, timestep, forward_batch_info)
            self.transformer = saved
            return result

    def clear(self):
        super().clear()
        for key in self.cudagraphs_low:
            self.cudagraphs_low[key] = None
        self.cudagraphs_low.clear()
