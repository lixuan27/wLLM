import torch
import torch.distributed as dist

from wllm.serving.rt_config import RTConfig
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.memory.base_cache import BaseKVCache
from wllm.serving.memory.base_prope_cache import BasePropeKVCache
from wllm.serving.distributed.parallel_state import get_sp_world_size
from wllm.serving.distributed.utils import padded_head_count_for_sp
from wllm.serving.logger import init_logger
logger = init_logger(__name__)

class PreAllocatedKVCache(BaseKVCache):
    def __init__(self, cfg: RTConfig, deivce: torch.device):
        super().__init__(cfg, deivce)

        self.dtype = parse_dtype_getattr(self.cfg.kv_dtype)

        self.k_cache = None
        self.v_cache = None

        self.encoder_k_cache = None
        self.encoder_v_cache = None

        self.alloc()

    def alloc(self):
        world_size = get_sp_world_size()
        cond_tokens = getattr(self.cfg, "kv_cond_tokens", 0)
        total_kv_tokens = (self.cfg.chunk_size + self.cfg.context_window_size) * self.cfg.kv_spatial + cond_tokens

        # Heads may be padded up so the global count is divisible by SP and
        # each rank's head shard is even (rope kernel needs BH=2). Each rank
        # then stores H_padded / world_size heads.
        H_padded = padded_head_count_for_sp(self.cfg.dit_config.num_key_value_heads, world_size)
        head_shard = H_padded // world_size

        self.k_cache = [
            torch.empty(
                (
                    total_kv_tokens,
                    head_shard,
                    self.cfg.dit_config.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.v_cache = [
            torch.empty(
                (
                    total_kv_tokens,
                    head_shard,
                    self.cfg.dit_config.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.encoder_k_cache = [
            torch.empty(
                (1, self.cfg.max_sequence_length, self.cfg.dit_config.num_key_value_heads, self.cfg.dit_config.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.encoder_v_cache = [
            torch.empty(
                (1, self.cfg.max_sequence_length, self.cfg.dit_config.num_key_value_heads, self.cfg.dit_config.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        total_bytes = 0
        for caches in (self.k_cache, self.v_cache, self.encoder_k_cache, self.encoder_v_cache):
            for t in caches:
                total_bytes += t.numel() * t.element_size()

        total_gb = total_bytes / (1024 ** 3)
        logger.info(f"PreAllocatedKVCache allocated total ~{total_gb:.3f} GiB on device={self.device}, dtype={self.dtype}")

    def get_kv(self, layer_id: int, offset: int):
        k_all = self.k_cache[layer_id].narrow(dim=0, start=0, length=offset).unsqueeze(0)
        v_all = self.v_cache[layer_id].narrow(dim=0, start=0, length=offset).unsqueeze(0)
        return k_all, v_all

    def set_encoder_kv(self, layer_id: int, encoder_k: torch.Tensor, encoder_v: torch.Tensor):
        self.encoder_k_cache[layer_id].copy_(encoder_k)
        self.encoder_v_cache[layer_id].copy_(encoder_v)

    def get_encoder_kv(self, layer_id: int):
        return self.encoder_k_cache[layer_id], self.encoder_v_cache[layer_id]

    def get_k_ptr(self, layer_id: int):
        return self.k_cache[layer_id]

    def get_v_ptr(self, layer_id: int):
        return self.v_cache[layer_id]

class PreAllocatedPropeKVCache(BasePropeKVCache):
    """
    Pre-allocate KV cache tensors for all layers.

    Caches:
      - k_cache / v_cache: decoder KV cache, sharded by tensor parallel world size.
      - encoder_k_cache / encoder_v_cache: encoder KV cache, not sharded.
    """

    def __init__(self, cfg: RTConfig, deivce: torch.device):
        super().__init__(cfg, deivce)

        self.dtype = parse_dtype_getattr(self.cfg.kv_dtype)

        self.k_cache = None
        self.v_cache = None

        self.encoder_k_cache = None
        self.encoder_v_cache = None

        self.alloc()

    def alloc(self):
        """
        Allocate KV caches for all layers.
        """
        world_size = get_sp_world_size()
        cond_tokens = self.cfg.kv_cond_tokens or 0
        total_kv_tokens = (self.cfg.chunk_size + self.cfg.context_window_size) * self.cfg.kv_spatial + cond_tokens

        self.k_cache = [
            torch.empty(
                (
                    2,
                    total_kv_tokens,
                    self.cfg.dit_config.num_key_value_heads // world_size,
                    self.cfg.dit_config.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.v_cache = [
            torch.empty(
                (
                    2,
                    total_kv_tokens,
                    self.cfg.dit_config.num_key_value_heads // world_size,
                    self.cfg.dit_config.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.encoder_k_cache = [
            torch.empty(
                (1, self.cfg.max_sequence_length, self.cfg.dit_config.num_key_value_heads, self.cfg.dit_config.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        self.encoder_v_cache = [
            torch.empty(
                (1, self.cfg.max_sequence_length, self.cfg.dit_config.num_key_value_heads, self.cfg.dit_config.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.cfg.dit_config.num_layers)
        ]

        # ---- Logging: total allocated size (GB) ----
        total_bytes = 0
        for caches in (self.k_cache, self.v_cache, self.encoder_k_cache, self.encoder_v_cache):
            for t in caches:
                total_bytes += t.numel() * t.element_size()

        total_gb = total_bytes / (1024 ** 3)
        logger.info(f"PreAllocatedKVCache allocated total ~{total_gb:.3f} GiB on device={self.device}, dtype={self.dtype}")

    def get_kv(self, layer_id: int, offset: int):
        k_all = self.k_cache[layer_id].narrow(dim=1, start=0, length=offset)
        v_all = self.v_cache[layer_id].narrow(dim=1, start=0, length=offset)
        return k_all, v_all

    def set_encoder_kv(self, layer_id: int, encoder_k: torch.Tensor, encoder_v: torch.Tensor):
        self.encoder_k_cache[layer_id].copy_(encoder_k)
        self.encoder_v_cache[layer_id].copy_(encoder_v)

    def get_encoder_kv(self, layer_id: int):
        return self.encoder_k_cache[layer_id], self.encoder_v_cache[layer_id]
    
    def get_rope_k_ptr(self, layer_id: int):

        return self.k_cache[layer_id][0]
    
    def get_rope_v_ptr(self, layer_id: int):

        return self.v_cache[layer_id][0]
    
    def get_prope_k_ptr(self, layer_id: int):

        return self.k_cache[layer_id][1]
    
    def get_prope_v_ptr(self, layer_id: int):

        return self.v_cache[layer_id][1]