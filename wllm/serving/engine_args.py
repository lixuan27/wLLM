
# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of wLLM Inference."""
import argparse
import dataclasses
import json
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any
from wllm.serving.configs.utils import clean_cli_args
from wllm.serving.layers.quantization import QUANTIZATION_METHODS, QuantizationMethods
from wllm.serving.logger import init_logger
from wllm.serving.utils.utils import FlexibleArgumentParser, StoreBoolean


logger = init_logger(__name__)


class ExecutionMode(str, Enum):
    """
    Enumeration for different pipeline modes.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    INFERENCE = "inference"

    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [mode.value for mode in cls]


class WorkloadType(str, Enum):
    """
    Enumeration for different workload types.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image

    @classmethod
    def from_string(cls, value: str) -> "WorkloadType":
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [workload.value for workload in cls]


# args for wllm framework
@dataclasses.dataclass
class wLLMArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Running mode
    mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    workload_type: WorkloadType = WorkloadType.T2V

    # Distributed executor backend
    distributed_executor_backend: str = "mp"


    #Shared Memory Channels
    video_buffer_name: str = None
    ctrl_buffer_name: str = None
    action_buffer_name: str = None

    inference_mode: bool = True  # if False == training mode

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

   
    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = False
    dit_layerwise_offload: bool = True
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # Compilation
    enable_torch_compile: bool = False
    torch_compile_kwargs: dict[str, Any] = field(default_factory=dict)
    use_cuda_graph: bool = False
    cuda_graph_kwargs: list[dict[str, Any]] = field(default_factory=list)

    #Attention & KV memory
    attention_backend: str = None
    kv_memory: str = None

    #Precision
    dtype: str = None
    kv_dtype: str = None
    
    #video player
    max_num_frames: int = None
    max_num_actions: int = None
    height: int = None
    width: int = None
    
    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed training/inference
    master_port: int | None = None

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_txt: str | None = None

    # LTX-2 VAE tiling overrides
    ltx2_vae_tiling: bool | None = None
    ltx2_vae_spatial_tile_size_in_pixels: int | None = None
    ltx2_vae_spatial_tile_overlap_in_pixels: int | None = None
    ltx2_vae_temporal_tile_size_in_frames: int | None = None
    ltx2_vae_temporal_tile_overlap_in_frames: int | None = None
    ltx2_initial_latent_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(default_factory=lambda: {
        "transformer": True,
        "vae": True,
        "upsampler": True,
    })

    override_text_encoder_safetensors: str | None = None  # path to safetensors file for text encoder override
    override_text_encoder_quant: QuantizationMethods = None

    override_transformer_cls_name: str | None = None
    init_weights_from_safetensors: str = ""  # path to safetensors file for initial weight loading
    init_weights_from_safetensors_2: str = ""  # path to safetensors file for initial weight loading for transformer_2

    override_pipeline_cls_name: str | None = None

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = 0.875

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    def __post_init__(self):
        if self.moba_config_path:
            try:
                with open(self.moba_config_path) as f:
                    self.moba_config = json.load(f)
                logger.info("Loaded V-MoBA config from %s",
                            self.moba_config_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error("Failed to load V-MoBA config from %s: %s",
                             self.moba_config_path, e)
                raise
        self.check_wllm_args()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help=
            "The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )

        # Running mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=ExecutionMode.choices(),
            default=wLLMArgs.mode.value,
            help="The mode to run wLLM",
        )

        # Workload type
        parser.add_argument(
            "--workload-type",
            type=str,
            choices=WorkloadType.choices(),
            default=wLLMArgs.workload_type.value,
            help="The workload type",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=wLLMArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        parser.add_argument(
            "--inference-mode",
            action=StoreBoolean,
            default=wLLMArgs.inference_mode,
            help="Whether to use inference mode",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=wLLMArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=wLLMArgs.revision,
            help=
            "The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=wLLMArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=wLLMArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sp-size",
            type=int,
            default=wLLMArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=wLLMArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=wLLMArgs.hsdp_shard_dim,
            help="The data parallelism shards.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=wLLMArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Output type
        parser.add_argument(
            "--output-type",
            type=str,
            default=wLLMArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-txt",
            type=str,
            default=wLLMArgs.prompt_txt,
            help=
            "Path to a text file containing prompts (one per line) for batch processing",
        )

        # LTX-2 VAE tiling overrides
        parser.add_argument(
            "--ltx2-vae-tiling",
            action=StoreBoolean,
            default=wLLMArgs.ltx2_vae_tiling,
            help="Enable LTX-2 VAE tiling overrides.",
        )
        parser.add_argument(
            "--ltx2-vae-spatial-tile-size-in-pixels",
            type=int,
            default=wLLMArgs.ltx2_vae_spatial_tile_size_in_pixels,
            help="LTX-2 VAE spatial tile size in pixels.",
        )
        parser.add_argument(
            "--ltx2-vae-spatial-tile-overlap-in-pixels",
            type=int,
            default=wLLMArgs.ltx2_vae_spatial_tile_overlap_in_pixels,
            help="LTX-2 VAE spatial tile overlap in pixels.",
        )
        parser.add_argument(
            "--ltx2-vae-temporal-tile-size-in-frames",
            type=int,
            default=wLLMArgs.ltx2_vae_temporal_tile_size_in_frames,
            help="LTX-2 VAE temporal tile size in frames.",
        )
        parser.add_argument(
            "--ltx2-vae-temporal-tile-overlap-in-frames",
            type=int,
            default=wLLMArgs.ltx2_vae_temporal_tile_overlap_in_frames,
            help="LTX-2 VAE temporal tile overlap in frames.",
        )
        parser.add_argument(
            "--ltx2-initial-latent-path",
            type=str,
            default=wLLMArgs.ltx2_initial_latent_path,
            help="Path to load/save a precomputed LTX-2 initial latent.",
        )

        # LoRA parameters (inference-time adapter loading)
        parser.add_argument(
            "--lora-path",
            type=str,
            default=wLLMArgs.lora_path,
            help=
            "Path to a LoRA adapter (directory or HF repo id). If set, LoRA will be applied at inference.",
        )
        parser.add_argument(
            "--lora-nickname",
            type=str,
            default=wLLMArgs.lora_nickname,
            help=
            "Nickname to refer to the loaded LoRA adapter (useful for swapping).",
        )
        parser.add_argument(
            "--lora-target-modules",
            nargs="+",
            type=str,
            default=wLLMArgs.lora_target_modules,
            help=
            "Optional list of module name substrings to restrict LoRA injection (e.g. q_proj k_proj v_proj).",
        )

        # BSA runtime control (LongCat)
        parser.add_argument(
            "--enable-bsa",
            action=StoreBoolean,
            help=
            "Enable Block Sparse Attention (BSA) at runtime (overrides config).",
        )
        parser.add_argument(
            "--bsa-sparsity",
            type=float,
            help="BSA sparsity (e.g., 0.9375).",
        )
        parser.add_argument(
            "--bsa-cdf-threshold",
            type=float,
            help="BSA CDF threshold (optional).",
        )
        parser.add_argument(
            "--bsa-chunk-q",
            nargs=3,
            type=int,
            metavar=("T", "H", "W"),
            help="BSA chunk_3d_shape_q as three ints, e.g., 4 4 4.",
        )
        parser.add_argument(
            "--bsa-chunk-k",
            nargs=3,
            type=int,
            metavar=("T", "H", "W"),
            help="BSA chunk_3d_shape_k as three ints, e.g., 4 4 4.",
        )

        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=wLLMArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference." +
            "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )
        parser.add_argument(
            "--torch-compile-kwargs",
            type=str,
            default=None,
            help=
            "JSON string of kwargs to pass to torch.compile. Example: '{\"backend\":\"inductor\",\"mode\":\"reduce-overhead\"}'",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--dit-layerwise-offload",
            action=StoreBoolean,
            help="Enable layerwise CPU offload with async H2D prefetch overlap.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help=
            "Use FSDP for inference by sharding the model weights. FSDP helps reduce GPU memory usage but may introduce"
            +
            " weight transfer overhead depending on the specific setup. Enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help=
            "Pin memory for CPU offload. Only added as a temp workaround if it throws \"CUDA error: invalid argument\". "
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help=
            "Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # VSA parameters
        parser.add_argument(
            "--VSA-sparsity",
            type=float,
            default=wLLMArgs.VSA_sparsity,
            help="Validation sparsity for VSA",
        )

        # Master port for distributed training/inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=wLLMArgs.master_port,
            help="Master port for distributed training/inference",
        )

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=wLLMArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )
        parser.add_argument(
            "--override-text-encoder-safetensors",
            type=str,
            default=wLLMArgs.override_text_encoder_safetensors,
            help="Path to safetensors file for text encoder override",
        )
        parser.add_argument(
            "--override-text-encoder-quant",
            type=str,
            choices=QUANTIZATION_METHODS,
            default=wLLMArgs.override_text_encoder_quant,
            help="Quantization method for text encoder override",
        )
        parser.add_argument(
            "--override-transformer-cls-name",
            type=str,
            default=wLLMArgs.override_transformer_cls_name,
            help="Override transformer cls name",
        )
        parser.add_argument(
            "--override-pipeline-cls-name",
            type=str,
            default=wLLMArgs.override_pipeline_cls_name,
            help="Override pipeline cls name",
        )
        parser.add_argument(
            "--init-weights-from-safetensors",
            type=str,
            help="Path to safetensors file for initial weight loading")
        parser.add_argument(
            "--init-weights-from-safetensors-2",
            type=str,
            help="Path to safetensors file for initial weight loading")

        
       
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "wLLMArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == 'mode':
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, wLLMArgs.mode.value)
                kwargs['mode'] = ExecutionMode.from_string(
                    mode_value) if isinstance(mode_value, str) else mode_value
            elif attr == 'torch_compile_kwargs':
                # Parse JSON string for torch.compile kwargs
                torch_compile_kwargs_str = getattr(args, 'torch_compile_kwargs',
                                                   None)
                if torch_compile_kwargs_str:
                    try:
                        import json
                        kwargs['torch_compile_kwargs'] = json.loads(
                            torch_compile_kwargs_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON for torch_compile_kwargs: {e}"
                        ) from e
                else:
                    kwargs['torch_compile_kwargs'] = {}
            elif attr == 'workload_type':
                # Convert string to WorkloadType enum
                workload_type_value = getattr(args, 'workload_type',
                                              wLLMArgs.workload_type.value)
                kwargs['workload_type'] = WorkloadType.from_string(
                    workload_type_value) if isinstance(
                        workload_type_value, str) else workload_type_value
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check if it has a default_factory
                field = dataclasses.fields(cls)[next(
                    i for i, f in enumerate(dataclasses.fields(cls))
                    if f.name == attr)]
                if field.default_factory is not dataclasses.MISSING:
                    # Use the default_factory to create the default value
                    default_value = field.default_factory()
                else:
                    default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

        return cls(**kwargs)  # type: ignore

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "wLLMArgs":
        # Convert mode string to enum if necessary
        if 'mode' in kwargs and isinstance(kwargs['mode'], str):
            kwargs['mode'] = ExecutionMode.from_string(kwargs['mode'])

        # Convert workload_type string to enum if necessary
        if 'workload_type' in kwargs and isinstance(kwargs['workload_type'],
                                                    str):
            kwargs['workload_type'] = WorkloadType.from_string(
                kwargs['workload_type'])

        return cls(**kwargs)

    def check_wllm_args(self) -> None:
        if self.dit_layerwise_offload:
            if self.use_fsdp_inference:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling use_fsdp_inference."
                )
                self.use_fsdp_inference = False
            if self.dit_cpu_offload:
                logger.warning(
                    "dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload."
                )
                self.dit_cpu_offload = False

        # Validate mode and inference_mode consistency
        assert isinstance(
            self.mode, ExecutionMode
        ), f"Mode must be an ExecutionMode enum, got {type(self.mode)}"
        assert self.mode in ExecutionMode.choices(
        ), f"Invalid execution mode: {self.mode}"

        # Validate workload type
        assert isinstance(
            self.workload_type, WorkloadType
        ), f"Workload type must be a WorkloadType enum, got {type(self.workload_type)}"
        assert self.workload_type in WorkloadType.choices(
        ), f"Invalid workload type: {self.workload_type}"

        if self.mode in [ExecutionMode.DISTILLATION, ExecutionMode.FINETUNING
                         ] and self.inference_mode:
            logger.warning(
                "Mode is 'training' but inference_mode is True. Setting inference_mode to False."
            )
            self.inference_mode = False
        elif self.mode in [ExecutionMode.INFERENCE, ExecutionMode.PREPROCESS
                           ] and not self.inference_mode:
            logger.warning(
                "Mode is '%s' but inference_mode is False. Setting inference_mode to True.",
                self.mode)
            self.inference_mode = True

        if not self.inference_mode:
            assert self.hsdp_replicate_dim != -1, "hsdp_replicate_dim must be set for training"
            assert self.hsdp_shard_dim != -1, "hsdp_shard_dim must be set for training"
            assert self.sp_size != -1, "sp_size must be set for training"

        if self.tp_size == -1:
            self.tp_size = 1
        if self.sp_size == -1:
            self.sp_size = self.num_gpus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0, "num_gpus must >= and be divisible by sp_size"
        assert self.hsdp_replicate_dim <= self.num_gpus and self.num_gpus % self.hsdp_replicate_dim == 0, "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert self.hsdp_shard_dim <= self.num_gpus and self.num_gpus % self.hsdp_shard_dim == 0, "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)



_current_wllm_args = None


def prepare_wllm_args(argv: list[str]) -> wLLMArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    wLLMArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    wllm_args = wLLMArgs.from_cli_args(raw_args)
    global _current_wllm_args
    _current_wllm_args = wllm_args
    return wllm_args


@contextmanager
def set_current_wllm_args(wllm_args: wLLMArgs):
    """
    Temporarily set the current wllm config.
    Used during model initialization.
    We save the current wllm config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the wllm config to determine how to dispatch.
    """
    global _current_wllm_args
    old_wllm_args = _current_wllm_args
    try:
        _current_wllm_args = wllm_args
        yield
    finally:
        _current_wllm_args = old_wllm_args


def get_current_wllm_args() -> wLLMArgs:
    if _current_wllm_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the wllm config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current wllm args is not set.")
    return _current_wllm_args



def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]
