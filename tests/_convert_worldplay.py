import os
import sys

sys.path.insert(0, "/public/home/lixuan/lixuan/wllm-infra")
from wllm.serving.weights.components import wan_5b_ar_config
from wllm.serving.weights.convert import generator_pt_to_safetensors

SRC = "checkpoints/hy_worldplay/model.pt"
OUT = "checkpoints/worldplay-5b"

assert os.path.getsize(SRC) == 42346502132, "source pickle size mismatch"
generator_pt_to_safetensors(
    SRC,
    OUT,
    "bfloat16",
    wan_5b_ar_config("WanTransformer3DModel"),
)
sz = os.path.getsize(os.path.join(OUT, "diffusion_pytorch_model.safetensors"))
print(f"safetensors bytes: {sz}")
assert sz > 5_000_000_000, "converted file suspiciously small"
print("CONVERT_OK")
