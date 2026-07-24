import sys
sys.path.insert(0, "/public/home/lixuan/lixuan/wllm-infra")
from wllm.serving.weights.components import wan_5b_ar_config
from wllm.serving.weights.convert import generator_pt_to_safetensors
generator_pt_to_safetensors(
    "checkpoints/LongLive-2.0-5B/model_bf16.pt",
    "checkpoints/longlive-2.0-5b",
    None,
    wan_5b_ar_config("LongLiveTransformer3DModel"),
)
print("CONVERT_OK")
