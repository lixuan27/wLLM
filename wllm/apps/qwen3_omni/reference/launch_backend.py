import os

from wllm.apps.qwen3_omni.reference.worker import Qwen3OmniWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    Qwen3OmniWorker(cfg_path=_CFG_PATH).loop()
