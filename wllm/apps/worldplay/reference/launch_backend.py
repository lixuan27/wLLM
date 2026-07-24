import os

from wllm.apps.worldplay.reference.worker import WorldPlayWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    WorldPlayWorker(cfg_path=_CFG_PATH).loop()
