import os

from wllm.apps.krea_sam.reference.worker import KreaSAMWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    KreaSAMWorker(cfg_path=_CFG_PATH).loop()
