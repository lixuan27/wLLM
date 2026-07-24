import os

from wllm.apps.longlive.reference.worker import LongLiveWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    LongLiveWorker(cfg_path=_CFG_PATH).loop()
