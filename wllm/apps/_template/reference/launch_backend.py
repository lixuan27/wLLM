import os

from wllm.apps._template.reference.worker import AppWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    AppWorker(cfg_path=_CFG_PATH).loop()
