import os

from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker


_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


if __name__ == "__main__":
    worker = None
    try:
        worker = LiveAvatarWorker(cfg_path=_CFG_PATH)
        worker.loop()
    finally:
        # Always run terminate() on Ctrl+C / graceful TERM / exit so processes and
        # shm buffers are cleaned up instead of leaked. terminate() is idempotent.
        if worker is not None:
            worker.terminate()
