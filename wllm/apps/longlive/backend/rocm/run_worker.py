"""Single-GPU LongLive backend runner.

Constructs the worker class named by ``--worker`` against ``--cfg`` and runs
its loop. The worker class must accept ``cfg_path=<path>`` and expose
``loop()``. The READY marker is emitted after construction, i.e. after model
load and warmup, so it means the backend can actually serve.

``wllm/apps/longlive/backend/launch.py`` invokes this; it is not meant to be
run directly.
"""

from __future__ import annotations

import argparse
import importlib

from wllm.serving.logger import init_logger

logger = init_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", required=True, help="'module.path:ClassName' of the worker")
    ap.add_argument("--cfg", required=True, help="runtime config YAML")
    args = ap.parse_args()

    mod_name, cls_name = args.worker.split(":")
    worker_cls = getattr(importlib.import_module(mod_name), cls_name)

    worker = worker_cls(cfg_path=args.cfg)
    logger.info("LongLive backend READY")
    worker.loop()


if __name__ == "__main__":
    main()
