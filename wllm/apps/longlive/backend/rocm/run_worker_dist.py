"""torchrun entry point for the multi-GPU LongLive backends.

Every rank runs this and dispatches to the variant ``main(cfg_path)`` named by
``--sp-main`` (``module:func``), which sets up the distributed world and then
runs either the rank-0 coordinator or a follower loop. The coordinator emits
the READY marker once the whole topology is up.

``wllm/apps/longlive/backend/launch.py`` invokes this under
``torch.distributed.run``; it is not meant to be run directly.
"""

from __future__ import annotations

import argparse
import importlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sp-main", required=True, help="'module:func' entry for this variant")
    ap.add_argument("--cfg", required=True, help="runtime config YAML")
    args = ap.parse_args()

    mod_name, fn_name = args.sp_main.split(":")
    getattr(importlib.import_module(mod_name), fn_name)(args.cfg)


if __name__ == "__main__":
    main()
