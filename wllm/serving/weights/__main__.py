"""Download the checkpoints an application needs.

Usage:
  python -m wllm.weights <app> [<app> ...]
  python -m wllm.weights --all
  python -m wllm.weights <app> --dry-run

Components shared between apps (the Wan text encoder, tokenizer, and
VAEs) download once into checkpoints/wan/ and are reused by every app
that needs them.
"""

from __future__ import annotations

import argparse
import os

from wllm.serving.weights.engine import CHECKPOINTS_DIR, ensure_all

_APPS = ("worldplay", "liveavatar", "krea_sam", "qwen3_omni", "longlive")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download (and, where needed, convert) app checkpoints.")
    ap.add_argument("apps", nargs="*", choices=_APPS,
                    help=f"one or more of: {', '.join(_APPS)}")
    ap.add_argument("--all", action="store_true", help="all applications")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be downloaded and exit")
    args = ap.parse_args()

    apps = list(_APPS) if args.all else args.apps
    if not apps:
        ap.error("name at least one app or pass --all")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"[weights] checkpoints directory: {CHECKPOINTS_DIR}")
    ensure_all(apps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
