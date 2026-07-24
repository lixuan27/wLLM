"""Dataset loaders for multi-sample calibration (and dataset A/B tests).

Current contents:

* :mod:`wllm_native.datasets.libero` — LeRobot-v2 LIBERO layout
  (``meta/info.json`` + ``data/chunk-000/episode_XXXXXX.parquet``).

These modules are framework-agnostic: they return plain ``dict`` obs
shaped like the frontend ``infer(obs)`` contract, so the same calibration
pipeline works across RTX / Thor / JAX frontends.
"""
