"""combined_stream_pp_vae: streaming + 5-stage (4 DiT + 1 VAE) pipeline parallelism.

The full 5-GPU pipeline: denoising steps 0..3 on ranks 0..3 (one per GPU,
cross-chunk pipelined) PLUS the causal VAE decode as a dedicated 5th rank. Extends
combined_stream_pp (which kept the VAE on the worker GPU) by promoting the VAE to
a first-class pipeline stage, which needs GPU-to-GPU P2P (see cluster.py).
"""
