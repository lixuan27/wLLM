"""VAE encode+decode service for the krea_vae_dit_split model-graph pipeline.

Owns the VAE half of the Krea model graph: `vae_encode` (streaming causal
encode, state `vae_enc_cache`) and `vae_decode` (per-frame causal decode,
state `vae_dec_cache`). These caches are disjoint from the DiT's state, so
this service runs on its own GPU and pipelines across chunks with the DiT
service: while the DiT denoises chunk N, this service can encode chunk N+1
and decode chunk N-1.

With ``--stream-frames`` the decode emits each latent frame to the
coordinator as soon as it is decoded (``seq`` partials + a ``final``
marker), realizing the producer-side ``vae_decode → composite`` streaming
edge, instead of accumulating the whole chunk and sending it once. The
``vae.run`` call sequence (and therefore the decoder's causal cache) is
identical in both modes — only the send granularity changes.

Both ops are vendored verbatim from KreaSAMPipeline.step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


from wllm.serving.runner.vae_runner import VAERunner
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.utils.torch_utils import set_torch_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stream-frames", action="store_true",
                    help="stream each latent's decoded frames to the coordinator as produced")
    args = ap.parse_args()
    stream_frames = args.stream_frames

    set_torch_options()
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ref_cfg = KreaSAMReferenceConfig.from_yaml(args.cfg, is_path=True)
    cfg = ref_cfg.to_runtime_config()
    dtype = parse_dtype_getattr(cfg.dtype)

    print("[vae] building VAE runner ...", flush=True)
    vae = VAERunner(cfg, dtype, device)
    chunk_frames = int(cfg.chunk_size)
    scale_t = int(cfg.vae_config.scale_factor_temporal)
    # Warm up BOTH the first-chunk path AND the streaming path the live pipeline
    # uses for every chunk after the first, so no cold inductor autotune lands on
    # a live chunk. Chunk 0 encodes with stream=False and its first decoded latent
    # is is_first=True; every later chunk encodes with stream=True and every later
    # latent decodes with is_first=False. The old warmup covered only the former,
    # so the is_first=False decode autotuned on the *first live chunk* — a
    # multi-second stall when nothing warms the pipeline in front of it (the
    # frontend launcher has no warmup; the benchmark harness does, which is why it
    # only showed up live).
    warm_px = torch.zeros((1, 3, 1 + (chunk_frames - 1) * scale_t,
                           cfg.height, cfg.width), device=device, dtype=dtype)
    _ = vae.encode(warm_px, stream=False)      # first-chunk (fresh) encode
    warm_px2 = torch.zeros((1, 3, chunk_frames * scale_t, cfg.height, cfg.width),
                           device=device, dtype=dtype)
    _ = vae.encode(warm_px2, stream=True)      # streaming encode (chunk >= 1)
    dl = torch.zeros((1, cfg.vae_config.z_dim, 1, cfg.latent_height, cfg.latent_width),
                     device=device, dtype=dtype)
    vae.run(dl, True)                          # first decoded latent (is_first)
    vae.run(dl, False)                         # streaming decoded latents (is_first=False)
    vae.clear()
    print("[vae] ready", flush=True)

    from wllm.apps.krea_sam.backend.cuda.engine.ipc import connect_to_coordinator
    conn = connect_to_coordinator(args.address)
    conn.send({"ack": "ready"})

    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "stop":
                break
            elif cmd in ("start", "reset"):
                vae.clear()
                conn.send({"ack": cmd})
            elif cmd == "encode":
                block_idx = msg["block_idx"]
                raw = msg["frames"]  # [T,H,W,3] uint8
                ten = torch.from_numpy(raw).to(device=device, dtype=torch.uint8)
                pixels = (ten.permute(3, 0, 1, 2).unsqueeze(0).to(dtype)
                          .div_(127.5).sub_(1.0).contiguous())  # [1,C,T,H,W]
                input_latents = vae.encode(pixels, stream=block_idx > 0).to(device=device, dtype=dtype)
                if int(input_latents.shape[2]) < chunk_frames:
                    conn.send({"id": msg["id"], "stage": "encode", "latents": None})
                else:
                    input_latents = input_latents[:, :, -chunk_frames:].contiguous()
                    conn.send({"id": msg["id"], "stage": "encode",
                               "latents": input_latents.float().cpu().numpy()})
            elif cmd == "decode":
                block_idx = msg["block_idx"]
                denoised = torch.from_numpy(msg["latents"]).to(device=device, dtype=dtype)
                if stream_frames:
                    # producer-side streaming: emit each latent's decoded frames
                    # the instant it is decoded (same vae.run sequence as batched).
                    seq = 0
                    for frame_i in range(int(denoised.shape[2])):
                        latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
                        is_first = (block_idx == 0 and frame_i == 0)
                        decoded_i = vae.run(latent_i, is_first)
                        conn.send({"id": msg["id"], "stage": "decode", "seq": seq,
                                   "frames": decoded_i[0].cpu().numpy(), "final": False})
                        seq += 1
                    conn.send({"id": msg["id"], "stage": "decode", "final": True})
                else:
                    chunk_video = []
                    for frame_i in range(int(denoised.shape[2])):
                        latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
                        is_first = (block_idx == 0 and frame_i == 0)
                        decoded_i = vae.run(latent_i, is_first)
                        chunk_video.append(decoded_i[0].cpu().numpy())
                    conn.send({"id": msg["id"], "stage": "decode",
                               "out": np.concatenate(chunk_video, axis=0)})
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
