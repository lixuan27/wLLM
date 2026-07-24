"""Direction #12: exact 2-GPU plan for Wan2.2 — CFG branch parallelism.

The denoise loop runs classifier-free guidance: every step evaluates the
transformer on the conditional and unconditional branches.  This plan
places one branch per GPU (symmetric ranks, all_gather of the two noise
predictions, identical CFG combine + scheduler step on both ranks — no
broadcasts, no divergence points), then rank0 decodes.

Exactness: each branch's math is identical to the single-GPU run of the
same vendored loop at batch=1 per branch.  The single-GPU reference here
runs the SAME loop (sequential branches, batch=1) so the comparison is
like-for-like numerics; the batched-CFG (batch=2) variant is measured
separately since batching may legally retile kernels.

Run modes (WLLM_CFGPAR_MODE):
  ref1     single GPU, sequential branches (reference + anchor timing)
  batched  single GPU, batched CFG (the pipeline's native shape)
  par2     torchrun 2-proc, one branch per rank
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = "/public/home/lixuan/lixuan/pretrained-model/Wan2.2-TI2V-5B-Diffusers"
PROMPT = ("A red vintage car drives along a coastal road at sunset, "
          "waves crashing on the rocks, cinematic lighting.")
NEG = "low quality, blurry, distorted"
H, W, FRAMES, STEPS, SEED, CFG = 480, 832, 33, 20, 1234, 5.0
MODE = os.environ.get("WLLM_CFGPAR_MODE", "ref1")


def main() -> int:
    import torch
    from diffusers import WanPipeline

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl", device_id=torch.device(device))

    pipe = WanPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to(device)
    if rank == 0:
        print(f"[load] mode={MODE} world={world}", flush=True)

    with torch.inference_mode():
        pe, ne = pipe.encode_prompt(
            prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
            device=device)

    tr, sch = pipe.transformer, pipe.scheduler
    z_dim = tr.config.out_channels
    lat_h, lat_w = H // 16, W // 16
    lat_f = (FRAMES - 1) // 4 + 1
    shape = (1, z_dim, lat_f, lat_h, lat_w)

    def denoise() -> torch.Tensor:
        gen = torch.Generator(device).manual_seed(SEED)
        lat = torch.randn(shape, generator=gen, device=device,
                          dtype=torch.float32)
        sch.set_timesteps(STEPS, device=device)
        for t in sch.timesteps:
            ts = t.expand(1)
            with torch.inference_mode():
                if MODE == "batched":
                    both = torch.cat([lat, lat]).to(torch.bfloat16)
                    emb = torch.cat([pe, ne])
                    out = tr(hidden_states=both, timestep=ts.expand(2),
                             encoder_hidden_states=emb, return_dict=False)[0]
                    cond, unc = out.chunk(2)
                elif MODE == "par2":
                    my_emb = pe if rank == 0 else ne
                    mine = tr(hidden_states=lat.to(torch.bfloat16), timestep=ts,
                              encoder_hidden_states=my_emb,
                              return_dict=False)[0].contiguous()
                    import torch.distributed as dist
                    pair = [torch.empty_like(mine) for _ in range(2)]
                    dist.all_gather(pair, mine)
                    cond, unc = pair[0], pair[1]
                else:  # ref1: sequential branches, batch=1 each
                    cond = tr(hidden_states=lat.to(torch.bfloat16), timestep=ts,
                              encoder_hidden_states=pe, return_dict=False)[0]
                    unc = tr(hidden_states=lat.to(torch.bfloat16), timestep=ts,
                             encoder_hidden_states=ne, return_dict=False)[0]
            pred = unc.float() + CFG * (cond.float() - unc.float())
            lat = sch.step(pred, t, lat, return_dict=False)[0]
        return lat

    e2e = os.environ.get("WLLM_CFGPAR_E2E", "0") == "1"

    if e2e and hasattr(pipe.vae, "enable_tiling"):
        # full-clip conv3d decode peaks over HBM with the transformer
        # resident (OOM in job 196279); tiling bounds the activation peak.
        # Applied identically in every mode — the comparison stays fair.
        pipe.vae.enable_tiling()

    @torch.inference_mode()   # job 196279/196282 OOM root cause: decode ran
    def decode(lat: torch.Tensor) -> torch.Tensor:   # with autograd tracking
        """Denormalize + VAE decode, as the pipeline's own call path does."""
        vae = pipe.vae
        mean = (torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1).to(device))
        inv_std = (1.0 / torch.tensor(vae.config.latents_std)
                   .view(1, vae.config.z_dim, 1, 1, 1)).to(device)
        z = lat.to(torch.bfloat16) / inv_std.to(torch.bfloat16) \
            + mean.to(torch.bfloat16)
        video = vae.decode(z, return_dict=False)[0]
        return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)

    def run_once():
        lat = denoise()
        frames = None
        if e2e and rank == 0:
            frames = decode(lat)
        if world > 1:
            import torch.distributed as dist
            dist.barrier()   # rank1 waits so E2E wall time is symmetric
        return lat, frames

    lat, frames = run_once()  # warmup
    torch.cuda.synchronize()
    times = []
    for _ in range(2):
        t0 = time.monotonic()
        lat, frames = run_once()
        torch.cuda.synchronize()
        times.append((time.monotonic() - t0) * 1000.0)
    med = sorted(times)[len(times) // 2]

    if rank == 0:
        out = ROOT / "benchmarks/results"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(lat.cpu(), out / f"wan22_cfgpar_latent_{MODE}.pt")
        if frames is not None:
            torch.save(frames.cpu(), out / f"wan22_cfgpar_frames_{MODE}.pt")
        rec = {"mode": MODE, "world": world, "median_ms": med,
               "times_ms": times, "steps": STEPS, "frames": FRAMES,
               "h": H, "w": W, "seed": SEED, "cfg": CFG}
        (out / f"wan22_cfgpar_{MODE}_{time.strftime('%H%M%S')}.json"
         ).write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1), flush=True)
        print(f"CFGPAR_{MODE.upper()}_OK", flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
