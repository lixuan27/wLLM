# Running Krea-Realtime + SAM3

The application itself is described in `wllm/apps/krea_sam/README.md`.

## Setup

Install the repo (see `docs/setup.md`), then download the checkpoints:

```bash
python -m wllm.weights krea_sam
```

Copy `.env.example` to `.env` at the repo root and fill in your LiveKit
credentials once (see `docs/frontends.md`).

## Run a backend

Pick the script for your GPU budget from `cuda/` (or `rocm/` if you
built the ROCm stack), for example `cuda/run_4gpu.sh`.
`run_reference.sh` is the sequential reference baseline.

The backend is up once it logs `KreaSAM backend READY`. The first
frames after a session starts can take extra time while kernels compile.
Run one backend at a time; Ctrl-C stops all of its processes.

## Run the frontend

With a backend READY:

```bash
bash run_frontend.sh
```

The frontend prints a join URL when it starts. Open it in a browser and
allow the webcam; the restyled stream plays back in the room.
