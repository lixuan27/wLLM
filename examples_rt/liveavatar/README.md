# Running LiveAvatar

The application itself is described in `wllm/apps/liveavatar/README.md`.

## Setup

Install the repo (see `docs/setup.md`), then download the
checkpoints:

```bash
python -m wllm.weights liveavatar
```

Copy `.env.example` to `.env` at the repo root and fill in your LiveKit
credentials once (see `docs/frontends.md`).

## Run a backend

Pick the script for your GPU budget from `cuda/` (or `rocm/` if you
built the ROCm stack), for example `cuda/run_2gpu.sh`.
`run_reference.sh` is the sequential reference baseline.

The backend is up once it logs `LiveAvatar backend READY`. First
launches are slow because the engines start up and models compile. Run
one backend at a time; Ctrl-C stops all of its processes.

## Run the frontend

With a backend READY:

```bash
bash run_frontend.sh
```

The frontend prints a join URL when it starts. Open it in a browser,
allow the mic, and speak; the avatar answers.
