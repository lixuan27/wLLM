# Running WorldPlay

The application itself is described in `wllm/apps/worldplay/README.md`.

## Setup

Install the repo (see `docs/setup.md`), then download the
checkpoints:

```bash
python -m wllm.weights worldplay
```

## Run a backend

Pick the script for your GPU budget from `cuda/` (or `rocm/` if you
built the ROCm stack), for example `cuda/run_4gpu_latency.sh`. A
`_latency` or `_fps` suffix selects the variant tuned for lower latency
or higher frame rate at that GPU count. `run_reference.sh` is the
sequential reference baseline.

The backend is up once it logs `WorldPlay backend READY`. The first
launch takes a while (model load plus compile warmup); later launches
reuse the caches. Run one backend at a time; Ctrl-C stops it.

## Run the frontend

With a backend READY:

```bash
bash run_frontend.sh
```

Open `http://localhost:8080` (forward the port if the machine is
remote), press Start, and drive with WASD / arrow keys. You can upload a
custom start image while in the waiting state.
