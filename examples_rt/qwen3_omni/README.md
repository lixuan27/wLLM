# Running Qwen3-Omni

The application itself is described in `wllm/apps/qwen3_omni/README.md`.

## Setup

Install the repo (see `docs/setup.md`). The model loads by
HuggingFace repo id into your HF cache on first launch; there is nothing
to download ahead of time.

## Run a backend

Pick the script for your GPU budget from `cuda/` (or `rocm/` if you
built the ROCm stack), for example `cuda/run_2gpu.sh`.
`run_reference.sh` is the sequential reference baseline.

The backend is up once it logs `Qwen3-Omni backend READY`; engine
startup takes a few minutes. The first prompt includes warmup, so judge
latency from the second prompt onward. Run one backend at a time;
Ctrl-C stops it.

## Run the frontend

With a backend READY:

```bash
bash run_frontend.sh
```

Open `http://localhost:8080` (forward the port if the machine is
remote), type a prompt, and the response audio streams into the page as
it is generated.
