# `<app>` frontend

Put the app's frontend here: whatever serves the user-facing surface and
drives a running backend through the app's `adapter.py`.

Two shapes exist in this repo:

- A self-hosted web page: an aiohttp server that serves an HTML client and
  bridges it to the adapter over WebRTC. See
  `wllm/apps/worldplay/frontend/`.
- A LiveKit publisher: a process that joins a LiveKit room, publishes the
  backend's output track(s), and forwards user input to the adapter. Use the
  shared helpers in `wllm/frontend/livekit_utils.py` for credentials
  (`.env` at the repo root) and for printing a ready-to-open
  meet.livekit.io join URL. See `docs/frontends.md`.

The frontend must talk to backends only through the adapter, so it works
unchanged against the reference backend and every optimized variant.
