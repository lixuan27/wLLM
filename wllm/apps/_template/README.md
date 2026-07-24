# `wllm/apps/_template/` — new application template

Copy this directory to start a new application:

```bash
cp -r wllm/apps/_template wllm/apps/<app>
```

Then fill in, in this order:

1. `config.yaml` — model paths, generation settings, and the shared-memory
   buffer names. Pick one fixed name per buffer (`<app>_video`, `<app>_ctrl`,
   ...). Every backend for the application creates buffers under these
   exact names, which is what lets one frontend attach to any backend.
2. `adapter.py` — the IPC contract. The adapter is the only interface between
   the frontend and a backend: it attaches to the buffers the backend
   creates, sends control opcodes (start / terminate / reset), pushes inputs,
   and reads outputs. Both the reference backend and any agent-written
   backend must honor it exactly.
3. `reference/` — the sequential single-GPU reference backend: a pipeline
   that strings your models together (reuse the model runners under
   `wllm/runner/` and engines like vLLM where they fit) and a worker that
   owns the buffers and the control loop. Keep it simple and readable; it is
   what the optimization agent validates against, not the fast path. Log
   a `<App> backend READY` line once warmup is done; the frontend docs
   tell users to wait for it.
4. `frontend/` — however the application is driven: a web page + server (see
   `wllm/apps/worldplay/frontend/`) or a LiveKit publisher (see
   `wllm/frontend/livekit_utils.py` and `docs/frontends.md`).
5. This `README.md` — rewrite it as a short description of what the
   application does. How-to-run documentation is written later, under
   `examples/<app>/`, when you publish the application.

`backend/` stays as seeded: it is the optimization agent's workspace. Once
the reference runs end-to-end, point the agent at your application (see
`docs/adding_an_application.md` for a prompt template) and it fills
`backend/` with IR conversion, optimized deployment variants, and launch
tooling.

After copying, update the module paths in the stubs: replace
`wllm.apps._template` with `wllm.apps.<app>` and rename the `App*`
classes.

`wllm/apps/worldplay/` is a complete worked example of all of the above.
