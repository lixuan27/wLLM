# `wllm/apps/<app>/backend/<platform>/` — template

This is one platform's optimization workspace, seeded from
`wllm/apps/_template/`. Rename this `platform_name/` directory to the
platform you target (`cuda` or `rocm`); the `backend/launch.py`
dispatcher one level up routes to it at runtime. To restart from a clean
seed, recopy it with:

```bash
cp -r wllm/apps/_template/backend wllm/apps/<app>/backend
```

The agent fills this workspace in and rewrites this `README.md` in place
to describe:

1. The IR conversion status (graphs built, validation result, analysis
   summary).
2. The deployment variants produced, the current best variant, and the IR
   analysis results that motivated each one.
3. How the user should launch the best variant end-to-end against the
   unchanged frontend under `wllm/apps/<app>/frontend/`.

See `wllm/apps/AGENTS.md` for the layout conventions, the harness
patterns, the experiment-log format, and the process hygiene rules that
apply inside this directory. See the repo-root `AGENTS.md` for the overall
mission, the IR workflow, and the hard constraints.
