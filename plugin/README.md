# wLLM Claude Code plugin

A thin shell around the wLLM control plane: it adds discovery (one
skill) and transport (one MCP server entry) and nothing else. All
judgment — gates, thresholds, receipts, the rollback chain — lives in
the control plane and runs identically with no agent present.

## Install

1. Install the package; this provides the `wllm` and `wllm-mcp`
   console scripts:

   ```bash
   pip install -e /path/to/wllm-infra
   ```

2. Load the plugin directory:

   ```bash
   claude --plugin-dir /path/to/wllm-infra/plugin
   ```

   or, if the repo is registered as a plugin marketplace, add it with
   `/plugin marketplace add` and install `wllm` from there.

Fallback: if `wllm-mcp` is not on PATH (package importable but console
scripts not installed), edit `.mcp.json` to run the module directly:

```json
{"mcpServers": {"wllm": {"command": "python",
                         "args": ["-m", "wllm.control.mcp"]}}}
```

## Tools

The MCP server exposes the six control-plane tools:

| Tool | What it does |
|---|---|
| `wllm_inspect` | Discover project facts (entrypoints, model configs, checkpoints, GPUs) into an evidence-listing manifest; anything undetected is reported as UNKNOWN, never guessed. |
| `wllm_plan` | Rank capable backends and their legal optimization passes for a model under the active quality policy; every rejected pass carries a reason. Exit 3 means diagnose-only: nothing will be changed. |
| `wllm_verify` | Run the promote gate on a receipt: measured distributions present, authenticity proven, no forbidden-log hits, quality verdict compatible. |
| `wllm_apply` | Promote a receipt-backed plan (fail-closed; refuses with reasons when the gate blocks). |
| `wllm_rollback` | Walk the fallback chain one step (optimized -> last-known-good -> reference; reference never rolls away). |
| `wllm_report` | Current deploy state, active receipt, and recent apply/rollback history. |

## Boundary

The agent proposes; only evidence promotes. Baselines, benchmark
fixtures, verifiers, quality thresholds, and historical receipts stay
read-only to the agent — the tools refuse anything else.
