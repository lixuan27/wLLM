"""MCP stdio server: the one-sentence entry point for coding agents.

The agent is an untrusted proposer — every tool here dispatches into
the same fail-closed CLI logic that runs agent-free in CI. The server
adds transport, never judgment: no tool can relax a gate, edit a
receipt, or bypass the rollback chain.

Transport: newline-delimited JSON-RPC 2.0 over stdio (the MCP stdio
framing). Run with ``wllm-mcp`` or ``python -m wllm.control.mcp`` and
point a client at it, e.g.:

    {"mcpServers": {"wllm": {"command": "wllm-mcp"}}}
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

from .cli import main as cli_main

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "wllm", "version": "0.0.1a0"}

_STR = {"type": "string"}
_TOOLS: list[dict] = [
    {"name": "wllm_inspect",
     "description": "Discover project facts (entrypoints, model configs, "
                    "checkpoints, GPUs) into an evidence-listing manifest; "
                    "anything undetected is reported as UNKNOWN, never "
                    "guessed.",
     "inputSchema": {"type": "object",
                     "properties": {"root": _STR},
                     "required": ["root"]}},
    {"name": "wllm_plan",
     "description": "Rank capable backends and their legal optimization "
                    "passes for a model under the active quality policy; "
                    "every rejected pass carries a reason. Exit 3 means "
                    "diagnose-only: nothing will be changed.",
     "inputSchema": {"type": "object",
                     "properties": {"root": _STR, "model": _STR,
                                    "spec_path": _STR,
                                    "num_gpus": {"type": "integer"},
                                    "context_json": _STR},
                     "required": ["root", "model"]}},
    {"name": "wllm_verify",
     "description": "Run the promote gate on a receipt: measured "
                    "distributions present, authenticity proven, no "
                    "forbidden-log hits, quality verdict compatible.",
     "inputSchema": {"type": "object",
                     "properties": {"receipt": _STR,
                                    "quality_policy": _STR},
                     "required": ["receipt"]}},
    {"name": "wllm_apply",
     "description": "Promote a receipt-backed plan (fail-closed; refuses "
                    "with reasons when the gate blocks).",
     "inputSchema": {"type": "object",
                     "properties": {"root": _STR, "receipt": _STR,
                                    "quality_policy": _STR},
                     "required": ["root", "receipt"]}},
    {"name": "wllm_rollback",
     "description": "Walk the fallback chain one step (optimized -> "
                    "last-known-good -> reference; reference never rolls "
                    "away).",
     "inputSchema": {"type": "object",
                     "properties": {"root": _STR},
                     "required": ["root"]}},
    {"name": "wllm_report",
     "description": "Current deploy state, active receipt, and recent "
                    "apply/rollback history.",
     "inputSchema": {"type": "object",
                     "properties": {"root": _STR},
                     "required": ["root"]}},
]


def _argv_for(name: str, args: dict) -> list[str]:
    if name == "wllm_inspect":
        return ["inspect", args["root"], "--no-gpu-probe"]
    if name == "wllm_plan":
        argv = ["plan", args["root"], "--model", args["model"]]
        if args.get("spec_path"):
            argv += ["--spec", args["spec_path"]]
        if args.get("num_gpus"):
            argv += ["--num-gpus", str(int(args["num_gpus"]))]
        if args.get("context_json"):
            argv += ["--context", args["context_json"]]
        return argv
    if name == "wllm_verify":
        argv = ["verify", "--receipt", args["receipt"]]
        if args.get("quality_policy"):
            argv += ["--quality-policy", args["quality_policy"]]
        return argv
    if name == "wllm_apply":
        argv = ["apply", args["root"], "--receipt", args["receipt"]]
        if args.get("quality_policy"):
            argv += ["--quality-policy", args["quality_policy"]]
        return argv
    if name == "wllm_rollback":
        return ["rollback", args["root"]]
    if name == "wllm_report":
        return ["report", args["root"]]
    raise KeyError(f"unknown tool {name!r}")


def call_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run one tool; returns (text, is_error). Never raises to transport."""
    try:
        argv = _argv_for(name, args or {})
    except KeyError as exc:
        return str(exc), True
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = cli_main(argv)
    except SystemExit as exc:                 # argparse errors
        rc = int(exc.code or 2)
    except Exception as exc:  # noqa: BLE001 — surface, don't kill server
        buf.write(f"\n{type(exc).__name__}: {exc}")
        rc = 1
    text = buf.getvalue().rstrip() or "(no output)"
    text += f"\n[exit code {rc}]"
    # exit 3 (diagnose-only) is a truthful outcome, not a tool error
    return text, rc not in (0, 3)


def _response(req_id, result=None, error=None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message in, one response out (None for notifications)."""
    method = msg.get("method", "")
    req_id = msg.get("id")
    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        return _response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return _response(req_id, {})
    if method == "tools/list":
        return _response(req_id, {"tools": _TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        text, is_error = call_tool(str(params.get("name", "")),
                                   params.get("arguments") or {})
        return _response(req_id, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })
    if req_id is None:
        return None
    return _response(req_id, error={"code": -32601,
                                    "message": f"method not found: {method}"})


def serve(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_response(
                None, error={"code": -32700, "message": "parse error"}))
                + "\n")
            stdout.flush()
            continue
        resp = handle(msg)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
