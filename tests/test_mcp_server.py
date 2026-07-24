"""MCP server: in-process handler tests + a subprocess stdio round-trip."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.mcp import _TOOLS, call_tool, handle
from wllm.control.receipt import Receipt


def _rpc(method, req_id=1, **params):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        msg["params"] = params
    return msg


def _fake_project(td: Path) -> Path:
    (td / "inference.py").write_text("print('run')\n")
    (td / "config.json").write_text(json.dumps(
        {"architectures": ["DemoDiT"], "_name_or_path": "org/demo"}))
    return td


def _good_receipt(path_dir: Path) -> Path:
    rec = Receipt(plan_id="mcp-plan", backend="wllm-serving",
                  source_revision="abc", hardware="1xTEST",
                  passes=["static_kv_cache"],
                  perf={"p50_ms": 100.0, "p95_ms": 120.0},
                  baseline_perf={"p50_ms": 200.0},
                  quality={"verdict": "exact"},
                  authenticity={"cache_active": True})
    return rec.save(path_dir)


# ------------------------------------------------------------ handler layer

def test_initialize_and_tools_list():
    init = handle(_rpc("initialize"))
    assert init["result"]["protocolVersion"]
    assert "tools" in init["result"]["capabilities"]
    listing = handle(_rpc("tools/list", req_id=2))
    names = {t["name"] for t in listing["result"]["tools"]}
    assert names == {"wllm_inspect", "wllm_plan", "wllm_verify",
                     "wllm_apply", "wllm_rollback", "wllm_report"}
    for t in _TOOLS:
        assert t["description"] and t["inputSchema"]["required"]


def test_notifications_and_unknown_methods():
    assert handle({"jsonrpc": "2.0",
                   "method": "notifications/initialized"}) is None
    err = handle(_rpc("no/such/method", req_id=3))
    assert err["error"]["code"] == -32601


def test_tool_flow_inspect_verify_apply_report_rollback():
    with tempfile.TemporaryDirectory() as td:
        proj = _fake_project(Path(td))
        text, is_err = call_tool("wllm_inspect", {"root": str(proj)})
        assert not is_err and "manifest ->" in text
        rpath = _good_receipt(proj / ".wllm" / "receipts")
        text, is_err = call_tool("wllm_verify", {"receipt": str(rpath)})
        assert not is_err and "promote gate: PASS" in text
        text, is_err = call_tool("wllm_apply", {"root": str(proj),
                                                "receipt": str(rpath)})
        assert not is_err and "active plan: mcp-plan" in text
        text, is_err = call_tool("wllm_report", {"root": str(proj)})
        assert not is_err and "mcp-plan" in text
        text, is_err = call_tool("wllm_rollback", {"root": str(proj)})
        assert not is_err and "reference" in text


def test_tool_failures_are_reported_not_hidden():
    with tempfile.TemporaryDirectory() as td:
        proj = _fake_project(Path(td))
        bad = Receipt(plan_id="bad", backend="wllm-serving",
                      perf={}, quality={"verdict": "exact"},
                      authenticity={"x": True})
        bpath = bad.save(proj / ".wllm" / "receipts")
        text, is_err = call_tool("wllm_verify", {"receipt": str(bpath)})
        assert is_err and "BLOCK" in text
        # diagnose-only (exit 3) is a truthful outcome, not a tool error
        text, is_err = call_tool("wllm_plan", {"root": str(proj),
                                               "model": "nobody/unknown"})
        assert not is_err and "diagnose-only" in text
        text, is_err = call_tool("mystery_tool", {})
        assert is_err


# ------------------------------------------------------- stdio round-trip

def test_stdio_subprocess_round_trip():
    with tempfile.TemporaryDirectory() as td:
        proj = _fake_project(Path(td))
        lines = "\n".join(json.dumps(m) for m in [
            _rpc("initialize", 1),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            _rpc("tools/list", 2),
            _rpc("tools/call", 3, name="wllm_inspect",
                 arguments={"root": str(proj)}),
        ]) + "\n"
        out = subprocess.run(
            [sys.executable, "-m", "wllm.control.mcp"],
            input=lines, capture_output=True, text=True, timeout=60,
            cwd=ROOT)
        responses = [json.loads(ln) for ln in out.stdout.splitlines() if ln]
        assert len(responses) == 3            # notification gets no reply
        by_id = {r["id"]: r for r in responses}
        assert by_id[1]["result"]["serverInfo"]["name"] == "wllm"
        assert len(by_id[2]["result"]["tools"]) == 6
        call = by_id[3]["result"]
        assert call["isError"] is False
        assert "manifest ->" in call["content"][0]["text"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:200]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
