"""The agent-facing package: manifests, entry point, and a live server.

Every other test in this repo exercises wLLM through Python imports.
An agent does not import anything — it loads a plugin manifest, launches
a declared command, and calls tools by name.  All of that can break
while the internal suite stays green: a renamed entry point, a skill
document naming a tool that no longer exists, a manifest whose version
drifts from the package.  These tests speak to the package the way an
agent does.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLUGIN = ROOT / "plugin"
SKILL = PLUGIN / "skills" / "optimize" / "SKILL.md"


def _pyproject_text() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _console_scripts() -> dict:
    """Parse ``[project.scripts]`` without adding a TOML dependency."""
    text = _pyproject_text()
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    out = {}
    for line in block.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, target = line.split("=", 1)
            out[name.strip()] = target.strip().strip('"')
    return out


# ------------------------------------------------------------- manifests

def test_plugin_manifest_is_well_formed():
    manifest = json.loads(
        (PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    for key in ("name", "version", "description"):
        assert manifest.get(key), f"plugin.json missing {key}"
    assert manifest["name"] == "wllm"


def test_plugin_version_tracks_the_package_version():
    """A plugin advertising a version the package does not have is a lie
    told to whoever is debugging a deployment."""
    manifest = json.loads(
        (PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    m = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject_text(),
                  re.MULTILINE)
    assert m, "pyproject has no version"
    assert manifest["version"] == m.group(1), (
        f"plugin.json {manifest['version']} != package {m.group(1)}")


def test_declared_mcp_command_is_a_real_entry_point():
    """The manifest's command must be installable, not aspirational."""
    mcp = json.loads((PLUGIN / ".mcp.json").read_text())
    servers = mcp.get("mcpServers", {})
    assert servers, ".mcp.json declares no servers"
    scripts = _console_scripts()
    for name, spec in servers.items():
        cmd = spec.get("command")
        assert cmd, f"server {name} declares no command"
        assert cmd in scripts, (
            f"server {name} launches {cmd!r}, which is not a console "
            f"script in pyproject ({sorted(scripts)})")
        module, _, func = scripts[cmd].partition(":")
        mod = __import__(module, fromlist=[func or "main"])
        assert callable(getattr(mod, func or "main"))


# ------------------------------------------------------------ skill text

def test_skill_only_names_tools_that_exist():
    """A skill document is the agent's API reference.

    If it names a tool the server does not serve, the agent's very first
    call fails — with the failure surfacing as "wLLM is broken" rather
    than "the doc drifted".
    """
    from wllm.control.mcp import _TOOLS

    served = {t["name"] for t in _TOOLS}
    named = set(re.findall(r"wllm_[a-z]+", SKILL.read_text()))
    assert named, "the skill document names no tools at all"
    missing = sorted(named - served)
    assert not missing, f"skill names tools the server does not serve: {missing}"


def test_skill_does_not_teach_hand_optimization():
    """The product law: the agent expresses intent, the infrastructure
    decides.  A skill that teaches flag-twiddling has inverted it."""
    text = SKILL.read_text().lower()
    for banned in ("torch.compile(", "cuda_graph", "set the batch size to",
                   "try fp8", "manually tune"):
        assert banned not in text, (
            f"skill teaches hand optimization ({banned!r}); it must teach "
            f"the agent to call the infrastructure instead")


# ------------------------------------------------------- live round trip

def test_declared_command_serves_a_real_session():
    """Launch what the manifest declares and speak to it as an agent.

    Falls back to ``python -m`` when the console script is not on PATH
    (a source checkout that was never pip-installed), because the point
    is the server contract, not the installer.
    """
    mcp = json.loads((PLUGIN / ".mcp.json").read_text())
    cmd = mcp["mcpServers"]["wllm"]["command"]
    exe = ROOT / ".venv" / "bin" / cmd
    argv = [str(exe)] if exe.exists() else [sys.executable, "-m",
                                            "wllm.control.mcp"]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    proc = subprocess.run(
        argv, input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True, text=True, timeout=90, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-800:]
    replies = [json.loads(line) for line in proc.stdout.splitlines()
               if line.strip()]
    assert len(replies) == 2, proc.stdout[:400]
    assert replies[0]["result"]["protocolVersion"]
    tools = {t["name"] for t in replies[1]["result"]["tools"]}
    # the tools an agent is told to call must be the tools it is served
    named = set(re.findall(r"wllm_[a-z]+", SKILL.read_text()))
    assert named <= tools, sorted(named - tools)


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
