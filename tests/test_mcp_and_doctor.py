import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kinetixai import cli, mcp_server

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "papers.json"


@pytest.mark.parametrize("client", list(mcp_server.CLIENTS))
def test_every_client_has_a_snippet(client):
    res = runner.invoke(cli.app, ["mcp", "--config", client, "--json"])
    assert res.exit_code == 0, res.output
    snip = json.loads(res.stdout)
    assert "mcp" in snip["snippet"] and snip["file"]
    if snip["format"] == "json":
        cfg = json.loads(snip["snippet"])
        assert cfg["mcpServers"]["kinetixai"]["args"] == ["mcp", "serve"]


def test_windows_paths_are_escaped_in_every_format():
    cmd = r"C:\Users\Rohit\.local\bin\kx.exe"
    for client in mcp_server.CLIENTS:
        s = mcp_server.config_snippet(client, cmd)["snippet"]
        if s.lstrip().startswith("{"):
            assert json.loads(s)["mcpServers"]["kinetixai"]["command"] == cmd
        elif client not in ("open-webui", "claude-code"):
            assert json.dumps(cmd) in s  # double-quoted, backslashes escaped (YAML/TOML-safe)


def test_goose_snippet_is_valid_toml_and_yaml_shapes():
    s = mcp_server.config_snippet("openhands", "kx")["snippet"]
    if sys.version_info >= (3, 11):
        import tomllib
        assert tomllib.loads(s)["mcp"]["stdio_servers"][0]["command"] == "kx"


def test_unknown_client_is_usage_error():
    assert runner.invoke(cli.app, ["mcp", "--config", "nope"]).exit_code == 2


def test_mcp_info_json():
    data = json.loads(runner.invoke(cli.app, ["mcp", "--json"]).stdout)
    assert data["tools"] == mcp_server.TOOLS and data["transport"] == "stdio"


def _rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def _read(proc, want):
    while True:
        line = proc.stdout.readline()
        assert line, proc.stderr.read()
        msg = json.loads(line)
        if msg.get("id") == want:
            return msg


def test_stdio_handshake_list_and_call(tmp_path):
    """A real MCP session against `python -m kinetixai mcp` (offline tool only)."""
    pytest.importorskip("mcp")
    assert runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE)]).exit_code == 0
    import os
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, "-m", "kinetixai", "mcp"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        assert _read(proc, 1)["result"]["serverInfo"]["name"] == "kinetixai"
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in _read(proc, 2)["result"]["tools"]}
        assert names == set(mcp_server.TOOLS)
        _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "library_search", "arguments": {"query": "low rank"}}})
        result = _read(proc, 3)["result"]
        payload = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        assert payload["papers"][0]["key"] == "arxiv:2106.09685" and not result.get("isError")
    finally:
        proc.stdin.close()
        proc.wait(timeout=15)


def test_doctor_offline_json():
    res = runner.invoke(cli.app, ["doctor", "--offline", "--json"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.stdout)
    assert d["kx"] and d["python"]["ok"] is True and d["network"] == []
    assert set(d["extras"]) == {"rag", "mcp", "keyring"}
    assert d["ollama"]["running"] is False and any("Ollama" in a for a in d["advice"])
    assert d["paths"]["config"].endswith("config.json")


def test_doctor_human_output():
    res = runner.invoke(cli.app, ["doctor", "--offline"])
    assert res.exit_code == 0 and "Python" in res.stdout and "[rag] extra" in res.stdout


def test_version_flag_and_json():
    assert runner.invoke(cli.app, ["--version"]).stdout.startswith("kx ")
    assert json.loads(runner.invoke(cli.app, ["version", "--json"]).stdout)["version"]
