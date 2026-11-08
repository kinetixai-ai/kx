"""MCP smoke test used by CI: start `kx mcp` (or the given command), do the
stdio handshake, list tools and call the offline ones. Raw JSON-RPC, so it
does not depend on the server's SDK version.

    python packaging/mcp_smoke.py kx mcp
    python packaging/mcp_smoke.py ./dist/kx mcp
"""
import json, subprocess, sys, os

cmd = sys.argv[1:] or ["kx", "mcp"]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

def send(msg):
    p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()

def recv(want_id):
    while True:
        line = p.stdout.readline()
        if not line:
            raise SystemExit("server closed stdout; stderr:\n" + p.stderr.read())
        msg = json.loads(line)
        if msg.get("id") == want_id:
            return msg

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": "kx-e2e", "version": "0"}}})
init = recv(1)
print("initialize ->", init["result"]["serverInfo"], "protocol", init["result"]["protocolVersion"])
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
tools = recv(2)["result"]["tools"]
print("tools/list ->", [t["name"] for t in tools])
send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "library_search", "arguments": {"query": "attention", "limit": 5}}})
res = recv(3)["result"]
payload = res.get("structuredContent") or json.loads(res["content"][0]["text"])
print("tools/call library_search ->", [x["key"] for x in payload["papers"]], "isError", res.get("isError"))
send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "rag_ask", "arguments": {"question": "what does LoRA freeze?", "k": 2, "use_llm": False}}})
res = recv(4)["result"]
payload = res.get("structuredContent") or json.loads(res["content"][0]["text"])
print("tools/call rag_ask ->", [(x["n"], x["key"]) for x in payload["passages"]], "local", payload.get("local"))
expected = {"search_papers", "get_paper", "find_code", "find_models", "find_datasets", "library_search", "rag_ask"}
missing = expected - {t["name"] for t in tools}
if missing:
    raise SystemExit(f"missing tools: {sorted(missing)}")
p.stdin.close(); p.wait(timeout=10)
print("exit", p.returncode)
sys.exit(0)
