import json
import os
import stat
import sys

import httpx
import pytest
from typer.testing import CliRunner

from kinetixai import auth, cli, config
from kinetixai.api import ApiError, KinetixApi

runner = CliRunner()
TOKEN = "kx_" + "A" * 43
USER = {"id": "u1", "email": "rohit@example.com", "name": "Rohit", "plan": "free",
        "access": {"label": "Free early access"}}


def api_router(state):
    """A fake KinetixAI API that knows one valid token."""
    def handler(req):
        authz = req.headers.get("authorization")
        good = authz == f"Bearer {TOKEN}" and not state.get("revoked")
        if req.url.path == "/api/auth/me":
            return httpx.Response(200, json={"ok": True, "user": USER if good else None})
        if req.url.path == "/api/ask":
            body = json.loads(req.content)
            return httpx.Response(200, json={"ok": True, "answer": f"echo {body['question']}",
                                             "citations": [], "grounded": good})
        if req.url.path == "/api/auth/tokens" and req.method == "DELETE":
            if not good:
                return httpx.Response(401, json={"ok": False, "error": "Sign in to manage API tokens."})
            state["revoked"] = True
            return httpx.Response(200, json={"ok": True, "revoked": "t1"})
        return httpx.Response(404)
    return handler


def test_token_format():
    assert auth.valid(TOKEN)
    assert not auth.valid("kx_short") and not auth.valid("ghp_" + "a" * 40) and not auth.valid("")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_file_store_is_owner_only():
    auth.save("https://kinetixai.ai", TOKEN)
    mode = stat.S_IMODE(os.stat(auth.creds_path()).st_mode)
    assert mode == 0o600


def test_tokens_are_per_endpoint_and_deletable():
    auth.save("https://kinetixai.ai/", TOKEN)
    assert auth.load("https://kinetixai.ai") == TOKEN
    assert auth.load("http://localhost:8793") is None
    assert auth.source("https://kinetixai.ai") == "file"
    assert auth.delete("https://kinetixai.ai") is True
    assert auth.load("https://kinetixai.ai") is None and auth.delete("https://kinetixai.ai") is False


def test_env_token_wins(monkeypatch):
    monkeypatch.setenv("KX_TOKEN", TOKEN)
    assert auth.load("https://anything") == TOKEN and auth.source("https://anything") == "env"


def test_keyring_used_when_available(monkeypatch):
    store = {}

    class FakeKeyring:
        priority = 5

        def get_password(self, s, u):
            return store.get((s, u))

        def set_password(self, s, u, p):
            store[(s, u)] = p

        def delete_password(self, s, u):
            store.pop((s, u))

    fake = FakeKeyring()
    monkeypatch.delenv("KX_NO_KEYRING")
    monkeypatch.setattr(auth, "_keyring", lambda: fake)
    assert auth.save("https://kinetixai.ai", TOKEN) == "keyring"
    assert store[("kinetixai", "https://kinetixai.ai")] == TOKEN
    assert not auth.creds_path().exists()
    assert auth.load("https://kinetixai.ai") == TOKEN and auth.delete("https://kinetixai.ai")


def test_login_whoami_ask_logout_flow(mock_http):
    state = {}
    seen = mock_http(api_router(state))
    config.set_value("endpoint", "http://localhost:8793")

    res = runner.invoke(cli.app, ["login", "--token-stdin", "--json"], input=TOKEN + "\n")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["email"] == USER["email"] and out["store"] == "file"
    assert TOKEN not in res.stdout and TOKEN not in res.stderr  # never printed

    res = runner.invoke(cli.app, ["whoami"])
    assert res.exit_code == 0 and "rohit@example.com" in res.stdout and TOKEN not in res.stdout

    res = runner.invoke(cli.app, ["ask", "what is lora", "--json"])
    assert json.loads(res.stdout)["grounded"] is True
    assert seen[-1].headers["authorization"] == f"Bearer {TOKEN}"

    res = runner.invoke(cli.app, ["logout", "--json"])
    assert json.loads(res.stdout) == {"ok": True, "endpoint": "http://localhost:8793", "revoked": True, "removed": True}
    assert seen[-1].method == "DELETE" and seen[-1].url.params["id"] == "current"
    res = runner.invoke(cli.app, ["whoami", "--json"])
    assert res.exit_code == 1 and json.loads(res.stdout)["user"] is None


def test_login_rejects_bad_and_revoked_tokens(mock_http):
    state = {"revoked": True}
    mock_http(api_router(state))
    res = runner.invoke(cli.app, ["login", "--token-stdin"], input="not-a-token\n")
    assert res.exit_code == 1 and "kx_" in res.stderr
    res = runner.invoke(cli.app, ["login", "--token-stdin"], input=TOKEN + "\n")
    assert res.exit_code == 1 and "not accepted" in res.stderr
    assert auth.load("https://kinetixai.ai") is None


def test_login_from_env(monkeypatch, mock_http):
    mock_http(api_router({}))
    monkeypatch.setenv("KX_TOKEN", TOKEN)
    res = runner.invoke(cli.app, ["login"])
    assert res.exit_code == 0 and "rohit@example.com" in res.stdout


def test_whoami_with_revoked_token(mock_http):
    mock_http(api_router({"revoked": True}))
    auth.save("https://kinetixai.ai", TOKEN)
    res = runner.invoke(cli.app, ["whoami"])
    assert res.exit_code == 1 and "no longer valid" in res.stderr


def test_ask_without_login_sends_no_token(mock_http):
    seen = mock_http(api_router({}))
    res = runner.invoke(cli.app, ["ask", "hi"])
    assert res.exit_code == 0 and "authorization" not in seen[0].headers


def test_token_never_sent_over_plain_http_to_remote_hosts():
    with httpx.Client() as c:
        with pytest.raises(ApiError):
            KinetixApi("http://example.com", c, TOKEN)
        KinetixApi("http://127.0.0.1:8793", c, TOKEN)  # local dev is fine
        KinetixApi("https://kinetixai.ai", c, TOKEN)


def test_api_errors_are_readable(mock_http):
    mock_http(lambda r: httpx.Response(502, json={"ok": False, "error": "The model is busy right now."}))
    res = runner.invoke(cli.app, ["ask", "x"])
    assert res.exit_code == 1 and "model is busy" in res.stderr


def test_config_set_validates():
    assert runner.invoke(cli.app, ["config", "set", "endpoint", "kinetixai.ai"]).exit_code == 2
    assert runner.invoke(cli.app, ["config", "set", "limit", "ten"]).exit_code == 2
    res = runner.invoke(cli.app, ["config", "set", "endpoint", "http://localhost:8793/", "--json"])
    assert json.loads(res.stdout)["endpoint"] == "http://localhost:8793"
    shown = json.loads(runner.invoke(cli.app, ["config", "show", "--json"]).stdout)
    assert shown["settings"]["endpoint"] == "http://localhost:8793"
