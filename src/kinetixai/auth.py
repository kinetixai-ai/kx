"""Where the personal API token lives.

Order of preference:
  1. KX_TOKEN in the environment (CI, scripts; never written anywhere)
  2. the OS keyring (macOS Keychain, Windows Credential Manager, Secret
     Service on Linux) when the optional `keyring` package is installed and
     has a working backend
  3. credentials.json in the config directory, created owner-only (0600) on
     macOS/Linux; on Windows it sits in the per-user profile directory.

Tokens are stored per endpoint, so a localhost token never gets sent to
kinetixai.ai and vice versa. The token is never printed.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from . import config

SERVICE = "kinetixai"
TOKEN_RE = re.compile(r"^kx_[A-Za-z0-9_-]{20,128}$")


def valid(token: str) -> bool:
    return bool(TOKEN_RE.match((token or "").strip()))


def creds_path() -> Path:
    return config.config_dir() / "credentials.json"


def _keyring():
    if os.environ.get("KX_NO_KEYRING") == "1" or config.get("token_store") == "file":
        return None
    try:
        import keyring
        from keyring.backends import fail
    except Exception:
        return None
    try:
        kr = keyring.get_keyring()
        if isinstance(kr, fail.Keyring) or getattr(kr, "priority", 1) <= 0:
            return None
    except Exception:
        return None
    return keyring


def _read_file() -> dict:
    p = creds_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def _write_file(data: dict) -> None:
    p = creds_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o600)  # owner-only from the first byte on POSIX
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    if os.name != "nt":
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, p)


def store_name() -> str:
    return "keyring" if _keyring() else "file"


def save(endpoint: str, token: str) -> str:
    """Store the token for an endpoint. Returns where it went ('keyring' or a file path)."""
    endpoint = endpoint.rstrip("/")
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE, endpoint, token)
            _forget_file(endpoint)
            return "keyring"
        except Exception:
            pass  # fall through to the file
    data = _read_file()
    data[endpoint] = token
    _write_file(data)
    return str(creds_path())


def load(endpoint: str) -> str | None:
    env = os.environ.get("KX_TOKEN")
    if env:
        return env.strip()
    endpoint = endpoint.rstrip("/")
    kr = _keyring()
    if kr is not None:
        try:
            tok = kr.get_password(SERVICE, endpoint)
            if tok:
                return tok
        except Exception:
            pass
    return _read_file().get(endpoint)


def source(endpoint: str) -> str | None:
    """Where the active token comes from: 'env', 'keyring', 'file' or None."""
    if os.environ.get("KX_TOKEN"):
        return "env"
    endpoint = endpoint.rstrip("/")
    kr = _keyring()
    if kr is not None:
        try:
            if kr.get_password(SERVICE, endpoint):
                return "keyring"
        except Exception:
            pass
    return "file" if _read_file().get(endpoint) else None


def _forget_file(endpoint: str) -> None:
    data = _read_file()
    if endpoint in data:
        del data[endpoint]
        _write_file(data)


def delete(endpoint: str) -> bool:
    endpoint = endpoint.rstrip("/")
    found = False
    kr = _keyring()
    if kr is not None:
        try:
            if kr.get_password(SERVICE, endpoint):
                kr.delete_password(SERVICE, endpoint)
                found = True
        except Exception:
            pass
    if _read_file().get(endpoint):
        _forget_file(endpoint)
        found = True
    return found
