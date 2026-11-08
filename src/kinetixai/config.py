"""User configuration and per-user directories.

Settings live in a small JSON file in the OS config directory; the paper
library and the local RAG index live in the OS data directory. Both come
from platformdirs, so they land in the right place on each OS:

  Linux    ~/.config/kinetixai          ~/.local/share/kinetixai
  macOS    ~/Library/Application Support/kinetixai (both)
  Windows  %LOCALAPPDATA%\\kinetixai (both)

KX_CONFIG_DIR and KX_DATA_DIR override them (handy for tests and for
keeping a project-specific library).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from platformdirs import user_config_dir, user_data_dir

APP = "kinetixai"
DEFAULTS: dict = {
    "endpoint": "https://kinetixai.ai",
    "limit": 10,
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1",
    "embed_model": "nomic-embed-text",
    # Primers: remembered so `kx primers build --topic X` needs no flags.
    "primer_engine": "auto",
    "primer_model": "",
    "primer_watermark": "Narmadya Innovations",
    # Where finished primers are copied in addition to --out-dir.
    "primer_copy_to": "",
}
INT_KEYS = {"limit"}
# Settings that may be set back to empty to turn them off.
BLANKABLE = {"primer_model", "primer_copy_to"}
URL_KEYS = {"endpoint", "ollama_url"}
TOKEN_STORES = {"auto", "file"}
KEYS = set(DEFAULTS) | {"token_store"}
_warned: set[str] = set()


def _warn(msg: str) -> None:
    """One warning per message per process, on stderr (never stdout: MCP uses it)."""
    if msg not in _warned:
        _warned.add(msg)
        print(f"warning: {msg}", file=sys.stderr)


def valid_url(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value)
        host = parts.hostname
        parts.port  # raises ValueError on a bad port
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(host) and not any(c.isspace() for c in value)


def validate(key: str, value):
    """Check a value for `kx config set` (and a stored value on load). Returns
    the cleaned value or raises ValueError with a message for the user."""
    if key not in KEYS:
        raise ValueError(f"unknown setting {key!r}. Settings: {', '.join(sorted(KEYS))}")
    if key in INT_KEYS:
        if isinstance(value, bool) or not str(value).isdigit() or not 1 <= int(value) <= 100:
            raise ValueError(f"{key} must be a whole number from 1 to 100.")
        return int(value)
    if key in BLANKABLE and (value is None or str(value).strip() == ""):
        return ""                       # clearing an optional setting
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be text.")
    value = value.strip()
    if key == "primer_engine":
        from .primers import providers        # cheap: no httpx at import time
        name = providers.resolve(value)
        if name != "auto" and name not in providers.PROVIDERS:
            raise ValueError(
                f"primer_engine must be 'auto' or one of: "
                f"{', '.join(providers.NAMES)}. Run `kx primers engines` to see "
                f"which are ready. Got {value!r}.")
        return name
    if key == "primer_copy_to":
        folder = Path(value).expanduser()
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"primer_copy_to must be a folder, and {folder} is a file.")
        return str(folder)
    if key in URL_KEYS:
        if not valid_url(value):
            raise ValueError(f"{key} must be an http(s) URL like https://kinetixai.ai, not {value!r}.")
        return value.rstrip("/")
    if key == "token_store" and value not in TOKEN_STORES:
        raise ValueError("token_store must be auto or file.")
    return value


def config_dir() -> Path:
    env = os.environ.get("KX_CONFIG_DIR")
    return Path(env) if env else Path(user_config_dir(APP, appauthor=False))


def data_dir() -> Path:
    env = os.environ.get("KX_DATA_DIR")
    return Path(env) if env else Path(user_data_dir(APP, appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.json"


def _stored() -> dict:
    path = config_path()
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        _warn(f"{path} is not valid JSON; using defaults.")
        return {}
    if not isinstance(data, dict):
        _warn(f"{path} should hold a JSON object; using defaults.")
        return {}
    return data


def load() -> dict:
    cfg = dict(DEFAULTS)
    for key, value in _stored().items():
        if key not in KEYS:
            continue  # unknown keys are kept in the file but ignored
        try:
            cfg[key] = validate(key, value)
        except ValueError as exc:
            _warn(f"ignoring {key} in {config_path()}: {exc} Using {DEFAULTS.get(key, 'the default')!r}.")
    # environment wins, so CI and scripts can override without writing a file
    env = os.environ.get("KINETIXAI_ENDPOINT")
    if env:
        if valid_url(env):
            cfg["endpoint"] = env.rstrip("/")
        else:
            _warn(f"ignoring KINETIXAI_ENDPOINT={env!r}: not an http(s) URL.")
    host = os.environ.get("OLLAMA_HOST")
    if host:
        url = normalise_ollama(host)
        if valid_url(url):
            cfg["ollama_url"] = url
        else:
            _warn(f"ignoring OLLAMA_HOST={host!r}: not a host or URL.")
    return cfg


def normalise_ollama(host: str) -> str:
    """OLLAMA_HOST may be '0.0.0.0', '127.0.0.1:11434' or a full URL."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    if rest.startswith("0.0.0.0"):
        rest = "localhost" + rest[len("0.0.0.0"):]
    if ":" not in rest.split("/", 1)[0]:
        rest = rest + ":11434"
    return f"{scheme}://{rest}"


def save(cfg: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(cfg, indent=2) + "\n")
    return path


def get(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> Path:
    stored = _stored()
    stored[key] = value
    return save(stored)


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file and os.replace, which is atomic on POSIX and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
